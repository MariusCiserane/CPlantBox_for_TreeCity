"""Quelle part de chaque voxel un solide occupe, en fraction plutot qu'en oui ou non.

Un tuyau de 15 cm a une section de 177 cm2 quand une colonne de voxel en fait 135 : aucun voxel
n'est jamais plein, quelle que soit la pose. La fraction n'est donc pas un raffinement du
masque, c'est tout le signal.

Sur-echantillonner tout le domaine est hors de portee. Mais la distance signee au centre classe
deja une cellule comme entierement dedans ou entierement dehors des que son module depasse la
demi-diagonale : seule la coquille de bord est echantillonnee, et elle peut donc l'etre finement.

Un prisme - une tranchee, la fosse - se separe en une fraction de plan et une fraction de
profondeur, cette derniere exacte et la premiere calculee sur 64 x 64 colonnes au lieu de
139 264 voxels. Un tuyau oblique ne se separe pas et paie la coquille en trois dimensions.
"""
import numpy as np

from grid import VoxelGrid
from network.primitives import (PipeRun, PipeSegment, SURFACE_CM, measure_bottom_cm,
                                measure_pipe_bounds_cm, measure_plan_distance,
                                measure_signed_distance)

PLAN_SUBSAMPLES = 32
SHELL_SUBSAMPLES = (5, 8, 8)
SHELL_CHUNK_CELLS = 8192


def measure_cell_axes_cm(grid: VoxelGrid) -> tuple:
    """Centres de cellule sur chaque axe, dans l'ordre (profondeur, ligne, colonne)."""
    depths = -grid.measure_layer_depths_cm()
    rows = (np.arange(grid.row_cells) + 0.5) * grid.cell_width_cm - grid.half_length_cm
    columns = (np.arange(grid.column_cells) + 0.5) * grid.cell_width_cm - grid.half_width_cm
    return depths, rows, columns


def build_offsets(sizes: tuple, counts: tuple) -> np.ndarray:
    """Sous-centres d'une cellule, relatifs a son centre, dans l'ordre des axes donnes."""
    steps = [((np.arange(count) + 0.5) / count - 0.5) * size
             for count, size in zip(counts, sizes)]
    return np.stack([axis.ravel() for axis in np.meshgrid(*steps, indexing="ij")], axis=1)


def measure_fractions(measure_distance, centres: np.ndarray, offsets: np.ndarray,
                      half_diagonal_cm: float) -> np.ndarray:
    """Un par cellule pleine, zero par cellule vide, la coquille sur-echantillonnee."""
    distances = measure_distance(centres)
    fractions = np.where(distances <= -half_diagonal_cm, 1.0, 0.0)
    shell = np.nonzero(np.abs(distances) < half_diagonal_cm)[0]

    for first in range(0, len(shell), SHELL_CHUNK_CELLS):
        block = centres[shell[first:first + SHELL_CHUNK_CELLS]]
        points = (block[:, None, :] + offsets[None, :, :]).reshape(-1, len(offsets[0]))
        inside = measure_distance(points) < 0.0
        fractions[shell[first:first + len(block)]] = inside.reshape(len(block), -1).mean(axis=1)
    return fractions


def build_plan_fraction(shape, grid: VoxelGrid) -> np.ndarray:
    _, rows, columns = measure_cell_axes_cm(grid)
    row_mesh, column_mesh = np.meshgrid(rows, columns, indexing="ij")
    centres = np.stack([column_mesh.ravel(), row_mesh.ravel()], axis=1)

    offsets = build_offsets((grid.cell_width_cm, grid.cell_width_cm),
                            (PLAN_SUBSAMPLES, PLAN_SUBSAMPLES))[:, ::-1]
    half_diagonal = 0.5 * np.sqrt(2.0) * grid.cell_width_cm
    fractions = measure_fractions(lambda points: measure_plan_distance(shape, points),
                                  centres, offsets, half_diagonal)
    return fractions.reshape(grid.row_cells, grid.column_cells)


def measure_depth_fraction(bottom_cm: float, grid: VoxelGrid) -> np.ndarray:
    edges = -(np.arange(grid.depth_cells) + 1.0) * grid.cell_height_cm
    overlap = np.minimum(SURFACE_CM, edges + grid.cell_height_cm) - np.maximum(bottom_cm, edges)
    return np.clip(overlap, 0.0, grid.cell_height_cm) / grid.cell_height_cm


def build_prism_coverage(shape, grid: VoxelGrid) -> np.ndarray:
    depth_fraction = measure_depth_fraction(measure_bottom_cm(shape), grid)
    plan_fraction = build_plan_fraction(shape, grid)
    return (depth_fraction[:, None, None] * plan_fraction[None, :, :]).astype(np.float32)


def select_window(axis_cm: np.ndarray, low_cm: float, high_cm: float, margin_cm: float) -> slice:
    inside = np.nonzero((axis_cm >= low_cm - margin_cm) & (axis_cm <= high_cm + margin_cm))[0]
    if len(inside) == 0:
        return slice(0, 0)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def build_pipe_coverage(shape, grid: VoxelGrid) -> np.ndarray:
    depths, rows, columns = measure_cell_axes_cm(grid)
    half_diagonal = 0.5 * float(np.sqrt(2.0 * grid.cell_width_cm ** 2 + grid.cell_height_cm ** 2))
    (low_x, low_y, low_z), (high_x, high_y, high_z) = measure_pipe_bounds_cm(shape)

    window = (select_window(depths, low_z, high_z, half_diagonal),
              select_window(rows, low_y, high_y, half_diagonal),
              select_window(columns, low_x, high_x, half_diagonal))

    coverage = np.zeros(grid.shape, np.float32)
    depth_mesh, row_mesh, column_mesh = np.meshgrid(depths[window[0]], rows[window[1]],
                                                    columns[window[2]], indexing="ij")
    if depth_mesh.size == 0:
        return coverage

    centres = np.stack([column_mesh.ravel(), row_mesh.ravel(), depth_mesh.ravel()], axis=1)
    offsets = build_offsets((grid.cell_height_cm, grid.cell_width_cm, grid.cell_width_cm),
                            SHELL_SUBSAMPLES)[:, ::-1]
    fractions = measure_fractions(lambda points: measure_signed_distance(shape, points),
                                  centres, offsets, half_diagonal)

    coverage[window] = fractions.reshape(depth_mesh.shape)
    return coverage


def build_coverage(shape, grid: VoxelGrid) -> np.ndarray:
    if isinstance(shape, (PipeSegment, PipeRun)):
        return build_pipe_coverage(shape, grid)
    return build_prism_coverage(shape, grid)


def build_label_field(shapes: tuple, grid: VoxelGrid) -> np.ndarray:
    """Quel solide possede chaque voxel, numerote depuis 1, pour ParaView.
    Un voxel entame par deux solides revient a celui qui en prend le plus."""
    labels = np.zeros(grid.shape, np.float32)
    owned = np.zeros(grid.shape, np.float32)
    for index, shape in enumerate(shapes, start=1):
        coverage = build_coverage(shape, grid)
        takes = coverage > owned
        labels[takes] = index
        owned[takes] = coverage[takes]
    return labels
