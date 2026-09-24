"""Des polylignes aux grilles d'occupation.

Deux cibles. La ligne mediane, celle de la v1 : un voxel est pris si l'axe d'une racine le
traverse. Et l'occupation reelle, qui tient compte du rayon apres croissance secondaire : un
voxel est pris des qu'une racine assez epaisse passe assez pres pour l'entamer.

Le test est exact - une sphere de rayon r centree sur l'echantillon coupe-t-elle la boite du
voxel - donc une racine posee sur une face deborde chez le voisin et la meme racine au centre
ne deborde pas. Une dilatation morphologique par classe de rayon gonflerait au contraire tout
voxel marque de la meme facon, apres avoir jete la position sous-voxel.

Le volume est accumule sur toutes les racines, fines comprises : c'est un canal de biomasse, et
les racines fines en portent.
"""
from typing import NamedTuple

import numpy as np

from dataset.occupancy import STRUCTURAL_SUBTYPES
from generator.polylines import ROOT_ORGAN_TYPE
from grid import VoxelGrid

SAMPLE_STEP_CM = 1.0
RADIUS_BIN_EDGES_CM = (0.0, 0.75, 2.0, 4.0, 8.0, 15.0)


class Polysegments(NamedTuple):
    starts: np.ndarray
    ends: np.ndarray
    start_radius: np.ndarray
    end_radius: np.ndarray
    subtype: np.ndarray


class Samples(NamedTuple):
    points: np.ndarray
    radii: np.ndarray
    length_cm: np.ndarray


def build_polysegments(polydata, node_radii: np.ndarray) -> Polysegments:
    """Les paires de noeuds consecutifs de chaque polyligne, racines seulement."""
    sizes = np.diff(polydata.offsets, prepend=0)
    keep = np.repeat(polydata.cell_arrays["organType"] == ROOT_ORGAN_TYPE, sizes)
    last = np.zeros(len(polydata.points), bool)
    last[polydata.offsets - 1] = True
    starts = np.nonzero(keep & ~last)[0]

    return Polysegments(polydata.points[starts], polydata.points[starts + 1],
                        node_radii[starts], node_radii[starts + 1],
                        np.repeat(polydata.cell_arrays["subType"], sizes)[starts].astype(int))


def select_structural(segments: Polysegments) -> Polysegments:
    keep = np.isin(segments.subtype, list(STRUCTURAL_SUBTYPES))
    return Polysegments(*(field[keep] for field in segments))


def sample_segments(segments: Polysegments, step_cm: float = SAMPLE_STEP_CM) -> Samples:
    lengths = np.linalg.norm(segments.ends - segments.starts, axis=1)
    counts = np.maximum(2, np.ceil(lengths / step_cm).astype(int) + 1)
    index = np.repeat(np.arange(len(counts)), counts)
    position = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    fraction = position / np.repeat(counts - 1, counts)

    points = segments.starts[index] + fraction[:, None] * (segments.ends - segments.starts)[index]
    radii = segments.start_radius[index] + fraction * (segments.end_radius - segments.start_radius)[index]
    return Samples(points, radii, lengths[index] / np.repeat(counts, counts))


def build_offset_table(radius_cm: float, grid: VoxelGrid) -> np.ndarray:
    reach = (int(np.ceil(radius_cm / grid.cell_height_cm)), int(np.ceil(radius_cm / grid.cell_width_cm)),
             int(np.ceil(radius_cm / grid.cell_width_cm)))
    ranges = [np.arange(-count, count + 1) for count in reach]
    return np.stack([axis.ravel() for axis in np.meshgrid(*ranges, indexing="ij")], axis=1)


def measure_cell_offsets_cm(points: np.ndarray, indices: tuple, grid: VoxelGrid) -> np.ndarray:
    """Position de chaque echantillon dans son propre voxel, en (x, y, z)."""
    depth, row, column = indices
    return points - np.stack([
        (column + 0.5) * grid.cell_width_cm - grid.half_width_cm,
        (row + 0.5) * grid.cell_width_cm - grid.half_length_cm,
        -(depth + 0.5) * grid.cell_height_cm], axis=1)


def stamp_centreline(samples: Samples, grid: VoxelGrid) -> tuple:
    depth, row, column, inside = grid.locate_points(samples.points)
    occupancy = np.zeros(grid.shape, bool)
    occupancy[depth[inside], row[inside], column[inside]] = True
    return occupancy, int((~inside).sum())


def stamp_radius(samples: Samples, grid: VoxelGrid) -> np.ndarray:
    depth, row, column, inside = grid.locate_points(samples.points)
    offsets_cm = measure_cell_offsets_cm(samples.points, (depth, row, column), grid)
    half = np.array([0.5 * grid.cell_width_cm, 0.5 * grid.cell_width_cm, 0.5 * grid.cell_height_cm])
    sizes = np.array([grid.cell_width_cm, grid.cell_width_cm, grid.cell_height_cm])

    occupancy = np.zeros(grid.shape, bool)
    for low, high in zip(RADIUS_BIN_EDGES_CM[:-1], RADIUS_BIN_EDGES_CM[1:]):
        chosen = inside & (samples.radii > low) & (samples.radii <= high)
        if not chosen.any():
            continue
        for step in build_offset_table(high, grid):
            centre = np.array([step[2], step[1], -step[0]]) * sizes
            gap = np.linalg.norm(np.maximum(np.abs(centre - offsets_cm[chosen]) - half, 0.0), axis=1)
            reached = gap <= samples.radii[chosen]
            neighbour = (depth[chosen] + step[0], row[chosen] + step[1], column[chosen] + step[2])
            within = ((neighbour[0] >= 0) & (neighbour[0] < grid.depth_cells)
                      & (neighbour[1] >= 0) & (neighbour[1] < grid.row_cells)
                      & (neighbour[2] >= 0) & (neighbour[2] < grid.column_cells) & reached)
            occupancy[neighbour[0][within], neighbour[1][within], neighbour[2][within]] = True
    return occupancy


def accumulate_volume(samples: Samples, grid: VoxelGrid) -> np.ndarray:
    depth, row, column, inside = grid.locate_points(samples.points)
    volume = np.zeros(grid.shape, np.float64)
    contribution = np.pi * np.square(samples.radii) * samples.length_cm
    np.add.at(volume, (depth[inside], row[inside], column[inside]), contribution[inside])
    cell_volume = grid.cell_width_cm ** 2 * grid.cell_height_cm
    return (volume / cell_volume).astype(np.float32)
