"""How much of each voxel a solid occupies, as a fraction rather than a yes or no.

A 15 cm pipe has a 177 cm2 cross section and a voxel column is 135 cm2, so a pipe never
fills a single voxel completely however it is placed: every voxel it touches is partial.
The fraction is therefore not a refinement of the mask, it is the whole signal, and v1's
binary test could only ever draw a staircase.

Axis-aligned boxes separate exactly into three one-dimensional overlaps, so they cost
nothing and carry no quadrature error. A horizontal cylinder spans the domain along its
axis, which leaves a two-dimensional cross section to sample - cheap enough to oversample
heavily instead of settling for a coarse grid that would reintroduce the staircase.
"""
import numpy as np

from grid import VoxelGrid
from obstacles.layouts import ALONG_Y
from obstacles.shapes import Cuboid, Cylinder

CROSS_SECTION_SUBSAMPLES = 32


def measure_cell_edges_cm(grid: VoxelGrid):
    """Lower edge of every cell on each axis, in world coordinates, plus the cell size."""
    depths = -(np.arange(grid.depth_cells) + 1.0) * grid.cell_height_cm
    rows = np.arange(grid.row_cells) * grid.cell_width_cm - grid.half_length_cm
    columns = np.arange(grid.column_cells) * grid.cell_width_cm - grid.half_width_cm
    return (depths, grid.cell_height_cm), (rows, grid.cell_width_cm), (columns, grid.cell_width_cm)


def measure_axis_coverage(lower_cm: float, upper_cm: float, edges_cm: np.ndarray,
                          cell_size_cm: float) -> np.ndarray:
    overlap = np.minimum(upper_cm, edges_cm + cell_size_cm) - np.maximum(lower_cm, edges_cm)
    return np.clip(overlap, 0.0, cell_size_cm) / cell_size_cm


def build_cuboid_coverage(cuboid: Cuboid, grid: VoxelGrid) -> np.ndarray:
    (depths, depth_size), (rows, row_size), (columns, column_size) = measure_cell_edges_cm(grid)
    along_depth = measure_axis_coverage(cuboid.minimum_cm[2], cuboid.maximum_cm[2], depths, depth_size)
    along_row = measure_axis_coverage(cuboid.minimum_cm[1], cuboid.maximum_cm[1], rows, row_size)
    along_column = measure_axis_coverage(cuboid.minimum_cm[0], cuboid.maximum_cm[0], columns, column_size)
    product = along_depth[:, None, None] * along_row[None, :, None] * along_column[None, None, :]
    return product.astype(np.float32)


def build_cross_section_coverage(cylinder: Cylinder, depth_edges, across_edges) -> np.ndarray:
    """Oversampled disc in the plane cutting the pipe, of shape (depth, across)."""
    subsamples = CROSS_SECTION_SUBSAMPLES
    step = (np.arange(subsamples) + 0.5) / subsamples
    verticals = (depth_edges[0][:, None] + step[None, :] * depth_edges[1]).ravel()
    across = (across_edges[0][:, None] + step[None, :] * across_edges[1]).ravel()
    inside = np.hypot(across[None, :] - cylinder.offset_cm,
                      verticals[:, None] + cylinder.depth_cm) <= cylinder.radius_cm
    cells = inside.reshape(len(depth_edges[0]), subsamples, len(across_edges[0]), subsamples)
    return cells.mean(axis=(1, 3))


def build_cylinder_coverage(cylinder: Cylinder, grid: VoxelGrid) -> np.ndarray:
    """Pipes cross the whole domain along their axis, so that axis contributes a full cell."""
    depth_edges, row_edges, column_edges = measure_cell_edges_cm(grid)
    if cylinder.axis == ALONG_Y:
        section = build_cross_section_coverage(cylinder, depth_edges, column_edges)
        return np.broadcast_to(section[:, None, :], grid.shape).astype(np.float32)
    section = build_cross_section_coverage(cylinder, depth_edges, row_edges)
    return np.broadcast_to(section[:, :, None], grid.shape).astype(np.float32)


def build_coverage(shape: Cylinder | Cuboid, grid: VoxelGrid) -> np.ndarray:
    if isinstance(shape, Cylinder):
        return build_cylinder_coverage(shape, grid)
    return build_cuboid_coverage(shape, grid)


def build_label_field(shapes: tuple[Cylinder | Cuboid, ...], grid: VoxelGrid) -> np.ndarray:
    """Which solid owns each voxel, numbered from 1, so ParaView can tell them apart.
    A voxel clipped by two solids goes to whichever covers more of it."""
    labels = np.zeros(grid.shape, np.float32)
    owned = np.zeros(grid.shape, np.float32)
    for index, shape in enumerate(shapes, start=1):
        coverage = build_coverage(shape, grid)
        takes = coverage > owned
        labels[takes] = index
        owned[takes] = coverage[takes]
    return labels
