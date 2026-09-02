"""Voxel masks and the obstacle distance channel.

Computed analytically from the layout rather than sampled from the SDF: exact,
and independent of how the geometry happens to be composed.
"""
from dataclasses import dataclass

import numpy as np

from grid import VoxelGrid
from obstacles.layouts import ALONG_Y, SIDEWALK, ObstacleLayout, Pavement, Pipe

OBSTACLE_DISTANCE_CLIP_CM = 1000.0


@dataclass(frozen=True)
class ObstacleMasks:
    pipe: np.ndarray
    pavement: np.ndarray
    road_base: np.ndarray
    distance_to_obstacle_cm: np.ndarray


def measure_offsets_to_pipe_axis(pipe: Pipe, grid: VoxelGrid):
    """Signed distances from every cell centre to the pipe axis, across and vertically."""
    depth_cm, row_cm, column_cm = grid.measure_cell_centres_cm()
    across = column_cm - pipe.offset_cm if pipe.axis == ALONG_Y else row_cm - pipe.offset_cm
    return across, depth_cm + pipe.depth_cm


def measure_distance_to_pipe_axis(pipe: Pipe, grid: VoxelGrid) -> np.ndarray:
    across, vertical = measure_offsets_to_pipe_axis(pipe, grid)
    return np.hypot(across, vertical)


def build_pipe_mask(pipe: Pipe, grid: VoxelGrid) -> np.ndarray:
    """A cell belongs to the pipe as soon as the cylinder passes through it.

    Testing the cell centre alone would be wrong here: pipes are 15 to 40 cm across
    while cells are 30 cm wide, so a pipe threading between two centres would be
    marked one cell thick, or not at all. This measures the distance from the axis
    to the cell box instead.
    """
    across, vertical = measure_offsets_to_pipe_axis(pipe, grid)
    horizontal_gap = np.maximum(np.abs(across) - 0.5 * grid.cell_width_cm, 0.0)
    vertical_gap = np.maximum(np.abs(vertical) - 0.5 * grid.cell_height_cm, 0.0)
    return np.hypot(horizontal_gap, vertical_gap) <= pipe.radius_cm


def build_pavement_footprint(pavement: Pavement, grid: VoxelGrid) -> np.ndarray:
    _, row_cm, column_cm = grid.measure_cell_centres_cm()
    if pavement.style == SIDEWALK:
        return column_cm >= pavement.boundary_cm
    inside_pit = (np.abs(column_cm) <= pavement.boundary_cm) & (np.abs(row_cm) <= pavement.boundary_cm)
    return ~inside_pit


def build_obstacle_masks(layout: ObstacleLayout, grid: VoxelGrid) -> ObstacleMasks:
    pipe_mask = np.zeros(grid.shape, dtype=bool)
    clearance = np.full(grid.shape, OBSTACLE_DISTANCE_CLIP_CM)
    for pipe in layout.pipes:
        pipe_mask |= build_pipe_mask(pipe, grid)
        clearance = np.minimum(clearance, measure_distance_to_pipe_axis(pipe, grid) - pipe.radius_cm)
    if layout.pipes and not pipe_mask.any():
        raise RuntimeError(f"empty pipe mask for layout {layout.family}")

    pavement_mask = np.zeros(grid.shape, dtype=bool)
    road_base_mask = np.zeros(grid.shape, dtype=bool)
    if layout.pavement is not None:
        footprint = build_pavement_footprint(layout.pavement, grid)
        depth_below_surface = grid.measure_layer_depths_cm()[:, None, None]
        pavement_mask = footprint & (depth_below_surface < layout.pavement.slab_depth_cm)
        road_base_mask = footprint & (depth_below_surface >= layout.pavement.slab_depth_cm) \
            & (depth_below_surface < layout.pavement.road_base_depth_cm)

    return ObstacleMasks(
        pipe=pipe_mask,
        pavement=pavement_mask,
        road_base=road_base_mask,
        distance_to_obstacle_cm=np.clip(clearance, 0.0, OBSTACLE_DISTANCE_CLIP_CM).astype(np.float32),
    )
