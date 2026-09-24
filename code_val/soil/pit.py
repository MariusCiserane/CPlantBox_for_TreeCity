"""The planting pit: a second soil zone, horizontal instead of layered.

A street tree is not planted in the ambient urban fill, it is planted in a
backfilled pit of amended substrate that the fill surrounds. Without it a hard
topsoil gives f_se = 0 right where the seed germinates and the tree is
stillborn, however good the subsoil below is.

The pit has a bottom: it is a backfilled excavation, so roots meet the ambient
soil underneath it. A pit reaching the whole grid depth would open a soft
chimney straight through every hard layer.
"""
import numpy as np

from grid import VoxelGrid
from soil.field import SoilField

TREE_PIT_DEPTH_RANGE_CM = (60.0, 100.0)


def build_pit_mask(pit_radius_cm: float, pit_depth_cm: float, grid: VoxelGrid) -> np.ndarray:
    """Depth-based like build_topsoil_mask, but also bounded by distance to the trunk."""
    distances = grid.measure_trunk_distances_cm()[None, :, :]
    depths = grid.measure_layer_depths_cm()[:, None, None]
    inside = (distances < pit_radius_cm) & (depths < pit_depth_cm)
    return np.broadcast_to(inside, grid.shape)


def apply_planting_pit(ambient: SoilField, pit: SoilField, mask: np.ndarray) -> SoilField:
    """Applied before any obstacle override: a utility crosses the pit like anything else."""
    return SoilField(
        penetrometer_mpa=np.where(mask, pit.penetrometer_mpa, ambient.penetrometer_mpa),
        bulk_density_g_cm3=np.where(mask, pit.bulk_density_g_cm3, ambient.bulk_density_g_cm3),
        available_water_fraction=np.where(mask, pit.available_water_fraction, ambient.available_water_fraction),
        sand_fraction=np.where(mask, pit.sand_fraction, ambient.sand_fraction),
        clay_fraction=np.where(mask, pit.clay_fraction, ambient.clay_fraction),
    )
