"""Localised wet pockets, to give a tropism something it can actually follow."""
from dataclasses import dataclass, replace

import numpy as np

from grid import VoxelGrid
from soil.archetypes import AVAILABLE_WATER_RANGE
from soil.field import SoilField

FULL_RESERVE_FRACTION = AVAILABLE_WATER_RANGE[1]


@dataclass(frozen=True)
class WetSpot:
    name: str
    depth_cm: float
    bearing_degrees: float
    distance_cm: float
    radius_cm: float
    available_water_fraction: float = FULL_RESERVE_FRACTION


def build_wet_spot_mask(spot: WetSpot, grid: VoxelGrid) -> np.ndarray:
    depths, rows, columns = grid.measure_cell_centres_cm()
    bearing = np.radians(spot.bearing_degrees)
    centre_column = spot.distance_cm * np.cos(bearing)
    centre_row = spot.distance_cm * np.sin(bearing)
    squared = (np.square(columns - centre_column) + np.square(rows - centre_row)
               + np.square(depths + spot.depth_cm))
    return squared < spot.radius_cm ** 2


def apply_wet_spots(field: SoilField, spots: tuple[WetSpot, ...], grid: VoxelGrid) -> SoilField:
    fraction = field.available_water_fraction.copy()
    for spot in spots:
        fraction = np.where(build_wet_spot_mask(spot, grid), spot.available_water_fraction, fraction)
    return replace(field, available_water_fraction=fraction.astype(np.float32))


def build_wet_spot_field(spots: tuple[WetSpot, ...], grid: VoxelGrid) -> np.ndarray:
    """One labelled grid, so the pockets show up next to the roots in ParaView."""
    labels = np.zeros(grid.shape, dtype=np.float32)
    for label, spot in enumerate(spots, start=1):
        labels[build_wet_spot_mask(spot, grid)] = label
    return labels


def build_control_shell_mask(spot: WetSpot, grid: VoxelGrid) -> np.ndarray:
    depths = grid.measure_layer_depths_cm()[:, None, None]
    distances = grid.measure_trunk_distances_cm()[None, :, :]
    ring = ((np.abs(depths - spot.depth_cm) < spot.radius_cm)
            & (np.abs(distances - spot.distance_cm) < spot.radius_cm))
    return np.broadcast_to(ring, grid.shape) & ~build_wet_spot_mask(spot, grid)
