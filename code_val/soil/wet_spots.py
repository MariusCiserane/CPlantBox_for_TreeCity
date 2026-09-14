"""Localised wet pockets, to give a tropism something it can actually follow."""
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from grid import VoxelGrid
from soil.archetypes import MOISTURE_RANGE_M3_M3
from soil.field import SoilField

SATURATED_MOISTURE_M3_M3 = MOISTURE_RANGE_M3_M3[1]

# All eight sit inside the 254 cm that hold 99 % of the root mass, otherwise nothing ever
# reaches them. The tropism reads the continuous SDF field, not these voxels, so the radius no
# longer has to compensate for the cell size.
PROBE_RADIUS_CM = 45.0
ISOLATED_DISTANCE_CM = 180.0
ISOLATED_BEARINGS_DEGREES = (0.0, 90.0, 180.0, 270.0)
ISOLATED_DEPTHS_CM = (90.0, 135.0, 180.0, 225.0)
CLUSTER_BEARING_DEGREES = 45.0
CLUSTER_DISTANCES_CM = (200.0, 310.0)
CLUSTER_DEPTHS_CM = (85.0, 190.0)


@dataclass(frozen=True)
class WetSpot:
    name: str
    depth_cm: float
    bearing_degrees: float
    distance_cm: float
    radius_cm: float
    moisture_m3_m3: float = SATURATED_MOISTURE_M3_M3


def build_wet_spot_mask(spot: WetSpot, grid: VoxelGrid) -> np.ndarray:
    depths, rows, columns = grid.measure_cell_centres_cm()
    bearing = np.radians(spot.bearing_degrees)
    centre_column = spot.distance_cm * np.cos(bearing)
    centre_row = spot.distance_cm * np.sin(bearing)
    squared = (np.square(columns - centre_column) + np.square(rows - centre_row)
               + np.square(depths + spot.depth_cm))
    return squared < spot.radius_cm ** 2


def apply_wet_spots(field: SoilField, spots: tuple[WetSpot, ...], grid: VoxelGrid) -> SoilField:
    moisture = field.moisture_m3_m3.copy()
    for spot in spots:
        moisture = np.where(build_wet_spot_mask(spot, grid), spot.moisture_m3_m3, moisture)
    return replace(field, moisture_m3_m3=moisture.astype(np.float32))


def build_probe_wet_spots() -> tuple[WetSpot, ...]:
    """Four pockets far from each other and from the trunk, four packed together well away from
    those: a root that only reaches the cluster found water, one that reaches everything wandered."""
    isolated = [WetSpot(f"isolee_{bearing:.0f}", depth_cm=depth, bearing_degrees=bearing,
                        distance_cm=ISOLATED_DISTANCE_CM, radius_cm=PROBE_RADIUS_CM)
                for bearing, depth in zip(ISOLATED_BEARINGS_DEGREES, ISOLATED_DEPTHS_CM)]
    clustered = [WetSpot(f"grappe_{distance:.0f}_{depth:.0f}", depth_cm=depth,
                         bearing_degrees=CLUSTER_BEARING_DEGREES, distance_cm=distance,
                         radius_cm=PROBE_RADIUS_CM)
                 for distance, depth in product(CLUSTER_DISTANCES_CM, CLUSTER_DEPTHS_CM)]
    return tuple(isolated + clustered)


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
