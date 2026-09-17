"""A leaking pipe wetting the soil around it, with a gradient the tropism can actually follow.

Wet spots saturate a sphere and stop dead at its edge, which only offers a gradient right at the
boundary. A leak fades linearly from the leaking point out to its reach, so the field slopes
everywhere inside it.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from grid import VoxelGrid
from soil.archetypes import MOISTURE_RANGE_M3_M3
from soil.field import SoilField, apply_moisture_to_resistance
from soil.wet_spots import build_probe_wet_spots

SATURATED_MOISTURE_M3_M3 = MOISTURE_RANGE_M3_M3[1]
LEAK_REACH_CM = 75.0


@dataclass(frozen=True)
class Leak:
    name: str
    depth_cm: float
    bearing_degrees: float
    distance_cm: float
    reach_cm: float = LEAK_REACH_CM
    moisture_m3_m3: float = SATURATED_MOISTURE_M3_M3


def build_probe_leaks() -> tuple[Leak, ...]:
    """Leaks sit where the probe spots sit, so the SDF control run and the interpolated field
    describe water in the same places."""
    return tuple(Leak(spot.name, depth_cm=spot.depth_cm, bearing_degrees=spot.bearing_degrees,
                      distance_cm=spot.distance_cm)
                 for spot in build_probe_wet_spots())


def build_leak_ramp(leak: Leak, grid: VoxelGrid) -> np.ndarray:
    """Share of the leak at each voxel: 1 at the leaking point, fading to 0 at its reach."""
    depths, rows, columns = grid.measure_cell_centres_cm()
    bearing = math.radians(leak.bearing_degrees)
    distances = np.sqrt(np.square(columns - leak.distance_cm * math.cos(bearing))
                        + np.square(rows - leak.distance_cm * math.sin(bearing))
                        + np.square(depths + leak.depth_cm))
    return np.clip(1.0 - distances / leak.reach_cm, 0.0, 1.0)


def build_leak_field(leaks: tuple[Leak, ...], grid: VoxelGrid) -> np.ndarray:
    """The wetted zone as it actually is: the strongest ramp at each voxel, 1 at a leaking point
    and 0 beyond every reach. This is what belongs next to the roots in ParaView - the wet spot
    masks describe a different, smaller shape."""
    ramps = [build_leak_ramp(leak, grid) for leak in leaks]
    return np.max(ramps, axis=0).astype(np.float32) if ramps else np.zeros(grid.shape, np.float32)


def apply_leaks(field: SoilField, leaks: tuple[Leak, ...], grid: VoxelGrid) -> SoilField:
    """Blending towards the ambient soil rather than overwriting it keeps the ramp sloping all the
    way out to the reach; taking a maximum instead would flatten it wherever the soil is already
    wetter than the tail of the ramp, and a flat field steers nothing.

    A leak is the one source that moves water without touching anything else, so the resistance
    has to be rescaled here: wet soil is softer, and leaving it hard would let a leak reach the
    water half of f_se while f_sa and the mechanical half never noticed."""
    moisture = field.moisture_m3_m3
    for leak in leaks:
        moisture = moisture + (leak.moisture_m3_m3 - moisture) * build_leak_ramp(leak, grid)
    wetted = replace(field, moisture_m3_m3=moisture.astype(np.float32))
    return apply_moisture_to_resistance(wetted, field.moisture_m3_m3)
