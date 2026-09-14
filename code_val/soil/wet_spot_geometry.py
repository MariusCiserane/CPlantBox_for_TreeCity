"""Analytical water field for the hydrotropism, kept as the control run.

InterpolatedGrid3D now feeds the tropism the real van Genuchten field, which is what the
simulations use. This stays because it describes the same places with an exact analytical ramp:
if both produce the same bias, the interpolated grid is being read correctly.
"""
import math

import plantbox as pb

from soil.wet_spots import WetSpot

ATTRACTION_RANGE_CM = 150.0
SLOPE_PER_RANGE = 4.0  # SoilLookUpSDF spreads its ramp over slope/4 either side of the boundary
SATURATED_WATER_SCALE = 1.0
DRY_WATER_SCALE = 0.0
ROUND_CROSS_SECTION = 0.0


def build_wet_spot_solid(spot: WetSpot) -> pb.SignedDistanceFunction:
    """A cylinder standing on its end stands in for the sphere of the voxel mask: CPlantBox has
    no sphere primitive, and only the distance falloff reaches the tropism. SDF_PlantContainer
    hangs from z=0 down to -h, so its top goes one radius above the pocket centre."""
    bearing = math.radians(spot.bearing_degrees)
    top = pb.Vector3d(spot.distance_cm * math.cos(bearing),
                      spot.distance_cm * math.sin(bearing),
                      -spot.depth_cm + spot.radius_cm)
    cylinder = pb.SDF_PlantContainer(spot.radius_cm, spot.radius_cm,
                                     2.0 * spot.radius_cm, ROUND_CROSS_SECTION)
    return pb.SDF_RotateTranslate(cylinder, top)


def build_wet_spot_lookup(spots: tuple[WetSpot, ...],
                          attraction_range_cm: float = ATTRACTION_RANGE_CM) -> pb.SoilLookUpSDF:
    solids = [build_wet_spot_solid(spot) for spot in spots]
    return pb.SoilLookUpSDF(pb.SDF_Union(solids), SATURATED_WATER_SCALE, DRY_WATER_SCALE,
                            SLOPE_PER_RANGE * attraction_range_cm)
