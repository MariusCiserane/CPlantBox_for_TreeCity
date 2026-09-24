"""The buried networks a roadside tree actually grows among, and how they are laid.

The pavement itself is not modelled: what reaches the roots is the excavation, not the slab
above it. The curb survives only as a coordinate - it is the line the parallel networks are
offset from, which is what keeps telecom, electricity, gas and water on one side and the
sewer on the other.
"""
from dataclasses import dataclass

import numpy as np

from obstacles.layouts import ALONG_X, ALONG_Y
from soil.archetypes import SoilHorizon

CURB_DISTANCE_RANGE_CM = (150.0, 280.0)

TRENCH_WORKING_SPACE_CM = 30.0
BEDDING_DEPTH_RANGE_CM = (10.0, 15.0)
SURROUND_COVER_RANGE_CM = (15.0, 30.0)
CROSSING_OFFSET_RANGE_CM = (100.0, 400.0)

RECOMPACTED_BACKFILL = SoilHorizon(3.0, 1.95, 0.25, 0.82, 0.08)
LOOSE_BACKFILL = SoilHorizon(0.7, 1.30, 0.50, 0.55, 0.15)
PIPE_SURROUND = SoilHorizon(0.5, 1.50, 0.20, 0.95, 0.01)

# LeakageStatusValue of the Team 3 CityGML utilities package.
NO_LEAK = "none"
MINOR_LEAK = "minor"
MAJOR_LEAK = "major"
LEAKAGE_STATUSES = (NO_LEAK, MINOR_LEAK, MAJOR_LEAK)
LEAKAGE_WEIGHTS = (0.34, 0.33, 0.33)


@dataclass(frozen=True)
class NetworkTemplate:
    name: str
    axis: str
    diameter_range_cm: tuple[float, float]
    crown_depth_range_cm: tuple[float, float]
    offset_range_cm: tuple[float, float]
    joint_spacing_cm: float

#
# Offsets are measured from the curb for the street-parallel networks, and from the trunkfor the crossing connection.
NETWORK_TEMPLATES = (
    NetworkTemplate("telecom", ALONG_Y, (10.0, 14.0), (40.0, 70.0), (-90.0, -30.0), 0.0),
    NetworkTemplate("electricity", ALONG_Y, (14.0, 18.0), (60.0, 85.0), (-140.0, -80.0), 0.0),
    NetworkTemplate("gas", ALONG_Y, (9.0, 13.0), (70.0, 100.0), (-190.0, -130.0), 0.0),
    NetworkTemplate("water", ALONG_Y, (12.0, 25.0), (90.0, 130.0), (-250.0, -170.0), 600.0),
    NetworkTemplate("sewer", ALONG_Y, (40.0, 80.0), (180.0, 300.0), (100.0, 170.0), 250.0),
    NetworkTemplate("sewer_connection", ALONG_X, (12.0, 20.0), (100.0, 200.0), CROSSING_OFFSET_RANGE_CM, 250.0),
)


@dataclass(frozen=True)
class Utility:
    name: str
    axis: str
    offset_cm: float
    depth_cm: float
    radius_cm: float
    trench_half_width_cm: float
    bedding_depth_cm: float
    surround_cover_cm: float
    joint_spacing_cm: float
    leakage_status: str


@dataclass(frozen=True)
class StreetScene:
    curb_x_cm: float
    pit_radius_cm: float
    pit_depth_cm: float
    utilities: tuple[Utility, ...]


def draw_utility(template: NetworkTemplate, curb_x_cm: float, domain_half_extent_cm: float,
                 rng: np.random.Generator) -> Utility:
    radius_cm = 0.5 * float(rng.uniform(*template.diameter_range_cm))
    trench_half_width_cm = radius_cm + TRENCH_WORKING_SPACE_CM
    if template.axis == ALONG_Y:
        offset_cm = curb_x_cm + float(rng.uniform(*template.offset_range_cm))
    else:
        side = -1.0 if rng.random() < 0.5 else 1.0
        offset_cm = side * float(rng.uniform(*template.offset_range_cm))
    # A network whose trench falls outside the scene teaches the network nothing, so it is held
    # inside it; the ranges are chosen so this rarely bites.
    limit_cm = domain_half_extent_cm - trench_half_width_cm
    offset_cm = min(max(offset_cm, -limit_cm), limit_cm)
    leaks = template.joint_spacing_cm > 0.0
    return Utility(
        name=template.name,
        axis=template.axis,
        offset_cm=offset_cm,
        depth_cm=float(rng.uniform(*template.crown_depth_range_cm)) + radius_cm,
        radius_cm=radius_cm,
        trench_half_width_cm=trench_half_width_cm,
        bedding_depth_cm=float(rng.uniform(*BEDDING_DEPTH_RANGE_CM)),
        surround_cover_cm=float(rng.uniform(*SURROUND_COVER_RANGE_CM)),
        joint_spacing_cm=template.joint_spacing_cm,
        leakage_status=str(rng.choice(LEAKAGE_STATUSES, p=LEAKAGE_WEIGHTS)) if leaks else NO_LEAK,
    )


def draw_street_scene(pit_radius_cm: float, pit_depth_cm: float, domain_half_extent_cm: float, rng: np.random.Generator) -> StreetScene:
    curb_x_cm = float(rng.uniform(*CURB_DISTANCE_RANGE_CM))
    utilities = tuple(draw_utility(template, curb_x_cm, domain_half_extent_cm, rng) for template in NETWORK_TEMPLATES)
    return StreetScene(curb_x_cm, pit_radius_cm, pit_depth_cm, utilities)
