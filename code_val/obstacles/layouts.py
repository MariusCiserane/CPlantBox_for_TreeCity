"""The four obstacle families of the specification, and how to draw one.

A family fixes what is present; the numbers are drawn per configuration so that
no two layouts are identical.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from soil.field import RoadBase

PIPE_DEPTH_RANGE_CM = (60.0, 150.0)
PIPE_DIAMETER_RANGE_CM = (15.0, 40.0)
PIPE_OFFSET_RANGE_CM = (100.0, 400.0)

SLAB_DEPTH_CM = 15.0
ROAD_BASE_DEPTH_CM = 50.0
ROAD_BASE = RoadBase(penetrometer_mpa=3.5, bulk_density_g_cm3=1.90, available_water_fraction=0.15)
SIDEWALK_START_RANGE_CM = (100.0, 400.0)
TREE_PIT_HALF_SIZE_RANGE_CM = (50.0, 150.0)

ALONG_X = "x"
ALONG_Y = "y"
SIDEWALK = "sidewalk"
TREE_PIT = "tree_pit"

NO_OBSTACLE = "none"
SINGLE_PIPE = "single_pipe"
CROSSING_PIPES = "crossing_pipes"
PIPE_AND_PAVEMENT = "pipe_and_pavement"
OBSTACLE_FAMILIES = (NO_OBSTACLE, SINGLE_PIPE, CROSSING_PIPES, PIPE_AND_PAVEMENT)


@dataclass(frozen=True)
class Pipe:
    axis: str
    offset_cm: float
    depth_cm: float
    radius_cm: float


@dataclass(frozen=True)
class Pavement:
    style: str
    boundary_cm: float
    slab_depth_cm: float = SLAB_DEPTH_CM
    road_base_depth_cm: float = ROAD_BASE_DEPTH_CM


@dataclass(frozen=True)
class ObstacleLayout:
    family: str
    pipes: tuple[Pipe, ...] = ()
    pavement: Optional[Pavement] = None


def draw_pipe(axis: str, rng: np.random.Generator) -> Pipe:
    side = -1.0 if rng.random() < 0.5 else 1.0
    return Pipe(
        axis=axis,
        offset_cm=side * float(rng.uniform(*PIPE_OFFSET_RANGE_CM)),
        depth_cm=float(rng.uniform(*PIPE_DEPTH_RANGE_CM)),
        radius_cm=0.5 * float(rng.uniform(*PIPE_DIAMETER_RANGE_CM)),
    )


def draw_pavement(rng: np.random.Generator) -> Pavement:
    if rng.random() < 0.5:
        return Pavement(SIDEWALK, float(rng.uniform(*SIDEWALK_START_RANGE_CM)))
    return Pavement(TREE_PIT, float(rng.uniform(*TREE_PIT_HALF_SIZE_RANGE_CM)))


def draw_obstacle_layout(family: str, rng: np.random.Generator) -> ObstacleLayout:
    if family == NO_OBSTACLE:
        return ObstacleLayout(family)
    if family == SINGLE_PIPE:
        return ObstacleLayout(family, (draw_pipe(ALONG_Y, rng),))
    if family == CROSSING_PIPES:
        return ObstacleLayout(family, (draw_pipe(ALONG_Y, rng), draw_pipe(ALONG_X, rng)))
    if family == PIPE_AND_PAVEMENT:
        return ObstacleLayout(family, (draw_pipe(ALONG_Y, rng),), draw_pavement(rng))
    raise ValueError(f"unknown obstacle family: {family}")
