"""One geometric description of the buried networks, shared by everything that reads it.

The hard wall, the soil penalty and the intrusion check all have to agree on where a pipe
is. Describing the scene three times invites them to drift apart, so it is described once,
here, as axis-aligned cuboids and horizontal cylinders, and each consumer reads the tuple
it needs.

Fills are ordered by physical priority and consumed against a decreasing budget, which is
what carves the rings: a lumen reserves its volume, the sand surround takes what is left
inside the trench, and the backfill takes the rest. No shape has to be described as a ring.
"""
from dataclasses import dataclass
from typing import Optional

from grid import VoxelGrid
from obstacles.layouts import ALONG_Y
from obstacles.street import (
    LOOSE_BACKFILL,
    PIPE_SURROUND,
    RECOMPACTED_BACKFILL,
    StreetScene,
    Utility,
)
from soil.archetypes import SoilHorizon


@dataclass(frozen=True)
class Cylinder:
    name: str
    axis: str
    offset_cm: float
    depth_cm: float
    radius_cm: float


@dataclass(frozen=True)
class Cuboid:
    name: str
    minimum_cm: tuple[float, float, float]
    maximum_cm: tuple[float, float, float]

    @property
    def is_empty(self) -> bool:
        return any(low >= high for low, high in zip(self.minimum_cm, self.maximum_cm))


@dataclass(frozen=True)
class StreetShapes:
    fills: tuple[tuple[Cylinder | Cuboid, Optional[SoilHorizon]], ...]
    lumens: tuple[Cylinder, ...]


def intersect_cuboids(name: str, first: Cuboid, second: Cuboid) -> Cuboid:
    minimum = tuple(max(a, b) for a, b in zip(first.minimum_cm, second.minimum_cm))
    maximum = tuple(min(a, b) for a, b in zip(first.maximum_cm, second.maximum_cm))
    return Cuboid(name, minimum, maximum)


def build_trench_cuboid(utility: Utility, grid: VoxelGrid) -> Cuboid:
    """Excavated from the surface down past the bedding, so it is a slot, not a sleeve."""
    bottom = -(utility.depth_cm + utility.radius_cm + utility.bedding_depth_cm)
    half_width = utility.trench_half_width_cm
    if utility.axis == ALONG_Y:
        return Cuboid(f"{utility.name}_trench",
                      (utility.offset_cm - half_width, -grid.half_length_cm, bottom),
                      (utility.offset_cm + half_width, grid.half_length_cm, 0.0))
    return Cuboid(f"{utility.name}_trench",
                  (-grid.half_width_cm, utility.offset_cm - half_width, bottom),
                  (grid.half_width_cm, utility.offset_cm + half_width, 0.0))


def build_lumen(utility: Utility) -> Cylinder:
    return Cylinder(utility.name, utility.axis, utility.offset_cm, utility.depth_cm,
                    utility.radius_cm)


def build_surround(utility: Utility) -> Cylinder:
    return Cylinder(f"{utility.name}_surround", utility.axis, utility.offset_cm,
                    utility.depth_cm, utility.radius_cm + utility.surround_cover_cm)


def build_scene_shapes(scene: StreetScene, grid: VoxelGrid) -> StreetShapes:
    pit_column = Cuboid("pit_column",
                        (-scene.pit_radius_cm, -scene.pit_radius_cm, -grid.depth_cm),
                        (scene.pit_radius_cm, scene.pit_radius_cm, 0.0))

    lumens = tuple(build_lumen(utility) for utility in scene.utilities)
    trenches = tuple(build_trench_cuboid(utility, grid) for utility in scene.utilities)
    under_pit = tuple(intersect_cuboids(f"{trench.name}_under_pit", trench, pit_column)
                      for trench in trenches)

    fills = (
        tuple((lumen, None) for lumen in lumens)
        + tuple((build_surround(utility), PIPE_SURROUND) for utility in scene.utilities)
        + tuple((piece, LOOSE_BACKFILL) for piece in under_pit if not piece.is_empty)
        + tuple((trench, RECOMPACTED_BACKFILL) for trench in trenches)
    )
    return StreetShapes(fills, lumens)
