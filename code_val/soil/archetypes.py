"""The ten reference soils of the specification, and the physical bounds they live in.

Each archetype is a starting point, not a fixed recipe: draw_soil_profile() samples
around it so no two configurations share the same numbers. Without that jitter a
network can identify "this is soil number 3" and memorise its answer.
"""
from dataclasses import dataclass

PENETROMETER_RANGE_MPA = (0.0, 5.0)
BULK_DENSITY_RANGE_G_CM3 = (1.0, 2.2)
MOISTURE_RANGE_M3_M3 = (0.02, 0.45)
TEXTURE_FRACTION_RANGE = (0.0, 1.0)

UNIFORM_PROFILE_DEPTH_CM = 40.0
TOPSOIL_DEPTH_RANGE_CM = (15.0, 70.0)


@dataclass(frozen=True)
class SoilHorizon:
    penetrometer_mpa: float
    bulk_density_g_cm3: float
    moisture_m3_m3: float
    sand_fraction: float
    clay_fraction: float

    @property
    def silt_fraction(self) -> float:
        return 1.0 - self.sand_fraction - self.clay_fraction


@dataclass(frozen=True)
class SoilArchetype:
    identifier: int
    name: str
    topsoil_depth_cm: float
    topsoil: SoilHorizon
    subsoil: SoilHorizon

    @property
    def is_uniform(self) -> bool:
        return self.topsoil == self.subsoil


def _uniform(identifier, name, horizon):
    return SoilArchetype(identifier, name, UNIFORM_PROFILE_DEPTH_CM, horizon, horizon)


ARCHETYPES = (
    _uniform(1, "loose_uniform", SoilHorizon(0.8, 1.25, 0.28, 0.40, 0.20)),
    _uniform(2, "medium_uniform", SoilHorizon(1.5, 1.45, 0.25, 0.40, 0.20)),
    _uniform(3, "compacted_uniform", SoilHorizon(3.0, 1.70, 0.20, 0.35, 0.25)),
    _uniform(4, "very_compacted", SoilHorizon(4.0, 1.85, 0.16, 0.30, 0.30)),
    _uniform(5, "sandy_dry", SoilHorizon(1.2, 1.50, 0.12, 0.80, 0.05)),
    _uniform(6, "clayey_wet", SoilHorizon(1.8, 1.50, 0.38, 0.15, 0.50)),
    SoilArchetype(7, "compacted_top", 40.0,
                  SoilHorizon(3.2, 1.75, 0.18, 0.35, 0.25),
                  SoilHorizon(1.2, 1.40, 0.28, 0.40, 0.20)),
    SoilArchetype(8, "hardpan_below", 30.0,
                  SoilHorizon(1.0, 1.30, 0.26, 0.40, 0.20),
                  SoilHorizon(3.8, 1.80, 0.18, 0.35, 0.30)),
    SoilArchetype(9, "hard_base_at_50", 50.0,
                  SoilHorizon(1.5, 1.45, 0.25, 0.40, 0.20),
                  SoilHorizon(5.0, 2.10, 0.10, 0.30, 0.25)),
    SoilArchetype(10, "sand_over_clay", 45.0,
                  SoilHorizon(1.2, 1.50, 0.14, 0.80, 0.05),
                  SoilHorizon(2.0, 1.55, 0.36, 0.15, 0.50)),
)


PIT_SUBSTRATE_ARCHETYPE = _uniform(0, "pit_substrate", SoilHorizon(0.4, 1.15, 0.30, 0.60, 0.12))


def find_archetype(identifier: int) -> SoilArchetype:
    return next(archetype for archetype in ARCHETYPES if archetype.identifier == identifier)
