"""The reference soils and the physical bounds they live in.
"""
from dataclasses import dataclass

# Physical ranges
PENETROMETER_RANGE_MPA = (0.0, 5.0)
BULK_DENSITY_RANGE_G_CM3 = (1.0, 2.2)
MOISTURE_RANGE_M3_M3 = (0.02, 0.45)
TEXTURE_FRACTION_RANGE = (0.0, 1.0)

TOPSOIL_DEPTH_RANGE_CM = (15.0, 70.0)

# --- Soil horizons ---
@dataclass(frozen=True)
class SoilHorizon:
    # Soil properties
    penetrometer_mpa: float
    bulk_density_g_cm3: float
    moisture_m3_m3: float
    sand_fraction: float
    clay_fraction: float

    # Return silt fraction (Soil = triangle of sand, silt, clay)
    @property
    def silt_fraction(self) -> float:
        return 1.0 - self.sand_fraction - self.clay_fraction

# --- Soil archetypes ---
@dataclass(frozen=True)
class SoilArchetype:
    identifier: int
    name: str
    topsoil_depth_cm: float
    # Soil horizons
    topsoil: SoilHorizon
    subsoil: SoilHorizon

    @property
    def is_uniform(self) -> bool:
        return self.topsoil == self.subsoil

# Tuples of soil archetypes examples
ARCHETYPES = (
    SoilArchetype(1, "loose_uniform", 40.0, SoilHorizon(0.8, 1.25, 0.28, 0.40, 0.20), SoilHorizon(0.8, 1.25, 0.28, 0.40, 0.20)),
    SoilArchetype(2, "medium_uniform",40.0, SoilHorizon(1.5, 1.45, 0.25, 0.40, 0.20), SoilHorizon(1.5, 1.45, 0.25, 0.40, 0.20)),
    SoilArchetype(3, "compacted_uniform",40.0 , SoilHorizon(3.0, 1.70, 0.20, 0.35, 0.25), SoilHorizon(3.0, 1.70, 0.20, 0.35, 0.25)),
    SoilArchetype(4, "very_compacted",40.0,SoilHorizon(4.0, 1.85, 0.16, 0.30, 0.30), SoilHorizon(4.0, 1.85, 0.16, 0.30, 0.30)),
    SoilArchetype(5, "sandy_dry", 40.0,SoilHorizon(1.2, 1.50, 0.12, 0.80, 0.05), SoilHorizon(1.2, 1.50, 0.12, 0.80, 0.05)),
    SoilArchetype(6, "clayey_wet", 40.0,SoilHorizon(1.8, 1.50, 0.38, 0.15, 0.50), SoilHorizon(1.8, 1.50, 0.38, 0.15, 0.50)),
    SoilArchetype(7, "compacted_top", 40.0, SoilHorizon(3.2, 1.75, 0.18, 0.35, 0.25),SoilHorizon(1.2, 1.40, 0.28, 0.40, 0.20)),
    SoilArchetype(8, "hardpan_below", 30.0, SoilHorizon(1.0, 1.30, 0.26, 0.40, 0.20), SoilHorizon(3.8, 1.80, 0.18, 0.35, 0.30)),
    SoilArchetype(9, "hard_base_at_50", 50.0, SoilHorizon(1.5, 1.45, 0.25, 0.40, 0.20), SoilHorizon(5.0, 2.10, 0.10, 0.30, 0.25)),
    SoilArchetype(10, "sand_over_clay", 45.0, SoilHorizon(1.2, 1.50, 0.14, 0.80, 0.05), SoilHorizon(2.0, 1.55, 0.36, 0.15, 0.50)),
)
# Pit archetype
PIT_SUBSTRATE_ARCHETYPE = SoilArchetype(0, "pit_substrate", 40.0, SoilHorizon(1.0, 1.30, 0.30, 0.60, 0.12), SoilHorizon(1.0, 1.30, 0.30, 0.60, 0.12))


def find_archetype(identifier: int) -> SoilArchetype:
    return next(archetype for archetype in ARCHETYPES if archetype.identifier == identifier)
