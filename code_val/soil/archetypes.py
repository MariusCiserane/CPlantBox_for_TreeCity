"""The reference soils and the physical bounds they live in.

Water is written as a **fraction of the available water capacity**, not as a volumetric
moisture. A moisture is not comparable across textures: theta = 0.12 is above field capacity
in a sand and below the wilting point in a clay, so the same number on two lines of this
table would describe two opposite soils. The fraction means the same thing everywhere - 0 is
the wilting point, 1 is field capacity - and soil/hydraulics.py turns it back into theta with
each voxel's own retention curve.
"""
from dataclasses import dataclass

# Physical ranges
PENETROMETER_RANGE_MPA = (0.0, 5.0)
BULK_DENSITY_RANGE_G_CM3 = (1.0, 2.2)
AVAILABLE_WATER_RANGE = (0.0, 1.0)
TEXTURE_FRACTION_RANGE = (0.0, 1.0)

TOPSOIL_DEPTH_RANGE_CM = (15.0, 70.0)

# --- Soil horizons ---
@dataclass(frozen=True)
class SoilHorizon:
    # Soil properties
    penetrometer_mpa: float
    bulk_density_g_cm3: float
    available_water_fraction: float
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
    SoilArchetype(1, "loose_uniform", 40.0, SoilHorizon(0.8, 1.25, 0.65, 0.40, 0.20), SoilHorizon(0.8, 1.25, 0.65, 0.40, 0.20)),
    SoilArchetype(2, "medium_uniform",40.0, SoilHorizon(1.5, 1.45, 0.55, 0.40, 0.20), SoilHorizon(1.5, 1.45, 0.55, 0.40, 0.20)),
    SoilArchetype(3, "compacted_uniform",40.0 , SoilHorizon(3.0, 1.70, 0.45, 0.35, 0.25), SoilHorizon(3.0, 1.70, 0.45, 0.35, 0.25)),
    SoilArchetype(4, "very_compacted",40.0,SoilHorizon(4.0, 1.85, 0.30, 0.30, 0.30), SoilHorizon(4.0, 1.85, 0.30, 0.30, 0.30)),
    SoilArchetype(5, "sandy_dry", 40.0,SoilHorizon(1.2, 1.50, 0.15, 0.80, 0.05), SoilHorizon(1.2, 1.50, 0.15, 0.80, 0.05)),
    SoilArchetype(6, "clayey_wet", 40.0,SoilHorizon(1.8, 1.50, 0.90, 0.15, 0.50), SoilHorizon(1.8, 1.50, 0.90, 0.15, 0.50)),
    SoilArchetype(7, "compacted_top", 40.0, SoilHorizon(3.2, 1.75, 0.35, 0.35, 0.25),SoilHorizon(1.2, 1.40, 0.60, 0.40, 0.20)),
    SoilArchetype(8, "hardpan_below", 30.0, SoilHorizon(1.0, 1.30, 0.60, 0.40, 0.20), SoilHorizon(3.8, 1.80, 0.35, 0.35, 0.30)),
    SoilArchetype(9, "hard_base_at_50", 50.0, SoilHorizon(1.5, 1.45, 0.55, 0.40, 0.20), SoilHorizon(5.0, 2.10, 0.20, 0.30, 0.25)),
    SoilArchetype(10, "sand_over_clay", 45.0, SoilHorizon(1.2, 1.50, 0.25, 0.80, 0.05), SoilHorizon(2.0, 1.55, 0.80, 0.15, 0.50)),
)

# Watered at planting and drained by design, so the substrate starts well supplied.
PIT_SUBSTRATE_ARCHETYPE = SoilArchetype(0, "pit_substrate", 40.0, SoilHorizon(1.0, 1.30, 0.70, 0.60, 0.12), SoilHorizon(1.0, 1.30, 0.70, 0.60, 0.12))


def find_archetype(identifier: int) -> SoilArchetype:
    return next(archetype for archetype in ARCHETYPES if archetype.identifier == identifier)
