"""Draw one concrete soil profile around an archetype."""
from dataclasses import dataclass

import numpy as np

from soil.archetypes import (AVAILABLE_WATER_RANGE, BULK_DENSITY_RANGE_G_CM3, PENETROMETER_RANGE_MPA, TEXTURE_FRACTION_RANGE, TOPSOIL_DEPTH_RANGE_CM, SoilArchetype, SoilHorizon)

RELATIVE_JITTER = 0.15
DEPTH_JITTER_CM = 10.0
MAXIMUM_SAND_PLUS_CLAY = 0.95



def clamp(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def jitter_value(value: float, bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return clamp(value * (1.0 + rng.uniform(-RELATIVE_JITTER, RELATIVE_JITTER)), bounds)


def jitter_texture(sand: float, clay: float, rng: np.random.Generator) -> tuple[float, float]:
    sand = jitter_value(sand, TEXTURE_FRACTION_RANGE, rng)
    clay = jitter_value(clay, TEXTURE_FRACTION_RANGE, rng)
    total = sand + clay
    if total > MAXIMUM_SAND_PLUS_CLAY:
        scale = MAXIMUM_SAND_PLUS_CLAY / total
        sand, clay = sand * scale, clay * scale
    return sand, clay


def jitter_horizon(horizon: SoilHorizon, rng: np.random.Generator) -> SoilHorizon:
    sand, clay = jitter_texture(horizon.sand_fraction, horizon.clay_fraction, rng)
    return SoilHorizon(
        penetrometer_mpa=jitter_value(horizon.penetrometer_mpa, PENETROMETER_RANGE_MPA, rng),
        bulk_density_g_cm3=jitter_value(horizon.bulk_density_g_cm3, BULK_DENSITY_RANGE_G_CM3, rng),
        available_water_fraction=jitter_value(horizon.available_water_fraction, AVAILABLE_WATER_RANGE, rng),
        sand_fraction=sand,
        clay_fraction=clay,
    )


@dataclass(frozen=True)
class SoilProfile:
    archetype_identifier: int
    archetype_name: str
    topsoil_depth_cm: float
    topsoil: SoilHorizon
    subsoil: SoilHorizon

def draw_soil_profile(archetype: SoilArchetype, rng: np.random.Generator) -> SoilProfile:
    topsoil = jitter_horizon(archetype.topsoil, rng)
    subsoil = topsoil if archetype.is_uniform else jitter_horizon(archetype.subsoil, rng)
    depth = clamp(archetype.topsoil_depth_cm + rng.uniform(-DEPTH_JITTER_CM, DEPTH_JITTER_CM), TOPSOIL_DEPTH_RANGE_CM)
    return SoilProfile(archetype.identifier, archetype.name, depth, topsoil, subsoil)
