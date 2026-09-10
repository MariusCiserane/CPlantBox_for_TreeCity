"""Expand a soil profile into the five physical channel grids.

A horizon is not uniform in reality, so every channel carries smooth spatial
noise. Two noise fields are drawn, one per physical process, and shared by the
channels that process drives: compaction moves resistance, density and moisture
together, while textural heterogeneity comes from different fill materials and
is unrelated to traffic loading. Noising each channel on its own, as before,
allowed a voxel to be dense, wet and weak at the same time.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from grid import VoxelGrid
from soil.archetypes import (
    BULK_DENSITY_RANGE_G_CM3,
    MOISTURE_RANGE_M3_M3,
    PENETROMETER_RANGE_MPA,
    TEXTURE_FRACTION_RANGE,
)
from soil.profile import SoilProfile

NOISE_CORRELATION_VOXELS = 2.0
NOISE_AMPLITUDE = 0.08

# Penetrometer resistance rises with density and falls with moisture; sand and clay
# move apart so a patch stays a plausible texture.
CHANNEL_NOISE_SIGNS = {
    "penetrometer_mpa": 1.0,
    "bulk_density_g_cm3": 1.0,
    "moisture_m3_m3": -1.0,
    "sand_fraction": 1.0,
    "clay_fraction": -1.0,
}
COMPACTION_CHANNELS = ("penetrometer_mpa", "bulk_density_g_cm3", "moisture_m3_m3")


@dataclass(frozen=True)
class RoadBase:
    penetrometer_mpa: float
    bulk_density_g_cm3: float
    moisture_m3_m3: float


@dataclass(frozen=True)
class SoilField:
    penetrometer_mpa: np.ndarray
    bulk_density_g_cm3: np.ndarray
    moisture_m3_m3: np.ndarray
    sand_fraction: np.ndarray
    clay_fraction: np.ndarray


def build_smooth_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    noise = gaussian_filter(rng.standard_normal(shape), sigma=NOISE_CORRELATION_VOXELS)
    return noise / noise.std()


def build_topsoil_mask(profile: SoilProfile, grid: VoxelGrid) -> np.ndarray:
    depths = grid.measure_layer_depths_cm()[:, None, None]
    return np.broadcast_to(depths < profile.topsoil_depth_cm, grid.shape)


def build_channel(topsoil_value, subsoil_value, topsoil_mask, bounds, signed_noise) -> np.ndarray:
    base = np.where(topsoil_mask, topsoil_value, subsoil_value)
    noisy = base * (1.0 + NOISE_AMPLITUDE * signed_noise)
    return np.clip(noisy, bounds[0], bounds[1]).astype(np.float32)


def build_soil_field(profile: SoilProfile, grid: VoxelGrid, rng: np.random.Generator) -> SoilField:
    topsoil_mask = build_topsoil_mask(profile, grid)
    compaction_noise = build_smooth_noise(grid.shape, rng)
    texture_noise = build_smooth_noise(grid.shape, rng)
    top, sub = profile.topsoil, profile.subsoil

    def channel(attribute, bounds):
        noise = compaction_noise if attribute in COMPACTION_CHANNELS else texture_noise
        return build_channel(getattr(top, attribute), getattr(sub, attribute), topsoil_mask, bounds, CHANNEL_NOISE_SIGNS[attribute] * noise)

    return SoilField(
        penetrometer_mpa=channel("penetrometer_mpa", PENETROMETER_RANGE_MPA),
        bulk_density_g_cm3=channel("bulk_density_g_cm3", BULK_DENSITY_RANGE_G_CM3),
        moisture_m3_m3=channel("moisture_m3_m3", MOISTURE_RANGE_M3_M3),
        sand_fraction=channel("sand_fraction", TEXTURE_FRACTION_RANGE),
        clay_fraction=channel("clay_fraction", TEXTURE_FRACTION_RANGE),
    )


def apply_road_base_override(field: SoilField, mask: np.ndarray, road_base: RoadBase) -> SoilField:
    """A pavement compacts and dries the aggregate sub-base underneath it."""
    return replace(
        field,
        penetrometer_mpa=np.where(mask, road_base.penetrometer_mpa, field.penetrometer_mpa),
        bulk_density_g_cm3=np.where(mask, road_base.bulk_density_g_cm3, field.bulk_density_g_cm3),
        moisture_m3_m3=np.where(mask, road_base.moisture_m3_m3, field.moisture_m3_m3),
    )
