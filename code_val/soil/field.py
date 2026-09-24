"""Expand a soil mosaic into the five physical channel grids.

A horizon is not uniform in reality, so every channel carries smooth spatial
noise. Two noise fields are drawn, one per physical process, and shared by the
channels that process drives: compaction moves resistance, density and water
together, while textural heterogeneity comes from different fill materials and
is unrelated to traffic loading. Noising each channel on its own, as before,
allowed a voxel to be dense, wet and weak at the same time.

Several archetypes can share one domain. Every channel is blended with the same
patch weights, which keeps a blended voxel on the segment between two real soils
instead of somewhere no soil exists. The noise stays shared across patches: one
noise field per patch would have to be blended too, and blending independent
gaussians shrinks the variance right inside the transition band.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from grid import VoxelGrid
from soil.archetypes import (
    AVAILABLE_WATER_RANGE,
    BULK_DENSITY_RANGE_G_CM3,
    PENETROMETER_RANGE_MPA,
    TEXTURE_FRACTION_RANGE,
)
from soil.mosaic import SoilMosaic, single_patch_mosaic
from soil.profile import SoilProfile

NOISE_CORRELATION_VOXELS = 2.0
NOISE_AMPLITUDE = 0.08

# Penetrometer resistance rises with density and falls with moisture; sand and clay
# move apart so a patch stays a plausible texture.
CHANNEL_NOISE_SIGNS = {
    "penetrometer_mpa": 1.0,
    "bulk_density_g_cm3": 1.0,
    "available_water_fraction": -1.0,
    "sand_fraction": 1.0,
    "clay_fraction": -1.0,
}
COMPACTION_CHANNELS = ("penetrometer_mpa", "bulk_density_g_cm3", "available_water_fraction")

# Literature : -1 to -5 (depending on soil type and moisture). It is published against
# volumetric moisture; applied here to the ratio of available water fractions, which keeps the
# direction and is bounded by RESISTANCE_SCALE_RANGE either way.
WATER_RESISTANCE_EXPONENT = -2.0
RESISTANCE_SCALE_RANGE = (0.25, 2.0)
DRIEST_REFERENCE_FRACTION = 0.02


@dataclass(frozen=True)
class RoadBase:
    penetrometer_mpa: float
    bulk_density_g_cm3: float
    available_water_fraction: float


@dataclass(frozen=True)
class SoilField:
    penetrometer_mpa: np.ndarray
    bulk_density_g_cm3: np.ndarray
    available_water_fraction: np.ndarray
    sand_fraction: np.ndarray
    clay_fraction: np.ndarray


def build_smooth_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    noise = gaussian_filter(rng.standard_normal(shape), sigma=NOISE_CORRELATION_VOXELS)
    return noise / noise.std()


def build_patch_values(mosaic: SoilMosaic, attribute: str, grid: VoxelGrid) -> np.ndarray:
    """Une colonne de valeurs par patche : chacune garde sa propre profondeur d'horizon."""
    depths = grid.measure_layer_depths_cm()
    return np.array([np.where(depths < patch.profile.topsoil_depth_cm,
                              getattr(patch.profile.topsoil, attribute),
                              getattr(patch.profile.subsoil, attribute))
                     for patch in mosaic.patches])


def build_channel(values: np.ndarray, weights: np.ndarray, bounds, signed_noise) -> np.ndarray:
    base = np.einsum("nd,nrc->drc", values, weights)
    noisy = base * (1.0 + NOISE_AMPLITUDE * signed_noise)
    return np.clip(noisy, bounds[0], bounds[1]).astype(np.float32)


def build_mosaic_field(mosaic: SoilMosaic, weights: np.ndarray, grid: VoxelGrid,
                       rng: np.random.Generator) -> SoilField:
    compaction_noise = build_smooth_noise(grid.shape, rng)
    texture_noise = build_smooth_noise(grid.shape, rng)

    def channel(attribute, bounds):
        noise = compaction_noise if attribute in COMPACTION_CHANNELS else texture_noise
        return build_channel(build_patch_values(mosaic, attribute, grid), weights, bounds,
                             CHANNEL_NOISE_SIGNS[attribute] * noise)

    return SoilField(
        penetrometer_mpa=channel("penetrometer_mpa", PENETROMETER_RANGE_MPA),
        bulk_density_g_cm3=channel("bulk_density_g_cm3", BULK_DENSITY_RANGE_G_CM3),
        available_water_fraction=channel("available_water_fraction", AVAILABLE_WATER_RANGE),
        sand_fraction=channel("sand_fraction", TEXTURE_FRACTION_RANGE),
        clay_fraction=channel("clay_fraction", TEXTURE_FRACTION_RANGE),
    )


def build_soil_field(profile: SoilProfile, grid: VoxelGrid, rng: np.random.Generator) -> SoilField:
    weights = np.ones((1, grid.row_cells, grid.column_cells))
    return build_mosaic_field(single_patch_mosaic(profile), weights, grid, rng)


def apply_water_to_resistance(field: SoilField, reference_fraction: np.ndarray,natural_share: np.ndarray) -> SoilField:
    """Rescale penetration resistance wherever the water content has moved on its own."""
    reference = np.maximum(reference_fraction, DRIEST_REFERENCE_FRACTION)
    scale = np.clip((field.available_water_fraction / reference) ** WATER_RESISTANCE_EXPONENT,*RESISTANCE_SCALE_RANGE) ** natural_share
    softened = np.clip(field.penetrometer_mpa * scale, *PENETROMETER_RANGE_MPA)
    return replace(field, penetrometer_mpa=softened.astype(np.float32))


def apply_road_base_override(field: SoilField, mask: np.ndarray, road_base: RoadBase) -> SoilField:
    """A pavement compacts and dries the aggregate sub-base underneath it."""
    return replace(
        field,
        penetrometer_mpa=np.where(mask, road_base.penetrometer_mpa, field.penetrometer_mpa),
        bulk_density_g_cm3=np.where(mask, road_base.bulk_density_g_cm3, field.bulk_density_g_cm3),
        available_water_fraction=np.where(mask, road_base.available_water_fraction, field.available_water_fraction),
    )
