"""The v1 soil pipeline, kept only so the sandbox can compare it with the current one.

Three differences, in the order the soil is built:
  1. noise drawn independently per channel, so a voxel could be dense, wet and weak;
  2. Rosetta called once per horizon instead of once per voxel, so textural noise
     never reached the retention parameters, only the moisture did;
  3. the two stress factors multiplied instead of combined by their minimum, which
     penalised a soil that is both hard and dry twice, and the mechanical factor
     floored at 0.05 instead of being allowed to reach zero.

Nothing here is imported by the production pipeline.
"""
import numpy as np
import plantbox.functional.van_genuchten as vg
from rosetta import SoilData, rosetta

from grid import VoxelGrid
from soil.archetypes import (
    BULK_DENSITY_RANGE_G_CM3,
    MOISTURE_RANGE_M3_M3,
    PENETROMETER_RANGE_MPA,
    TEXTURE_FRACTION_RANGE,
    SoilHorizon,
)
from soil.field import NOISE_AMPLITUDE, SoilField, build_smooth_noise
from soil.growth_response import apply_bengough, apply_water_stress
from soil.hydraulics import convert_moisture_to_pressure_head
from soil.profile import SoilProfile
from soil.rosetta_batch import LOG_BASE, ROSETTA_VERSION

FRACTION_TO_PERCENT = 100.0
LEGACY_MINIMUM_MECHANICAL_SCALE = 0.05


def build_topsoil_mask(profile: SoilProfile, grid: VoxelGrid) -> np.ndarray:
    """Un seul profil pour toute la scene : v1 ne connait pas la mosaique."""
    depths = grid.measure_layer_depths_cm()[:, None, None]
    return np.broadcast_to(depths < profile.topsoil_depth_cm, grid.shape)


def build_channel(topsoil_value, subsoil_value, topsoil_mask, bounds, noise) -> np.ndarray:
    base = np.where(topsoil_mask, topsoil_value, subsoil_value)
    return np.clip(base * (1.0 + NOISE_AMPLITUDE * noise), bounds[0], bounds[1]).astype(np.float32)


def build_soil_field(profile: SoilProfile, grid: VoxelGrid, rng: np.random.Generator) -> SoilField:
    """A fresh noise draw per channel, in the same order v1 consumed them."""
    topsoil_mask = build_topsoil_mask(profile, grid)
    top, sub = profile.topsoil, profile.subsoil

    def channel(attribute, bounds):
        return build_channel(getattr(top, attribute), getattr(sub, attribute), topsoil_mask,
                             bounds, build_smooth_noise(grid.shape, rng))

    return SoilField(
        penetrometer_mpa=channel("penetrometer_mpa", PENETROMETER_RANGE_MPA),
        bulk_density_g_cm3=channel("bulk_density_g_cm3", BULK_DENSITY_RANGE_G_CM3),
        moisture_m3_m3=channel("moisture_m3_m3", MOISTURE_RANGE_M3_M3),
        sand_fraction=channel("sand_fraction", TEXTURE_FRACTION_RANGE),
        clay_fraction=channel("clay_fraction", TEXTURE_FRACTION_RANGE),
    )


def estimate_van_genuchten_parameters(horizon: SoilHorizon) -> vg.Parameters:
    texture = SoilData.from_array([[
        horizon.sand_fraction * FRACTION_TO_PERCENT,
        horizon.silt_fraction * FRACTION_TO_PERCENT,
        horizon.clay_fraction * FRACTION_TO_PERCENT,
        horizon.bulk_density_g_cm3,
    ]])
    theta_r, theta_s, log_alpha, log_n, log_conductivity = rosetta(ROSETTA_VERSION, texture)[0][0]
    return vg.Parameters([theta_r, theta_s, LOG_BASE ** log_alpha, LOG_BASE ** log_n,
                          LOG_BASE ** log_conductivity])


def build_pressure_head_field(field: SoilField, profile: SoilProfile, grid: VoxelGrid) -> np.ndarray:
    """Two retention curves for the whole scene, composited by depth."""
    topsoil_head = convert_moisture_to_pressure_head(
        field.moisture_m3_m3, estimate_van_genuchten_parameters(profile.topsoil))
    subsoil_head = convert_moisture_to_pressure_head(
        field.moisture_m3_m3, estimate_van_genuchten_parameters(profile.subsoil))
    return np.where(build_topsoil_mask(profile, grid), topsoil_head, subsoil_head).astype(np.float32)


def compute_growth_scale(penetrometer_mpa: np.ndarray, pressure_head_cm: np.ndarray) -> np.ndarray:
    """The floor is reapplied here so this stays a faithful v1 reference."""
    mechanical = np.maximum(apply_bengough(penetrometer_mpa), LEGACY_MINIMUM_MECHANICAL_SCALE)
    return (mechanical * apply_water_stress(pressure_head_cm)).astype(np.float32)
