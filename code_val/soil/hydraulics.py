"""Convert soil texture and density into a matric potential grid.

Rosetta predicts the van Genuchten retention parameters from sand, silt, clay and
bulk density; van Genuchten then turns the moisture grid into a pressure head grid.
Rosetta is called once per horizon, the conversion once per configuration.
"""
import numpy as np
import plantbox.functional.van_genuchten as vg
from rosetta import SoilData, rosetta

from grid import VoxelGrid
from soil.archetypes import SoilHorizon
from soil.field import build_topsoil_mask
from soil.profile import SoilProfile

ROSETTA_VERSION = 3
FRACTION_TO_PERCENT = 100.0
MOISTURE_MARGIN_M3_M3 = 1e-3


def estimate_van_genuchten_parameters(horizon: SoilHorizon) -> vg.Parameters:
    texture = SoilData.from_array([[
        horizon.sand_fraction * FRACTION_TO_PERCENT,
        horizon.silt_fraction * FRACTION_TO_PERCENT,
        horizon.clay_fraction * FRACTION_TO_PERCENT,
        horizon.bulk_density_g_cm3,
    ]])
    theta_r, theta_s, log_alpha, log_n, log_conductivity = rosetta(ROSETTA_VERSION, texture)[0][0]
    return vg.Parameters([theta_r, theta_s, 10.0 ** log_alpha, 10.0 ** log_n, 10.0 ** log_conductivity])


def convert_moisture_to_pressure_head(moisture: np.ndarray, parameters: vg.Parameters) -> np.ndarray:
    """Moisture outside the retention curve is clipped: van Genuchten is undefined there."""
    lowest = min(parameters.theta_R + MOISTURE_MARGIN_M3_M3, parameters.theta_S)
    return vg.pressure_head(np.clip(moisture, lowest, parameters.theta_S), parameters)


def build_pressure_head_field(moisture: np.ndarray, profile: SoilProfile, grid: VoxelGrid) -> np.ndarray:
    topsoil_head = convert_moisture_to_pressure_head(
        moisture, estimate_van_genuchten_parameters(profile.topsoil))
    subsoil_head = convert_moisture_to_pressure_head(
        moisture, estimate_van_genuchten_parameters(profile.subsoil))
    return np.where(build_topsoil_mask(profile, grid), topsoil_head, subsoil_head).astype(np.float32)
