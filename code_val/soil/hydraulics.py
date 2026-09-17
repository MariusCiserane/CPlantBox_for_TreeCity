"""Convert the soil field into a matric potential grid.

Every voxel gets its own retention curve, predicted from its own noised texture
and density, so textural variation reaches the retention physics instead of only
sliding the moisture along one curve per horizon. The conversion runs on the
composited field, after any obstacle override, not on the raw archetype profile.
"""
import numpy as np
import plantbox.functional.van_genuchten as vg

from soil.field import SoilField
from soil.rosetta_batch import estimate_van_genuchten_parameters_field

MOISTURE_MARGIN_M3_M3 = 1e-3


def convert_moisture_to_pressure_head(moisture: np.ndarray, parameters: vg.Parameters) -> np.ndarray:
    """Moisture outside the retention curve is clipped: van Genuchten is undefined there."""
    # Clip moisture to the valid range of the retention curve, with a small margin to avoid numerical issues
    lowest = np.minimum(parameters.theta_R + MOISTURE_MARGIN_M3_M3, parameters.theta_S)
    return vg.pressure_head(np.clip(moisture, lowest, parameters.theta_S), parameters)


def build_pressure_head_field(field: SoilField) -> np.ndarray:
    parameters = estimate_van_genuchten_parameters_field(field.sand_fraction, field.clay_fraction, field.bulk_density_g_cm3)
    return convert_moisture_to_pressure_head(field.moisture_m3_m3, parameters).astype(np.float32)
