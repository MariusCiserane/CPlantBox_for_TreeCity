"""Turn the authored water fraction into the moisture and matric potential the physics needs.

Every voxel gets its own retention curve, predicted from its own noised texture
and density, so textural variation reaches the retention physics instead of only
sliding the moisture along one curve per horizon. The conversion runs on the
composited field, after any obstacle override, not on the raw archetype profile.

Both quantities come out of one Rosetta call: theta depends on the curve, and the
matric potential is read back off the same curve.
"""
from typing import NamedTuple

import numpy as np
import plantbox.functional.van_genuchten as vg

from soil.field import SoilField
from soil.rosetta_batch import estimate_van_genuchten_parameters_field

MOISTURE_MARGIN_M3_M3 = 1e-3
FIELD_CAPACITY_HEAD_CM = -100.0
WILTING_POINT_HEAD_CM = -15000.0


class WaterFields(NamedTuple):
    moisture_m3_m3: np.ndarray
    pressure_head_cm: np.ndarray


def convert_available_water_to_moisture(available_water_fraction: np.ndarray, parameters: vg.Parameters) -> np.ndarray:
    """0 sits at the wilting point and 1 at field capacity, so the same fraction is the same
    plant-available state in a sand and in a clay even though theta differs by a factor of five."""
    wilting = vg.water_content(WILTING_POINT_HEAD_CM, parameters)
    capacity = vg.water_content(FIELD_CAPACITY_HEAD_CM, parameters)
    return wilting + np.clip(available_water_fraction, 0.0, 1.0) * (capacity - wilting)


def convert_moisture_to_pressure_head(moisture: np.ndarray, parameters: vg.Parameters) -> np.ndarray:
    """Moisture outside the retention curve is clipped: van Genuchten is undefined there."""
    # Clip moisture to the valid range of the retention curve, with a small margin to avoid numerical issues
    lowest = np.minimum(parameters.theta_R + MOISTURE_MARGIN_M3_M3, parameters.theta_S)
    return vg.pressure_head(np.clip(moisture, lowest, parameters.theta_S), parameters)


def build_water_fields(field: SoilField) -> WaterFields:
    parameters = estimate_van_genuchten_parameters_field(field.sand_fraction, field.clay_fraction, field.bulk_density_g_cm3)
    moisture = convert_available_water_to_moisture(field.available_water_fraction, parameters)
    return WaterFields(moisture.astype(np.float32),
                       convert_moisture_to_pressure_head(moisture, parameters).astype(np.float32))
