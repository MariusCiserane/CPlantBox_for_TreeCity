"""Turn soil resistance and water status into the elongation scale CPlantBox reads.

"""
import numpy as np

FREE_GROWTH_RESISTANCE_MPA = 0.5
STOPPED_GROWTH_RESISTANCE_MPA = 2.5

FIELD_CAPACITY_SUCTION_CM = 100.0
WILTING_POINT_SUCTION_CM = 15000.0
MINIMUM_WATER_SCALE = 0.05
SMALLEST_SUCTION_CM = 1.0


def apply_bengough(penetrometer_mpa: np.ndarray) -> np.ndarray:
    """Mechanical impedance after Bengough et al. (2011): free below 0.5 MPa, stalled above 2.5.
    """
    span = STOPPED_GROWTH_RESISTANCE_MPA - FREE_GROWTH_RESISTANCE_MPA
    unimpeded = (STOPPED_GROWTH_RESISTANCE_MPA - penetrometer_mpa) / span
    return np.clip(unimpeded, 0.0, 1.0)


def apply_water_stress(pressure_head_cm: np.ndarray) -> np.ndarray:
    """Log-linear ramp between field capacity and wilting point."""
    suction = np.maximum(np.abs(pressure_head_cm), SMALLEST_SUCTION_CM)
    span = np.log10(WILTING_POINT_SUCTION_CM) - np.log10(FIELD_CAPACITY_SUCTION_CM)
    unimpeded = (np.log10(WILTING_POINT_SUCTION_CM) - np.log10(suction)) / span
    return np.clip(unimpeded, MINIMUM_WATER_SCALE, 1.0)


def compute_growth_scale(penetrometer_mpa: np.ndarray, pressure_head_cm: np.ndarray) -> np.ndarray:
    """ Liebig's law of the minimum: the most limiting factor governs, not the product."""
    return np.minimum(apply_bengough(penetrometer_mpa),apply_water_stress(pressure_head_cm)).astype(np.float32)
