"""Turn soil resistance and water status into the elongation scale CPlantBox reads.

Both responses are piecewise linear on a quantity that is already comparable across soils:
penetration resistance in MPa, and the share of the available water capacity that is left.
"""
import numpy as np

FREE_GROWTH_RESISTANCE_MPA = 0.5
STOPPED_GROWTH_RESISTANCE_MPA = 2.5

# Below this share of the available water capacity the reserve stops being freely extractable.
# The transpiration literature places the equivalent threshold at 0.3 to 0.4 across species.
WATER_STRESS_THRESHOLD = 0.40
MINIMUM_WATER_SCALE = 0.05
MINIMUM_ANGLE_SCALE = 0.35


def apply_bengough(penetrometer_mpa: np.ndarray) -> np.ndarray:
    """Mechanical impedance after Bengough et al. (2011): free below 0.5 MPa, stalled above 2.5"""
    span = STOPPED_GROWTH_RESISTANCE_MPA - FREE_GROWTH_RESISTANCE_MPA
    unimpeded = (STOPPED_GROWTH_RESISTANCE_MPA - penetrometer_mpa) / span
    return np.clip(unimpeded, 0.0, 1.0)


def apply_water_stress(available_water_fraction: np.ndarray) -> np.ndarray:
    """Full speed while the reserve is above the threshold, then a linear fall to the floor."""
    unimpeded = available_water_fraction / WATER_STRESS_THRESHOLD
    return np.clip(unimpeded, MINIMUM_WATER_SCALE, 1.0)


def compute_growth_scale(penetrometer_mpa: np.ndarray, available_water_fraction: np.ndarray) -> np.ndarray:
    return (apply_bengough(penetrometer_mpa)
            * apply_water_stress(available_water_fraction)).astype(np.float32)


def compute_angle_scale(penetrometer_mpa: np.ndarray) -> np.ndarray:
    """Insertion angle of a lateral, narrowed by mechanical impedance: a lateral that cannot
    push sideways emerges closer to its parent's heading.
    """
    unimpeded = apply_bengough(penetrometer_mpa)
    return (MINIMUM_ANGLE_SCALE + (1.0 - MINIMUM_ANGLE_SCALE) * unimpeded).astype(np.float32)
