"""Injecting a leak's water into the soil field.

The plume geometry lives in obstacles/leakage.py. What happens here is the injection, plus
the one side effect a leak has that no other source does: wet soil is softer, so the
penetration resistance has to follow the moisture that just moved.
"""
from dataclasses import replace

import numpy as np

from soil.field import SoilField, apply_water_to_resistance


def apply_leaks(field: SoilField, sources: tuple[tuple[np.ndarray, float], ...],
                natural_share: np.ndarray) -> SoilField:
    """Blending towards the ambient soil rather than overwriting it keeps the ramp sloping all the
    way out to the reach; taking a maximum instead would flatten it wherever the soil is already
    wetter than the tail of the ramp, and a flat field steers nothing.

    Rescaling the resistance here is what stops a leak from reaching the water half of f_se
    while f_sa and the mechanical half never noticed.
    """
    fraction = field.available_water_fraction
    for ramp, target_fraction in sources:
        fraction = fraction + (target_fraction - fraction) * ramp
    wetted = replace(field, available_water_fraction=fraction.astype(np.float32))
    return apply_water_to_resistance(wetted, field.available_water_fraction, natural_share)
