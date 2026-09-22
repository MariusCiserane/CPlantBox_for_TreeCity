"""Engineered materials written into the soil field, blended by volume fraction.

The planting pit swaps whole voxels because it is metres across. A pipe surround is
centimetres across, so a hard swap would round it to the nearest voxel and lose it; the
fraction of the voxel the material actually occupies is blended in instead.

Zones overlap - a trench is cut through the road body, a pipe sits inside its trench - so
the list is a priority order consumed against a decreasing budget. That is also what carves
the rings without any shape having to be described as one: a lumen reserves its volume and
writes nothing, the surround takes what is left inside it, the backfill takes the rest.
"""
from dataclasses import fields, replace
from typing import Optional

import numpy as np

from soil.archetypes import SoilHorizon
from soil.field import SoilField


def apply_fills(field: SoilField, horizons: tuple[Optional[SoilHorizon], ...],
                coverages: tuple[np.ndarray, ...]) -> SoilField:
    remaining = np.ones(field.penetrometer_mpa.shape, np.float32)
    for horizon, coverage in zip(horizons, coverages):
        share = np.minimum(coverage, remaining)
        remaining = remaining - share
        if horizon is None:
            continue
        blended = {channel.name: (getattr(field, channel.name) * (1.0 - share)
                                  + getattr(horizon, channel.name) * share).astype(np.float32)
                   for channel in fields(field)}
        field = replace(field, **blended)
    return field


def measure_natural_share(coverages: tuple[np.ndarray, ...], shape: tuple[int, ...]) -> np.ndarray:
    """What is left of the original soil once every engineered zone has taken its part."""
    if not coverages:
        return np.ones(shape, np.float32)
    return (1.0 - np.clip(sum(coverages), 0.0, 1.0)).astype(np.float32)
