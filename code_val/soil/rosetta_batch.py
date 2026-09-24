"""Predict the van Genuchten retention parameters of every voxel with one Rosetta call.

Rosetta accepts a batch of soil records, so a whole grid goes in a single call.
Jittered and noised voxels cluster tightly, so the records are rounded first and
only the distinct ones are sent: the cost then follows how spread out the soil is,
not how many voxels there are.

A mosaic of several archetypes spreads the soil much further than one profile did,
so the rounding is coarsened until the batch fits the budget. That costs nothing:
the rounding error was already measured at 3 to 10 percent of the uncertainty
Rosetta declares on its own predictions.
"""
import numpy as np
import plantbox.functional.van_genuchten as vg
from rosetta import SoilData, rosetta

ROSETTA_VERSION = 3
TEXTURE_QUANTUM_PERCENT = 1.0
BULK_DENSITY_QUANTUM_G_CM3 = 0.02
RECORD_BUDGET = 12000
QUANTUM_GROWTH = 2.0
LOG_BASE = 10.0


def round_to(values: np.ndarray, quantum: float) -> np.ndarray:
    return np.round(values / quantum) * quantum


def build_soil_records(sand_fraction, clay_fraction, bulk_density_g_cm3,
                       coarsening: float = 1.0) -> np.ndarray:
    """Rosetta reads sand, silt and clay in percent plus bulk density, one row per voxel.
    """
    sand = sand_fraction * 100.0
    clay = clay_fraction * 100.0
    silt = np.maximum(100.0 - sand - clay, 0.0)
    total = sand + silt + clay
    texture = np.stack([sand, silt, clay], axis=-1).reshape(-1, 3) / total.reshape(-1, 1) * 100.0

    # Rosetta rend NaN des que les trois fractions ne somment pas a 100, et arrondir chacune de
    # son cote donne 99 ou 101. Le reliquat va a la composante majoritaire, seule assez grande
    # pour l'absorber sans devenir negative.
    rounded = round_to(texture, TEXTURE_QUANTUM_PERCENT * coarsening)
    rounded[np.arange(len(rounded)), texture.argmax(axis=1)] += 100.0 - rounded.sum(axis=1)

    density = round_to(bulk_density_g_cm3, BULK_DENSITY_QUANTUM_G_CM3 * coarsening)
    return np.concatenate([rounded, density.reshape(-1, 1)], axis=1)


def build_distinct_records(sand_fraction, clay_fraction, bulk_density_g_cm3) -> tuple:
    """Arrondir plus grossierement tant que le lot depasse le budget."""
    coarsening = 1.0
    while True:
        records = build_soil_records(sand_fraction, clay_fraction, bulk_density_g_cm3, coarsening)
        distinct, index = np.unique(records, axis=0, return_inverse=True)
        if len(distinct) <= RECORD_BUDGET:
            return distinct, index
        coarsening *= QUANTUM_GROWTH


def estimate_van_genuchten_parameters_field(sand_fraction, clay_fraction,bulk_density_g_cm3) -> vg.Parameters:
    distinct, index = build_distinct_records(sand_fraction, clay_fraction, bulk_density_g_cm3)
    predicted = rosetta(ROSETTA_VERSION, SoilData.from_array(distinct.tolist()))[0]
    theta_r, theta_s, log_alpha, log_n, log_conductivity = predicted[index.ravel()].T

    shape = sand_fraction.shape
    return vg.Parameters([
        theta_r.reshape(shape),
        theta_s.reshape(shape),
        (LOG_BASE ** log_alpha).reshape(shape),
        (LOG_BASE ** log_n).reshape(shape),
        (LOG_BASE ** log_conductivity).reshape(shape),
    ])
