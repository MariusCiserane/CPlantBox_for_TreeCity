"""Predict the van Genuchten retention parameters of every voxel with one Rosetta call.

Rosetta accepts a batch of soil records, so a whole grid goes in a single call.
Jittered and noised voxels cluster tightly, so the records are rounded first and
only the distinct ones are sent: the cost then follows how spread out the soil is,
not how many voxels there are.
"""
import numpy as np
import plantbox.functional.van_genuchten as vg
from rosetta import SoilData, rosetta

ROSETTA_VERSION = 3
TEXTURE_QUANTUM_PERCENT = 1.0
BULK_DENSITY_QUANTUM_G_CM3 = 0.02
LOG_BASE = 10.0


def round_to(values: np.ndarray, quantum: float) -> np.ndarray:
    return np.round(values / quantum) * quantum


def build_soil_records(sand_fraction, clay_fraction, bulk_density_g_cm3) -> np.ndarray:
    """Rosetta reads sand, silt and clay in percent plus bulk density, one row per voxel.
    """
    sand = sand_fraction * 100.0
    clay = clay_fraction * 100.0
    silt = np.maximum(100.0 - sand - clay, 0.0)
    total = sand + silt + clay
    return np.stack([
        round_to(sand / total * 100.0, TEXTURE_QUANTUM_PERCENT),
        round_to(silt / total * 100.0, TEXTURE_QUANTUM_PERCENT),
        round_to(clay / total * 100.0, TEXTURE_QUANTUM_PERCENT),
        round_to(bulk_density_g_cm3, BULK_DENSITY_QUANTUM_G_CM3),
    ], axis=-1).reshape(-1, 4)


def estimate_van_genuchten_parameters_field(sand_fraction, clay_fraction,bulk_density_g_cm3) -> vg.Parameters:
    # Build the input records
    records = build_soil_records(sand_fraction, clay_fraction, bulk_density_g_cm3)

    # Voxel reduction
    distinct, index = np.unique(records, axis=0, return_inverse=True)
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
