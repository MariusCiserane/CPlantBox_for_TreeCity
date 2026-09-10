"""Build one training example: soil, obstacles, Monte Carlo growth, tensors."""
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from dataset.configuration import Configuration
from dataset.occupancy import compute_occupancy_probability, voxelize_structural_roots
from dataset.tensors import build_film_vector, build_input_tensor
from grid import VoxelGrid
from obstacles.geometry import build_growth_geometry
from obstacles.layouts import ROAD_BASE
from obstacles.masks import build_obstacle_masks
from soil.field import apply_road_base_override, build_soil_field
from soil.growth_response import compute_growth_scale
from soil.hydraulics import build_pressure_head_field
from tree.growth import build_elongation_grid, grow_root_system

VTP_RUNS_PER_CONFIGURATION = 1
VTP_NAME_TEMPLATE = "example_{identifier:04d}_run_{run:02d}.vtp"


class Example(NamedTuple):
    inputs: np.ndarray
    film: np.ndarray
    probability: np.ndarray
    hit_count: np.ndarray
    occupancy_runs: np.ndarray
    diagnostics: dict


def build_soil_environment(configuration: Configuration, grid: VoxelGrid):
    rng = np.random.default_rng(configuration.soil_noise_seed)
    field = build_soil_field(configuration.soil, grid, rng)
    masks = build_obstacle_masks(configuration.obstacles, grid)
    field = apply_road_base_override(field, masks.road_base, ROAD_BASE)
    pressure_head = build_pressure_head_field(field)
    return field, masks, compute_growth_scale(field.penetrometer_mpa, pressure_head)


def build_vtp_path(directory: Optional[Path], identifier: int, run: int) -> Optional[Path]:
    if directory is None or run > VTP_RUNS_PER_CONFIGURATION:
        return None
    return directory / VTP_NAME_TEMPLATE.format(identifier=identifier, run=run)


def build_example(configuration: Configuration, grid: VoxelGrid,
                  vtp_directory: Optional[Path] = None) -> Example:
    started = time.time()
    field, masks, growth_scale = build_soil_environment(configuration, grid)
    # CPlantBox tropisms hold the geometry through a weak reference, so it has to
    # stay referenced here for the whole Monte Carlo loop.
    geometry = build_growth_geometry(configuration.obstacles, grid)
    elongation_grid = build_elongation_grid(growth_scale, grid)

    occupancies, outside_samples = [], 0
    for run, seed in enumerate(configuration.root_seeds, start=1):
        roots = grow_root_system(geometry, elongation_grid, seed, configuration.tree.age_years,
                                 build_vtp_path(vtp_directory, configuration.identifier, run))
        occupancy = voxelize_structural_roots(roots.segments, roots.subtypes, grid)
        occupancies.append(occupancy.grid)
        outside_samples += occupancy.outside_samples

    ensemble = compute_occupancy_probability(occupancies)
    diagnostics = {
        "runtime_seconds": round(time.time() - started, 1),
        "samples_outside_grid": outside_samples,
        # Le support depend du nombre de tirages, la masse non : c'est elle qui donne
        # le desequilibre de classes vu par la fonction de perte.
        "support_voxel_fraction": float((ensemble.probability > 0).mean()),
        "mean_probability": float(ensemble.probability.mean()),
        "mean_growth_scale": float(growth_scale.mean()),
    }
    return Example(
        inputs=build_input_tensor(field, masks, configuration.tree, grid),
        film=build_film_vector(configuration.tree, configuration.soil.topsoil_depth_cm),
        probability=ensemble.probability,
        hit_count=ensemble.hit_count,
        occupancy_runs=np.packbits(np.stack(occupancies)),
        diagnostics=diagnostics,
    )
