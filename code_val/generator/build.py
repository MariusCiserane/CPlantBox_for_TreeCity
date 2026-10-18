"""Scene description, from its configuration to its voxels.
"""
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from dataset.tensors import build_prior
from generator.environment import build_environment
from generator.polylines import extract_polydata
from generator.scene import SceneConfiguration
from generator.tensors import build_film_vector, build_input_tensor
from generator.voxelise import (DIAMETER_CLASSES, accumulate_volume, build_polysegments,
                                sample_segments, select_structural, stamp_classes)
from grid import VoxelGrid
from hybrid.prior import LEAK_PRIOR_BOOST, build_prior_v2
from network.intrusion import find_intrusions
from secondary_growth.parameters import SecondaryGrowthParameters
from secondary_growth.radii import build_final_radii
from tree.growth import grow_plant
from tree.species import secondary_growth_for

VTP_RUNS_PER_SCENE = 1
PACKED_OCCUPANCIES = ("radius", "centreline", "radius_major", "radius_minor")


@dataclass(frozen=True)
class GrowthOptions:
    """Which soil hooks CPlantBox reads. The dataset always wires both of them; the sandbox can
    switch one off to isolate its effect."""
    use_angle_scale: bool = True
    use_hydrotropism: bool = True


@dataclass
class Example:
    inputs: np.ndarray
    film: np.ndarray
    prior_v2: np.ndarray
    hit_counts: dict
    packed_runs: dict
    root_volume: np.ndarray
    polylines: dict
    fields: dict
    diagnostics: dict


def measure_maximum_depth_cm(starts: np.ndarray, ends: np.ndarray) -> float:
    if len(starts) == 0:
        return 0.0
    return float(-min(starts[:, 2].min(), ends[:, 2].min()))


def grow_once(configuration: SceneConfiguration, environment, grid: VoxelGrid, seed: int,
              parameters: SecondaryGrowthParameters, vtp_path,
              options: GrowthOptions = GrowthOptions()) -> tuple:
    plant = grow_plant(environment.geometry, environment.elongation_grid, seed,
                       configuration.tree.age_years, Path(configuration.tree_xml), vtp_path,
                       angle_grid=environment.angle_grid if options.use_angle_scale else None,
                       water_lookup=environment.water_lookup if options.use_hydrotropism else None)
    polydata = extract_polydata(plant)
    radii, _ = build_final_radii(polydata, parameters)
    segments = build_polysegments(polydata, radii)

    occupancies, outside = stamp_classes(sample_segments(select_structural(segments)), grid)
    return (occupancies, accumulate_volume(sample_segments(segments), grid),
            polydata, radii, segments, outside)


def build_fields(environment, hit_counts: dict, runs: int) -> dict:
    return {"f_se": environment.growth_scale,
            "f_sa": environment.angle_scale,
            "bengough": environment.bengough,
            "eau": environment.water_scale,
            "theta": environment.moisture_m3_m3,
            "psi": environment.pressure_head,
            "occupancy": hit_counts["radius"] / runs,
            "occupancy_major": hit_counts["radius_major"] / runs,
            "occupancy_minor": hit_counts["radius_minor"] / runs,
            "occupancy_mediane": hit_counts["centreline"] / runs,
            "fuites": environment.leak_plume,
            "tuyaux": environment.pipe_labels,
            "remblais": environment.backfill_fraction,
            "patches": environment.patch_labels}


def build_scene(configuration: SceneConfiguration, grid: VoxelGrid,
                parameters: Optional[SecondaryGrowthParameters] = None,
                vtp_directory: Path = None, options: GrowthOptions = GrowthOptions(),
                vtp_runs: int = VTP_RUNS_PER_SCENE, observer: Optional[Callable] = None) -> Example:
    """parameters: secondary growth, by default the one of the species of the tree.
    vtp_runs: how many runs write their .vtp, when vtp_directory is given.
    observer(run_index, polydata, radii, segments): called after each run, to measure the roots
    without a second loop (run_index starts at 0)."""
    if parameters is None:
        parameters = secondary_growth_for(configuration.tree.species)
    started = time.perf_counter()
    environment = build_environment(configuration, grid)
    soil_seconds = time.perf_counter() - started

    stacks, volumes = defaultdict(list), []
    segment_counts, depths, intrusions, outside = [], [], {}, 0
    polylines = {}

    growth_started = time.perf_counter()
    for run, seed in enumerate(configuration.root_seeds, start=1):
        path = None if vtp_directory is None or run > vtp_runs else \
            vtp_directory / f"scene_{configuration.identifier:05d}_run_{run:02d}.vtp"
        occupancies, volume, polydata, radii, segments, missed = grow_once(
            configuration, environment, grid, seed, parameters, path, options)
        if observer is not None:
            observer(run - 1, polydata, radii, segments)

        for name, occupancy in occupancies.items():
            stacks[name].append(occupancy)
        volumes.append(volume)
        segment_counts.append(len(segments.starts))
        depths.append(measure_maximum_depth_cm(segments.starts, segments.ends))
        outside += missed
        for name, depth in find_intrusions(np.hstack([segments.starts, segments.ends]),
                                           environment.lumens).items():
            intrusions[name] = max(intrusions.get(name, 0.0), depth)
        if run == 1:
            polylines = {"points": polydata.points.astype(np.float32),
                         "offsets": polydata.offsets.astype(np.int32),
                         "radius": radii.astype(np.float32),
                         "subtype": polydata.cell_arrays["subType"].astype(np.int16)}
    growth_seconds = time.perf_counter() - growth_started

    runs = len(configuration.root_seeds)
    hit_counts = {name: np.sum(stack, axis=0, dtype=np.uint16) for name, stack in stacks.items()}
    target = hit_counts["radius"] / runs
    prior = build_prior(configuration.tree, grid)

    return Example(
        inputs=build_input_tensor(environment, prior, grid),
        film=build_film_vector(configuration),
        prior_v2=build_prior_v2(prior, environment.growth_scale, environment.pipe_fraction,
                                environment.leak_plume, LEAK_PRIOR_BOOST),
        hit_counts=hit_counts,
        packed_runs={name: np.packbits(np.stack(stacks[name])) for name in PACKED_OCCUPANCIES},
        root_volume=np.mean(volumes, axis=0).astype(np.float32),
        polylines=polylines,
        fields=build_fields(environment, hit_counts, runs),
        diagnostics={"tree_xml": configuration.tree_xml,
                     "growth_options": {"angle_scale": options.use_angle_scale,
                                        "hydrotropism": options.use_hydrotropism},
                     "runtime_seconds": round(time.perf_counter() - started, 1),
                     "soil_seconds": round(soil_seconds, 1),
                     "growth_seconds": round(growth_seconds, 1),
                     "segments_mean": float(np.mean(segment_counts)),
                     "segments_std": float(np.std(segment_counts)),
                     "max_depth_cm_mean": float(np.mean(depths)),
                     "max_depth_cm_std": float(np.std(depths)),
                     "mean_probability": float(target.mean()),
                     "mean_probability_centreline": float(hit_counts["centreline"].mean() / runs),
                     **{f"mean_probability_{name}": float(hit_counts[f"radius_{name}"].mean() / runs)
                        for name in DIAMETER_CLASSES},
                     "support_voxel_fraction": float((target > 0).mean()),
                     "samples_outside_grid": int(outside),
                     "intrusions": {name: float(depth) for name, depth in intrusions.items()}},
    )
