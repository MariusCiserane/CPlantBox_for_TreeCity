"""Scene description, from its configuration to its voxels.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generator.environment import build_environment
from generator.polylines import extract_polydata
from generator.scene import SceneConfiguration
from generator.tensors import build_film_vector, build_input_tensor
from generator.voxelise import (accumulate_volume, build_polysegments, sample_segments,
                                select_structural, stamp_centreline, stamp_radius)
from grid import VoxelGrid
from network.intrusion import find_intrusions
from secondary_growth.parameters import SecondaryGrowthParameters
from secondary_growth.radii import build_final_radii
from soil.growth_response import apply_bengough
from tree.growth import grow_plant

VTP_RUNS_PER_SCENE = 1


@dataclass
class Example:
    inputs: np.ndarray
    film: np.ndarray
    target_radius: np.ndarray
    target_centreline: np.ndarray
    root_volume: np.ndarray
    hit_count_radius: np.ndarray
    hit_count_centreline: np.ndarray
    radius_runs: np.ndarray
    centreline_runs: np.ndarray
    polylines: dict
    fields: dict
    diagnostics: dict


def measure_maximum_depth_cm(starts: np.ndarray, ends: np.ndarray) -> float:
    if len(starts) == 0:
        return 0.0
    return float(-min(starts[:, 2].min(), ends[:, 2].min()))


def grow_once(configuration: SceneConfiguration, environment, grid: VoxelGrid, seed: int,
              parameters: SecondaryGrowthParameters, vtp_path) -> tuple:
    plant = grow_plant(environment.geometry, environment.elongation_grid, seed,
                       configuration.tree.age_years, vtp_path,
                       water_lookup=environment.water_lookup)
    polydata = extract_polydata(plant)
    radii, _ = build_final_radii(polydata, parameters)
    segments = build_polysegments(polydata, radii)

    structural = sample_segments(select_structural(segments))
    centreline, outside = stamp_centreline(structural, grid)
    return (centreline, stamp_radius(structural, grid),
            accumulate_volume(sample_segments(segments), grid),
            polydata, radii, segments, outside)


def build_scene(configuration: SceneConfiguration, grid: VoxelGrid,
                parameters: SecondaryGrowthParameters = SecondaryGrowthParameters(),
                vtp_directory: Path = None) -> Example:
    started = time.perf_counter()
    environment = build_environment(configuration, grid)
    soil_seconds = time.perf_counter() - started

    centrelines, radius_grids, volumes = [], [], []
    segment_counts, depths, intrusions, outside = [], [], {}, 0
    polylines = {}

    growth_started = time.perf_counter()
    for run, seed in enumerate(configuration.root_seeds, start=1):
        path = None if vtp_directory is None or run > VTP_RUNS_PER_SCENE else \
            vtp_directory / f"scene_{configuration.identifier:05d}_run_{run:02d}.vtp"
        centreline, radius_grid, volume, polydata, radii, segments, missed = grow_once(
            configuration, environment, grid, seed, parameters, path)

        centrelines.append(centreline)
        radius_grids.append(radius_grid)
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

    hit_radius = np.sum(radius_grids, axis=0, dtype=np.uint16)
    hit_centreline = np.sum(centrelines, axis=0, dtype=np.uint16)
    runs = len(configuration.root_seeds)
    target_radius = (hit_radius / runs).astype(np.float32)
    target_centreline = (hit_centreline / runs).astype(np.float32)
    root_volume = np.mean(volumes, axis=0).astype(np.float32)

    return Example(
        inputs=build_input_tensor(environment, configuration, grid),
        film=build_film_vector(configuration),
        target_radius=target_radius,
        target_centreline=target_centreline,
        root_volume=root_volume,
        hit_count_radius=hit_radius,
        hit_count_centreline=hit_centreline,
        radius_runs=np.packbits(np.stack(radius_grids)),
        centreline_runs=np.packbits(np.stack(centrelines)),
        polylines=polylines,
        fields={"f_se": environment.growth_scale,
                "bengough": apply_bengough(environment.field.penetrometer_mpa).astype(np.float32),
                "eau": environment.water_scale,
                "psi": environment.pressure_head,
                "occupancy": target_radius,
                "occupancy_mediane": target_centreline,
                "fuites": environment.leak_plume,
                "tuyaux": environment.pipe_labels,
                "remblais": environment.backfill_fraction,
                "patches": environment.patch_labels},
        diagnostics={"runtime_seconds": round(time.perf_counter() - started, 1),
                     "soil_seconds": round(soil_seconds, 1),
                     "growth_seconds": round(growth_seconds, 1),
                     "segments_mean": float(np.mean(segment_counts)),
                     "segments_std": float(np.std(segment_counts)),
                     "max_depth_cm_mean": float(np.mean(depths)),
                     "max_depth_cm_std": float(np.std(depths)),
                     "mean_probability": float(target_radius.mean()),
                     "mean_probability_centreline": float(target_centreline.mean()),
                     "support_voxel_fraction": float((target_radius > 0).mean()),
                     "samples_outside_grid": int(outside),
                     "intrusions": {name: float(depth) for name, depth in intrusions.items()}},
    )
