"""Grow a few contrasted Picea abies scenes and dump one VTP plus one occupancy grid each.

A fast feedback loop for validating one soil change at a time: no input tensors,
no dataset bookkeeping. Several seeds per scene, because a single root system is
a draw, not a measurement: the soil hooks perturb the random stream even where
they change nothing, so one run cannot tell an effect from noise.

With --v1 the same scenes are grown through the old soil pipeline instead. The
soil profile and the CPlantBox seeds are drawn before either pipeline runs, so
the two modes differ only in how the soil becomes an f_se grid, and their VTP
files carry the same names in sibling directories for side-by-side reading.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sandbox_v1
from dataset.occupancy import compute_occupancy_probability, voxelize_structural_roots
from grid import VoxelGrid
from obstacles.geometry import build_growth_geometry
from obstacles.layouts import NO_OBSTACLE, TREE_PIT_HALF_SIZE_RANGE_CM, draw_obstacle_layout
from soil.archetypes import PIT_SUBSTRATE_ARCHETYPE, find_archetype
from soil.field import build_soil_field
from soil.growth_response import apply_bengough, apply_water_stress, compute_growth_scale
from soil.hydraulics import build_pressure_head_field
from soil.pit import TREE_PIT_DEPTH_RANGE_CM, apply_planting_pit, build_pit_mask
from soil.profile import draw_soil_profile
from tree.growth import build_elongation_grid, grow_root_system

SANDBOX_PARENT = Path(__file__).resolve().parent
SANDBOX_ARCHETYPE_IDENTIFIERS = (1,2, 3,4, 5,6, 7,8,9,10)  # loose, sandy dry, compacted, clayey wet, sand over clay, hardpan, hard base, compacted topsoil
SANDBOX_AGE_YEARS = 120.0
SANDBOX_SEED = 7
SANDBOX_RUNS = 10
SEED_STRIDE = 100


def measure_maximum_depth_cm(segments: np.ndarray) -> float:
    if len(segments) == 0:
        return 0.0
    return float(-min(segments[:, 2].min(), segments[:, 5].min()))


def grow_repeatedly(geometry, elongation_grid, grid: VoxelGrid, vtp_path: Path):
    segment_counts, depths, occupancies = [], [], []
    for run in range(SANDBOX_RUNS):
        roots = grow_root_system(geometry, elongation_grid, SANDBOX_SEED + run * SEED_STRIDE,
                                 SANDBOX_AGE_YEARS, vtp_path if run == 0 else None)
        segment_counts.append(len(roots.segments))
        depths.append(measure_maximum_depth_cm(roots.segments))
        occupancies.append(voxelize_structural_roots(roots.segments, roots.subtypes, grid).grid)
    return np.array(segment_counts), np.array(depths), occupancies


def build_soil_v2(profile, grid, rng):
    # Shared noise, per-voxel Rosetta, planting pit, factors combined by their minimum
    ambient = build_soil_field(profile, grid, rng)
    pit = build_soil_field(draw_soil_profile(PIT_SUBSTRATE_ARCHETYPE, rng), grid, rng)
    field = apply_planting_pit(ambient, pit, build_pit_mask(
        float(rng.uniform(*TREE_PIT_HALF_SIZE_RANGE_CM)),
        float(rng.uniform(*TREE_PIT_DEPTH_RANGE_CM)), grid))
    pressure_head = build_pressure_head_field(field)
    return field, pressure_head, compute_growth_scale(field.penetrometer_mpa, pressure_head)


def build_soil_v1(profile, grid, rng):
    # Independent noise per channel, Rosetta per horizon, factors multiplied
    field = sandbox_v1.build_soil_field(profile, grid, rng)
    pressure_head = sandbox_v1.build_pressure_head_field(field, profile, grid)
    return field, pressure_head, sandbox_v1.compute_growth_scale(field.penetrometer_mpa, pressure_head)


def run_scene(archetype_identifier: int, grid: VoxelGrid, build_soil, directory: Path) -> None:
    # Random soil profile, drawn before either pipeline so both modes get the same soil
    rng = np.random.default_rng([SANDBOX_SEED, archetype_identifier])
    profile = draw_soil_profile(find_archetype(archetype_identifier), rng)

    # Soil field  = Noised texture + density + moisture
    field, pressure_head, growth_scale = build_soil(profile, grid, rng)
    mechanical = apply_bengough(field.penetrometer_mpa)
    water = apply_water_stress(pressure_head)
    elongation_grid = build_elongation_grid(growth_scale, grid)

    growth_started = time.perf_counter()
    # Build geometry without obstacles, change family to see the difference
    geometry = build_growth_geometry(draw_obstacle_layout(NO_OBSTACLE, rng), grid)
    segment_counts, depths, occupancies = grow_repeatedly(geometry, elongation_grid, grid, directory / f"{profile.archetype_name}.vtp")
    growth_seconds = time.perf_counter() - growth_started

    ensemble = compute_occupancy_probability(occupancies)
    np.save(directory / f"{profile.archetype_name}.npy", ensemble.probability)

    print(f"{profile.archetype_name:<18s}"
          f" bengough {mechanical.mean():.3f}"
          f" eau {water.mean():.3f}"
          f" -> f_se {growth_scale.mean():.3f}")
    print(f"{'':<18s}"
          f" segments {segment_counts.mean():7.0f} +/- {segment_counts.std():5.0f}"
          f" profondeur {depths.mean():6.1f} +/- {depths.std():5.1f} cm"
          f" support {(ensemble.probability > 0).mean():.4f}"
          f" masse {ensemble.probability.mean():.5f}"
          f" croissance {growth_seconds:6.2f} s", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", action="store_true",
                        help="pipeline sol de v1, pour comparer les memes scenes")
    parser.add_argument("--output", type=Path, default=None,
                        help="dossier de sortie personnalisé (défaut: sandbox_v2 ou sandbox_v1)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    legacy = args.v1
    version = "v1" if legacy else "v2"
    build_soil = build_soil_v1 if legacy else build_soil_v2
    directory = args.output if args.output else SANDBOX_PARENT / f"sandbox_{version}"
    directory.mkdir(parents=True, exist_ok=True)

    grid = VoxelGrid()
    print(f"pipeline sol {version}, {SANDBOX_RUNS} graines, {SANDBOX_AGE_YEARS:.0f} ans, "
          f"sortie {directory}")
    for identifier in SANDBOX_ARCHETYPE_IDENTIFIERS:
        run_scene(identifier, grid, build_soil, directory)


if __name__ == "__main__":
    main()
