"""Grow a few contrasted Picea abies scenes and dump one VTP plus one occupancy grid each.
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
from paraview import write_voxel_grids
from soil.archetypes import PIT_SUBSTRATE_ARCHETYPE, find_archetype
from soil.field import build_soil_field
from soil.growth_response import (apply_bengough, apply_water_stress, compute_angle_scale,
                                  compute_growth_scale)
from soil.hydraulics import build_pressure_head_field
from soil.leak import apply_leaks, build_probe_leaks
from soil.pit import TREE_PIT_DEPTH_RANGE_CM, apply_planting_pit, build_pit_mask
from soil.profile import draw_soil_profile
from soil.wet_spot_geometry import build_wet_spot_lookup
from soil.wet_spots import build_probe_wet_spots, build_wet_spot_field
from tree.growth import build_interpolated_soil_grid, build_soil_grid, grow_root_system

SANDBOX_PARENT = Path(__file__).resolve().parent
SANDBOX_ARCHETYPE_IDENTIFIERS = (1,2, 3,4, 5,6, 7,8,9,10)  # loose, sandy dry, compacted, clayey wet, sand over clay, hardpan, hard base, compacted topsoil
SANDBOX_AGE_YEARS = 120.0
SANDBOX_SEED = 7
SANDBOX_RUNS = 10
SEED_STRIDE = 100
WET_SPOTS = build_probe_wet_spots()
LEAKS = build_probe_leaks()
INTERPOLATED_FIELD = "champ"
SDF_CONTROL = "sdf"


def measure_maximum_depth_cm(segments: np.ndarray) -> float:
    if len(segments) == 0:
        return 0.0
    return float(-min(segments[:, 2].min(), segments[:, 5].min()))


def measure_occupancy_shape(probability: np.ndarray, grid: VoxelGrid) -> tuple[float, float]:
    """Where the root mass sits: distance to the trunk and depth, weighted by occupancy.
    Segment counts are blind to f_sa by construction, since a scaled insertion angle moves
    roots around without changing how much length gets grown."""
    total = probability.sum()
    if total == 0.0:
        return 0.0, 0.0
    distances = grid.measure_trunk_distances_cm()[None, :, :]
    depths = grid.measure_layer_depths_cm()[:, None, None]
    return (float((probability * distances).sum() / total),
            float((probability * depths).sum() / total))


def grow_repeatedly(geometry, elongation_grid, angle_grid, water_lookup, grid: VoxelGrid,
                    vtp_path: Path):
    segment_counts, depths, occupancies = [], [], []
    for run in range(SANDBOX_RUNS):
        roots = grow_root_system(geometry, elongation_grid, SANDBOX_SEED + run * SEED_STRIDE,
                                 SANDBOX_AGE_YEARS, vtp_path if run == 0 else None,
                                 angle_grid=angle_grid, water_lookup=water_lookup)
        segment_counts.append(len(roots.segments))
        depths.append(measure_maximum_depth_cm(roots.segments))
        occupancies.append(voxelize_structural_roots(roots.segments, roots.subtypes, grid).grid)
    return np.array(segment_counts), np.array(depths), occupancies


def save_scene_fields(directory: Path, name: str, growth_scale: np.ndarray, mechanical: np.ndarray,
                      water: np.ndarray, occupancy: np.ndarray, angle_scale: np.ndarray | None,
                      grid: VoxelGrid) -> None:
    np.save(directory / f"{name}.npy", occupancy)
    fields = {"f_se": growth_scale, "bengough": mechanical, "eau": water, "occupancy": occupancy,
              "fuites": build_wet_spot_field(WET_SPOTS, grid)}
    if angle_scale is not None:
        fields["f_sa"] = angle_scale
    write_voxel_grids(directory / f"{name}.vti", fields, grid)


def build_water_lookup(water: np.ndarray, grid: VoxelGrid, source):
    """The SDF control describes the same places analytically, so a matching bias between the two
    says the interpolated field is read correctly."""
    if source == SDF_CONTROL:
        return build_wet_spot_lookup(WET_SPOTS)
    if source == INTERPOLATED_FIELD:
        return build_interpolated_soil_grid(water, grid)
    return None


def build_soil_v2(profile, grid, rng):
    # Shared noise, per-voxel Rosetta, planting pit, factors combined by their minimum
    ambient = build_soil_field(profile, grid, rng)
    pit = build_soil_field(draw_soil_profile(PIT_SUBSTRATE_ARCHETYPE, rng), grid, rng)
    field = apply_planting_pit(ambient, pit, build_pit_mask(
        float(rng.uniform(*TREE_PIT_HALF_SIZE_RANGE_CM)),
        float(rng.uniform(*TREE_PIT_DEPTH_RANGE_CM)), grid))
    field = apply_leaks(field, LEAKS, grid)
    pressure_head = build_pressure_head_field(field)
    return field, pressure_head, compute_growth_scale(field.penetrometer_mpa, pressure_head)


def build_soil_v1(profile, grid, rng):
    # Independent noise per channel, Rosetta per horizon, factors multiplied
    field = apply_leaks(sandbox_v1.build_soil_field(profile, grid, rng), LEAKS, grid)
    pressure_head = sandbox_v1.build_pressure_head_field(field, profile, grid)
    return field, pressure_head, sandbox_v1.compute_growth_scale(field.penetrometer_mpa, pressure_head)


def run_scene(archetype_identifier: int, grid: VoxelGrid, build_soil, directory: Path, scale_angle: bool, water_source) -> None:
    # Random soil profile, drawn before either pipeline so both modes get the same soil
    rng = np.random.default_rng([SANDBOX_SEED, archetype_identifier])
    profile = draw_soil_profile(find_archetype(archetype_identifier), rng)

    # Soil field  = Noised texture + density + moisture
    field, pressure_head, growth_scale = build_soil(profile, grid, rng)
    mechanical = apply_bengough(field.penetrometer_mpa)
    water = apply_water_stress(pressure_head)
    elongation_grid = build_soil_grid(growth_scale, grid)
    angle_scale = compute_angle_scale(field.penetrometer_mpa) if scale_angle else None
    angle_grid = build_soil_grid(angle_scale, grid) if scale_angle else None
    water_lookup = build_water_lookup(water, grid, water_source)

    growth_started = time.perf_counter()
    # Build geometry without obstacles, change family to see the difference
    geometry = build_growth_geometry(draw_obstacle_layout(NO_OBSTACLE, rng), grid)
    segment_counts, depths, occupancies = grow_repeatedly(
        geometry, elongation_grid, angle_grid, water_lookup, grid,
        directory / f"{profile.archetype_name}.vtp")
    growth_seconds = time.perf_counter() - growth_started

    ensemble = compute_occupancy_probability(occupancies)
    mean_radius_cm, mean_depth_cm = measure_occupancy_shape(ensemble.probability, grid)
    save_scene_fields(directory, profile.archetype_name, growth_scale, mechanical, water,
                      ensemble.probability, angle_scale, grid)

    print(f"{profile.archetype_name:<18s}"
          f" bengough {mechanical.mean():.3f}"
          f" eau {water.mean():.3f}"
          f" -> f_se {growth_scale.mean():.3f}"
          f"{f' f_sa {angle_scale.mean():.3f}' if scale_angle else ''}"
          f"{f' hydro[{water_source}]' if water_source else ''}")
    print(f"{'':<18s}"
          f" segments {segment_counts.mean():7.0f} +/- {segment_counts.std():5.0f}"
          f" profondeur {depths.mean():6.1f} +/- {depths.std():5.1f} cm"
          f" support {(ensemble.probability > 0).mean():.4f}"
          f" masse {ensemble.probability.mean():.5f}"
          f" rayon {mean_radius_cm:5.1f} cm profondeur_moy {mean_depth_cm:5.1f} cm"
          f" croissance {growth_seconds:6.2f} s", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", action="store_true",
                        help="pipeline sol de v1, pour comparer les memes scenes")
    parser.add_argument("--f-sa", action="store_true",
                        help="brancher aussi f_sa, l'angle d'insertion des laterales")
    parser.add_argument("--hydro", action="store_true",
                        help="brancher l'hydrotropisme sur le champ d'eau interpole")
    parser.add_argument("--hydro-sdf", action="store_true",
                        help="temoin : hydrotropisme sur les poches SDF analytiques")
    parser.add_argument("--output", type=Path, default=None,
                        help="dossier de sortie personnalisé (défaut: sandbox_v2 ou sandbox_v1)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    legacy = args.v1
    version = "v1" if legacy else "v2"
    build_soil = build_soil_v1 if legacy else build_soil_v2
    water_source = SDF_CONTROL if args.hydro_sdf else (INTERPOLATED_FIELD if args.hydro else None)
    suffix = "".join(part for part, enabled in (("_fsa", args.f_sa), ("_hydro", args.hydro),
                                                ("_hydrosdf", args.hydro_sdf)) if enabled)
    directory = args.output if args.output else SANDBOX_PARENT / f"sandbox_{version}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)

    grid = VoxelGrid()
    for identifier in SANDBOX_ARCHETYPE_IDENTIFIERS:
        run_scene(identifier, grid, build_soil, directory, args.f_sa, water_source)


if __name__ == "__main__":
    main()
