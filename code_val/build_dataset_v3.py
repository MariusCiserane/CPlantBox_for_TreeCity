"""Genere le dataset v3 : une scene par configuration, en parallele et reprenable.

Une scene deja ecrite est ignoree, donc on peut interrompre et relancer. Le plan d'experience
est note dans design.json a la premiere execution et une relance avec d'autres reglages est
refusee : un identifiant de scene n'a de sens qu'a l'interieur d'un plan.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from generator.build import build_scene
from generator.scene import MASTER_SEED, MONTE_CARLO_RUNS, SCENE_COUNT, draw_scenes
from generator.writer import is_complete, write_scene
from grid import VoxelGrid
from network.intrusion import measure_largest_growth_step_cm
from network.walls import PIPE_EXCLUSION_MARGIN_CM
from secondary_growth.parameters import SecondaryGrowthParameters
from tree.growth import XML_PATH

DATASET_DIRECTORY = Path(__file__).resolve().parent / "dataset_v3"
PILOT_DIRECTORY = Path(__file__).resolve().parent / "pilot_v3"
PILOT_SCENES = 20
VTP_SUBDIRECTORY = "vtp"
GRID = VoxelGrid(34, 64, 64, 15.0, 9.0)


def guard_design(directory: Path, scenes: int, runs: int) -> None:
    design = {"scenes": scenes, "monte_carlo_runs": runs, "master_seed": MASTER_SEED,
              "grid": [GRID.depth_cells, GRID.row_cells, GRID.column_cells]}
    path = directory / "design.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != design:
        raise SystemExit(f"{path} decrit un autre plan d'experience : choisir un autre dossier")
    path.write_text(json.dumps(design, indent=2), encoding="utf-8")


def produce_scene(task: tuple) -> str:
    configuration, directory, write_vtp = task
    if is_complete(directory, configuration.identifier):
        return f"scene {configuration.identifier:5d} deja presente, ignoree"

    parameters = SecondaryGrowthParameters()
    example = build_scene(configuration, GRID, parameters,
                          directory / VTP_SUBDIRECTORY if write_vtp else None)
    write_scene(directory, configuration, example, GRID, parameters)

    diagnostics = example.diagnostics
    return (f"scene {configuration.identifier:5d}"
            f"  {len(configuration.street.utilities):2d} reseaux"
            f"  age {configuration.tree.age_years:5.1f} ans"
            f"  segments {diagnostics['segments_mean']:7.0f}"
            f"  cible {diagnostics['mean_probability']:.5f}"
            f" (major {diagnostics['mean_probability_major']:.5f})"
            f"  intrusions {diagnostics['intrusions'] or 'aucune'}"
            f"  {diagnostics['runtime_seconds']:5.1f} s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help=f"{PILOT_SCENES} scenes reparties")
    parser.add_argument("--scenes", type=int, default=SCENE_COUNT)
    parser.add_argument("--runs", type=int, default=MONTE_CARLO_RUNS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--vtp", action="store_true", help="ecrire aussi le vtp du premier tirage")
    parser.add_argument("--output", type=Path, default=None,
                        help="dossier de sortie (remplace dataset_v3/ ou pilot_v3/)")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    assert PIPE_EXCLUSION_MARGIN_CM > measure_largest_growth_step_cm(XML_PATH), (
        "la marge d'exclusion ne couvre pas le plus grand pas de croissance du XML")

    scenes = draw_scenes(arguments.scenes, GRID, arguments.runs)
    if arguments.output is not None:
        directory = arguments.output.resolve()
    elif arguments.pilot:
        directory = PILOT_DIRECTORY
    else:
        directory = DATASET_DIRECTORY
    if arguments.pilot:
        scenes = scenes[::max(1, len(scenes) // PILOT_SCENES)][:PILOT_SCENES]

    directory.mkdir(parents=True, exist_ok=True)
    guard_design(directory, arguments.scenes, arguments.runs)

    tasks = [(configuration, directory, arguments.vtp or arguments.pilot)
             for configuration in scenes]
    with Pool(arguments.workers) as pool:
        for message in pool.imap_unordered(produce_scene, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
