"""Generate the AI3RSS v1 training dataset with CPlantBox.

One configuration produces one example: several stochastic root systems grown in
the same environment, voxelized and averaged into an occupancy probability map.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from dataset.configuration import (
    DRAWS_PER_COMBINATION,
    MASTER_SEED,
    MONTE_CARLO_RUNS,
    draw_configurations,
)
from dataset.example import build_example
from dataset.writer import build_paths, write_example
from grid import VoxelGrid

HERE = Path(__file__).resolve().parent
DATASET_DIRECTORY = HERE / "dataset_v1"
PILOT_DIRECTORY = HERE / "pilot_v1"
PILOT_CONFIGURATIONS = 5
VTP_SUBDIRECTORY = "vtp"
DESIGN_FILE_NAME = "design.json"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def produce_example(task: tuple) -> str:
    configuration, directory = task
    archive_path, _ = build_paths(directory, configuration.identifier)
    if archive_path.exists():
        return f"configuration {configuration.identifier:04d} : deja presente, ignoree"
    example = build_example(configuration, VoxelGrid(), directory / VTP_SUBDIRECTORY)
    write_example(directory, configuration, example)
    return (f"configuration {configuration.identifier:04d} : "
            f"{example.diagnostics['runtime_seconds']:6.1f} s, "
            f"support {example.diagnostics['support_voxel_fraction']:.3f}, "
            f"masse {example.diagnostics['mean_probability']:.4f}, "
            f"{example.diagnostics['samples_outside_grid']} echantillons hors grille")


def select_pilot(configurations: list, count: int) -> list:
    """A spread across the archetypes rather than the first few of the list."""
    return configurations[:: max(1, len(configurations) // count)][:count]


def guard_design(directory: Path, draws: int, runs: int) -> None:
    """A configuration identifier only means something within one design: reusing a
    directory with different settings would silently mix incompatible examples."""
    design = {"draws_per_combination": draws, "monte_carlo_runs": runs, "master_seed": MASTER_SEED}
    path = directory / DESIGN_FILE_NAME
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != design:
        raise SystemExit(f"{path} decrit un autre plan d'experience. Vider {directory} "
                         "ou ecrire dans un autre dossier.")
    path.write_text(json.dumps(design, indent=2), encoding="utf-8")


def run(configurations: list, directory: Path, workers: int, draws: int, runs: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    guard_design(directory, draws, runs)
    print(f"{len(configurations)} configurations, {workers} process , output {directory}")
    tasks = [(configuration, directory) for configuration in configurations]
    with Pool(workers) as pool:
        for message in pool.imap_unordered(produce_example, tasks):
            print(message, flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Pilot configurations are drawn from the archetypes, not from the dataset.
    parser.add_argument("--pilot", action="store_true",
                        help=f"{PILOT_CONFIGURATIONS} configurations reparties sur les archetypes")
    # Draws per combination are drawn from the dataset, not from the archetypes.
    parser.add_argument("--draws", type=int, default=DRAWS_PER_COMBINATION,
                        help="tirages par combinaison sol x obstacle")
    # Monte-Carlo runs are drawn from the dataset, not from the archetypes.
    parser.add_argument("--runs", type=int, default=MONTE_CARLO_RUNS,
                        help="tirages Monte-Carlo par configuration")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.pilot:
        configurations = draw_configurations(monte_carlo_runs=arguments.runs)
        run(select_pilot(configurations, PILOT_CONFIGURATIONS), PILOT_DIRECTORY,
            arguments.workers, arguments.draws, arguments.runs)
        return
    configurations = draw_configurations(draws_per_combination=arguments.draws, monte_carlo_runs=arguments.runs)
    run(configurations, DATASET_DIRECTORY, arguments.workers, arguments.draws, arguments.runs)


if __name__ == "__main__":
    main()
