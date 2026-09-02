"""Write one example to disk: arrays in an .npz, description in a .json.

The individual Monte Carlo runs are kept in packed bit form so the probability
map can be recomputed from any subset of them without simulating again.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dataset.configuration import Configuration
from dataset.example import Example
from dataset.tensors import CHANNEL_NAMES, CHANNEL_SCHEMA

EXAMPLE_NAME_TEMPLATE = "example_{identifier:04d}"


def build_paths(directory: Path, identifier: int) -> tuple[Path, Path]:
    stem = directory / EXAMPLE_NAME_TEMPLATE.format(identifier=identifier)
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def build_descriptor(configuration: Configuration, example: Example) -> dict:
    return {
        "identifier": configuration.identifier,
        "channel_schema": CHANNEL_SCHEMA,
        "channel_names": list(CHANNEL_NAMES),
        "soil": asdict(configuration.soil),
        "obstacles": asdict(configuration.obstacles),
        "tree": asdict(configuration.tree),
        "soil_noise_seed": configuration.soil_noise_seed,
        "root_seeds": list(configuration.root_seeds),
        "diagnostics": example.diagnostics,
    }


def write_example(directory: Path, configuration: Configuration, example: Example) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    archive_path, descriptor_path = build_paths(directory, configuration.identifier)
    np.savez_compressed(
        archive_path,
        inputs=example.inputs.astype(np.float16),
        target=example.probability.astype(np.float16),
        hit_count=example.hit_count.astype(np.uint16),
        occupancy_runs=example.occupancy_runs,
        film=example.film,
    )
    descriptor_path.write_text(json.dumps(build_descriptor(configuration, example), indent=2),
                               encoding="utf-8")
    return archive_path
