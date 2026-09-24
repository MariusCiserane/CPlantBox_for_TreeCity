"""Les trois fichiers d'une scene : le .npz pour l'entrainement, le .json pour savoir ce qu'il y
a dedans, le .vti pour aller regarder.

Le descripteur est l'image de la configuration : tout ce qui a ete tire y figure parce qu'il
figure dans la dataclass, sans liste a tenir a jour separement.

L'ecriture passe par un fichier temporaire puis un renommage, et le .json vient en dernier :
c'est lui qui atteste qu'une scene est complete. La v1 testait l'existence du .npz, donc une
archive tronquee par une interruption passait pour finie.
"""
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from generator.build import Example
from generator.scene import SCHEMA_VERSION, SceneConfiguration
from generator.tensors import CHANNEL_NAMES, CHANNEL_SCHEMA, FILM_NAMES
from generator.voxelise import RADIUS_BIN_EDGES_CM, SAMPLE_STEP_CM
from grid import VoxelGrid
from paraview import write_voxel_grids
from dataset.occupancy import STRUCTURAL_SUBTYPES
from tree.growth import XML_PATH

SCENE_NAME_TEMPLATE = "scene_{identifier:05d}"


def build_paths(directory: Path, identifier: int) -> tuple:
    stem = SCENE_NAME_TEMPLATE.format(identifier=identifier)
    return (directory / f"{stem}.npz", directory / f"{stem}.json", directory / f"{stem}.vti")


def is_complete(directory: Path, identifier: int) -> bool:
    return build_paths(directory, identifier)[1].exists()


def read_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return "inconnu"


def build_descriptor(configuration: SceneConfiguration, example: Example, grid: VoxelGrid,
                     parameters) -> dict:
    secondary = asdict(parameters)
    secondary["structural_subtypes"] = sorted(secondary["structural_subtypes"])
    return {
        "schema_version": SCHEMA_VERSION,
        "channel_schema": CHANNEL_SCHEMA,
        "channel_names": list(CHANNEL_NAMES),
        "film_names": list(FILM_NAMES),
        "grid": asdict(grid),
        "configuration": asdict(configuration),
        "secondary_growth": secondary,
        "voxelisation": {"monte_carlo_runs": len(configuration.root_seeds),
                         "structural_subtypes": sorted(STRUCTURAL_SUBTYPES),
                         "sample_step_cm": SAMPLE_STEP_CM,
                         "radius_bin_edges_cm": list(RADIUS_BIN_EDGES_CM),
                         "packed_runs_shape": [len(configuration.root_seeds), *grid.shape]},
        "versions": {"commit": read_commit(), "xml": XML_PATH.name},
        "diagnostics": example.diagnostics,
    }


def write_scene(directory: Path, configuration: SceneConfiguration, example: Example, grid: VoxelGrid, parameters) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    archive_path, descriptor_path, voxel_path = build_paths(directory, configuration.identifier)

    # numpy ajoute .npz au nom qu'on lui passe, mais pas a un fichier deja ouvert.
    temporary = archive_path.with_suffix(".npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            inputs=example.inputs.astype(np.float16),
            film=example.film,
            target=example.target_radius.astype(np.float16),
            target_centreline=example.target_centreline.astype(np.float16),
            root_volume=example.root_volume.astype(np.float16),
            hit_count=example.hit_count_radius,
            hit_count_centreline=example.hit_count_centreline,
            occupancy_runs=example.radius_runs,
            centreline_runs=example.centreline_runs,
            **{f"polyline_{name}": values for name, values in example.polylines.items()})
    os.replace(temporary, archive_path)

    write_voxel_grids(voxel_path, example.fields, grid)

    descriptor = build_descriptor(configuration, example, grid, parameters)
    temporary = descriptor_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    os.replace(temporary, descriptor_path)
    return archive_path
