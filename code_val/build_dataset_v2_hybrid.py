"""Derive du dataset v2 sur disque les entrees du modele hybride, sans rien resimuler.

Les 2000 scenes de dataset_v2/ ont pousse avec l'ancien modele d'eau (theta, psi) : les champs
sont relus des fichiers, jamais recalcules avec le code courant, qui ferait correspondre aux cibles
un autre sol que celui ou elles ont pousse. bengough et le facteur hydrique viennent du .vti, seule
copie float32 de ce que CPlantBox a lu ; tout le reste est repris tel quel du .npz.

Chaque archive produite est autonome : 17 canaux, prior_v2, et toutes les cibles d'origine.
Reprenable : une scene dont le .json existe deja est ignoree.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from grid import VoxelGrid
from hybrid.channels import (ADDED_CHANNEL_NAMES, build_added_channels, read_channel,
                             select_hybrid_channel_names)
from hybrid.prior import LEAK_PRIOR_BOOST, build_prior_v2
from paraview import read_voxel_grids
from versions import read_commit

SOURCE_DIRECTORY = Path(__file__).resolve().parent / "dataset_v2"
TARGET_DIRECTORY = Path(__file__).resolve().parent / "dataset_v2_hybrid"
CHANNEL_SCHEMA = "v2-hybrid"


def build_superset(archive, descriptor: dict, fields: dict, grid: VoxelGrid) -> tuple:
    names = tuple(descriptor["channel_names"]) + ADDED_CHANNEL_NAMES
    added = build_added_channels(fields["bengough"], fields["eau"], grid)
    return np.concatenate([archive["inputs"].astype(np.float32), added]), names


def write_scene(stem: str, arrays: dict, descriptor: dict) -> None:
    temporary = TARGET_DIRECTORY / f"{stem}.npz.tmp"
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, TARGET_DIRECTORY / f"{stem}.npz")

    temporary = TARGET_DIRECTORY / f"{stem}.json.tmp"
    temporary.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    os.replace(temporary, TARGET_DIRECTORY / f"{stem}.json")


def convert_scene(descriptor_path: Path) -> str:
    stem = descriptor_path.stem
    if (TARGET_DIRECTORY / f"{stem}.json").exists():
        return f"{stem} deja convertie, ignoree"

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    grid = VoxelGrid(**descriptor["grid"])
    fields = read_voxel_grids(SOURCE_DIRECTORY / f"{stem}.vti", grid)
    with np.load(SOURCE_DIRECTORY / f"{stem}.npz") as archive:
        superset, names = build_superset(archive, descriptor, fields, grid)
        kept = select_hybrid_channel_names(names)
        prior_v2 = build_prior_v2(read_channel(superset, names, "prior"), fields["f_se"],
                                  read_channel(superset, names, "pipe_fraction"),
                                  read_channel(superset, names, "leak_plume"), LEAK_PRIOR_BOOST)
        arrays = {key: archive[key] for key in archive.files if key != "inputs"}

    arrays |= {"inputs": superset[[names.index(name) for name in kept]].astype(np.float16),
               "channel_names": np.array(kept), "prior_v2": prior_v2.astype(np.float16)}
    descriptor |= {"channel_schema": CHANNEL_SCHEMA, "channel_names": list(kept),
                   "derived_from": {"dataset": SOURCE_DIRECTORY.name, "commit": read_commit(),
                                    "leak_prior_boost": LEAK_PRIOR_BOOST}}
    write_scene(stem, arrays, descriptor)
    return f"{stem} convertie"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    arguments = parser.parse_args()

    TARGET_DIRECTORY.mkdir(exist_ok=True)
    descriptors = sorted(SOURCE_DIRECTORY.glob("scene_*.json"))
    with Pool(arguments.workers) as pool:
        for message in pool.imap_unordered(convert_scene, descriptors):
            print(message, flush=True)


if __name__ == "__main__":
    main()
