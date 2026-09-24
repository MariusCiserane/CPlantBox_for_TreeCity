"""Une scene de reseaux traduite en solides, dans l'ordre ou le sol doit les recevoir.

Les remplissages sont classes par priorite physique et consommes contre un budget decroissant,
ce qui creuse les anneaux sans jamais avoir a decrire un anneau : le lumen reserve son volume,
le sable d'enrobage prend ce qui reste dans la tranchee, le remblai prend le reste.

La fosse est une colonne circulaire, la meme que celle du masque de sol : la ou une tranchee la
traverse, le remblai y est lache et non recompacte.
"""
from dataclasses import dataclass

import numpy as np

from grid import VoxelGrid
from network.primitives import Intersection, PipeRun, PitColumn, TrenchSlab
from network.street import (LOOSE_BACKFILL, PIPE_SURROUND, RECOMPACTED_BACKFILL, StreetScene,
                            UtilityRun, build_lumen)


@dataclass(frozen=True)
class StreetShapes:
    fills: tuple
    lumens: tuple


def build_surround(run: UtilityRun) -> PipeRun:
    return PipeRun(f"{run.name}_surround", run.nodes_cm, run.radius_cm + run.surround_cover_cm)


def build_trench(run: UtilityRun) -> TrenchSlab:
    depth_cm = -min(node[2] for node in run.nodes_cm)
    return TrenchSlab(f"{run.name}_trench",
                      tuple((node[0], node[1]) for node in run.nodes_cm),
                      run.trench_half_width_cm,
                      -(depth_cm + run.radius_cm + run.bedding_depth_cm))


def build_scene_shapes(scene: StreetScene, grid: VoxelGrid) -> StreetShapes:
    pit = PitColumn("pit_column", scene.pit_radius_cm, scene.pit_depth_cm)
    lumens = tuple(build_lumen(run) for run in scene.utilities)
    trenches = tuple(build_trench(run) for run in scene.utilities)

    fills = (tuple((lumen, None) for lumen in lumens)
             + tuple((build_surround(run), PIPE_SURROUND) for run in scene.utilities)
             + tuple((Intersection(f"{trench.name}_under_pit", (trench, pit)), LOOSE_BACKFILL)
                     for trench in trenches)
             + tuple((trench, RECOMPACTED_BACKFILL) for trench in trenches))
    return StreetShapes(fills, lumens)
