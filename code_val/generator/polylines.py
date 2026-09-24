"""Les polylignes d'une plante vivante, au format que lit la croissance secondaire.

plant.write() ne fait que vider ces memes appels dans un fichier : passer par un vtp pour le
relire couterait un aller-retour disque par tirage, vingt fois par scene. L'objet rendu ici a
juste ce que build_root_tree consomme, et rien d'autre.

On prend tous les organes et non les seules racines : le lien du collet vit dans l'organe graine,
et la recurrence de Murray part de la. Le filtre racines se fait a la voxelisation.
"""
from typing import NamedTuple

import numpy as np

ALL_ORGANS = -1
ROOT_ORGAN_TYPE = 2
CELL_PARAMETERS = ("organType", "id", "creationTime", "age", "subType", "order", "radius")


class LivePolyData(NamedTuple):
    points: np.ndarray
    offsets: np.ndarray
    cell_arrays: dict
    point_arrays: dict


def extract_polydata(plant) -> LivePolyData:
    polylines = plant.getPolylines(ALL_ORGANS)
    points = np.array([[node.x, node.y, node.z] for line in polylines for node in line])
    offsets = np.cumsum([len(line) for line in polylines]).astype(np.intp)
    times = np.array([time for line in plant.getPolylineCTs(ALL_ORGANS) for time in line])
    cell_arrays = {name: np.array(plant.getParameter(name, ALL_ORGANS))
                   for name in CELL_PARAMETERS}
    return LivePolyData(points, offsets, cell_arrays, {"time": times})
