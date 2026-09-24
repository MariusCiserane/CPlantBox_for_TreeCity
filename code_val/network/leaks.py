"""L'eau qui s'echappe d'un joint, et le panache que les racines suivent vraiment.

Une fuite est a un joint, pas le long du tuyau : le beton vient en elements de 2,5 m et la fonte
en 6 m, et etaler la meme eau sur toute la conduite diluerait precisement ce qui deplace la
masse racinaire. Le panache n'est pas une sphere non plus. L'eau part du joint dans le sable de
pose, qui la conduit le long de la tranchee et, par gravite, vers le bas ; la paroi de la
tranchee la retient sur les cotes. C'est donc un ellipsoide, long le long du tuyau et court en
travers, et il tourne avec la tangente locale de la conduite.

La portee est un budget, pas un parametre libre : dans ce modele le biais par pas de croissance
multiplie par la portee vaut environ 5,9, donc un panache plus large est un panache plus faible.
Soixante centimetres en travers mettent l'attraction a la moitie du plongeon gravitropique, ce
qui est tout l'interet de l'exercice.
"""
from dataclasses import dataclass

import numpy as np

from grid import VoxelGrid
from network.street import MAJOR_LEAK, NO_LEAK, UtilityRun

PLUME_REACH_ALONG_TRENCH_CM = 150.0
PLUME_REACH_DOWNWARD_CM = 120.0
PLUME_REACH_UPWARD_CM = 50.0
PLUME_REACH_LATERAL_CM = 60.0
# A plume brings the soil back to field capacity; beyond that the water drains away.
MINOR_LEAK_WATER_FRACTION = 0.75
MAJOR_LEAK_WATER_FRACTION = 1.0
MINOR_LEAK_REACH_SCALE = 0.6
MAJOR_LEAK_REACH_SCALE = 1.0


@dataclass(frozen=True)
class JointLeak:
    name: str
    origin_cm: tuple[float, float, float]
    along_unit: tuple[float, float]
    available_water_fraction: float
    reach_scale: float


def find_nearest_joint(run: UtilityRun, phase_cm: float) -> tuple:
    """Le joint le plus proche du tronc : une fuite hors de la scene n'apprend rien au reseau."""
    nodes = np.asarray(run.nodes_cm, float)
    lengths = np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(lengths)])

    steps = np.arange(phase_cm, travelled[-1], run.joint_spacing_cm)
    distances = [np.linalg.norm(interpolate(nodes, travelled, step)[:2]) for step in steps]
    return interpolate(nodes, travelled, steps[int(np.argmin(distances))]), nodes


def interpolate(nodes: np.ndarray, travelled: np.ndarray, distance_cm: float) -> np.ndarray:
    index = int(np.clip(np.searchsorted(travelled, distance_cm) - 1, 0, len(nodes) - 2))
    span = travelled[index + 1] - travelled[index]
    return nodes[index] + (nodes[index + 1] - nodes[index]) * ((distance_cm - travelled[index]) / span)


def draw_joint_leak(run: UtilityRun, rng: np.random.Generator) -> tuple:
    if run.leakage_status == NO_LEAK:
        return ()
    origin, nodes = find_nearest_joint(run, float(rng.uniform(0.0, run.joint_spacing_cm)))
    tangent = nodes[1][:2] - nodes[0][:2]
    is_major = run.leakage_status == MAJOR_LEAK
    return (JointLeak(
        name=f"{run.name}_leak",
        origin_cm=tuple(float(value) for value in origin),
        along_unit=tuple(float(value) for value in tangent / np.linalg.norm(tangent)),
        available_water_fraction=MAJOR_LEAK_WATER_FRACTION if is_major else MINOR_LEAK_WATER_FRACTION,
        reach_scale=MAJOR_LEAK_REACH_SCALE if is_major else MINOR_LEAK_REACH_SCALE,
    ),)


def build_scene_leaks(utilities: tuple, rng: np.random.Generator) -> tuple:
    return tuple(leak for run in utilities for leak in draw_joint_leak(run, rng))


def build_plume_ramp(leak: JointLeak, grid: VoxelGrid) -> np.ndarray:
    """1 au joint, s'eteint sur un ellipsoide etire le long de la tranchee et vers le bas."""
    depths, rows, columns = grid.measure_cell_centres_cm()
    offsets = np.stack([columns - leak.origin_cm[0], rows - leak.origin_cm[1]], axis=-1)
    unit = np.asarray(leak.along_unit)

    along = offsets @ unit
    lateral = offsets @ np.array([-unit[1], unit[0]])
    vertical = depths - leak.origin_cm[2]
    upward = np.where(vertical >= 0.0, PLUME_REACH_UPWARD_CM, PLUME_REACH_DOWNWARD_CM)

    distance = np.sqrt(np.square(along / (PLUME_REACH_ALONG_TRENCH_CM * leak.reach_scale))
                       + np.square(lateral / (PLUME_REACH_LATERAL_CM * leak.reach_scale))
                       + np.square(vertical / (upward * leak.reach_scale)))
    return np.clip(1.0 - distance, 0.0, 1.0).astype(np.float32)
