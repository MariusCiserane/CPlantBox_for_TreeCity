"""Ce qu'affirmeraient les masques durs de la phase C, mesure sur la cible avant de les coder.

Un masque interdit un voxel : si la cible y a de la masse, il detruit de l'information reelle.
Les seuils sont des reglages et non des verites. Bengough (0,5 / 2,5 MPa) a ete mesure sur mais,
coton et arachide, pas sur des racines ligneuses.

Le sol bloquant ne l'est pas sur son bord : CPlantBox lit f_se a l'apex avant de l'avancer, donc
une racine finit son dernier pas dans le premier voxel bloque. Mesure sur le dataset v2, 99,9 % de
la masse en sol bloquant est a un voxel d'un voxel ouvert. D'ou une seconde zone, rognee de ce bord.
"""
import numpy as np
from scipy.ndimage import binary_dilation

PIPE_BLOCK_FRACTION = 0.9
GROWTH_BLOCK_SCALE = 1e-3
OVERSHOOT_VOXELS = 1


def build_forbidden_zones(pipe_fraction: np.ndarray, growth_scale: np.ndarray) -> dict:
    blocked = growth_scale < GROWTH_BLOCK_SCALE
    reachable = binary_dilation(~blocked, iterations=OVERSHOOT_VOXELS)
    return {"lumen": pipe_fraction > PIPE_BLOCK_FRACTION,
            "blocking_soil": blocked,
            "blocking_soil_interior": blocked & ~reachable}


def measure_forbidden_mass(hit_count: np.ndarray, zones: dict) -> dict:
    return {name: int(hit_count[zone].sum(dtype=np.int64)) for name, zone in zones.items()}
