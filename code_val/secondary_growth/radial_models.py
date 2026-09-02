"""Les quatre modeles de rayon secondaire de secondary_root_growth_math.md.

Pipe model, da Vinci et Murray sont la meme recurrence depuis les pointes,
coefficient * (somme des r_fille^exposant)^(1/exposant), a coefficient et
exposant differents. Une seule remontee les couvre tous les trois.
"""
import numpy as np

from secondary_growth.root_tree import NO_PARENT

PIPE_EXPONENT = 2.0
DA_VINCI_EXPONENT = 2.0
NEUTRAL_COEFFICIENT = 1.0


def compute_age_radius(primary_radii: np.ndarray, ages: np.ndarray, is_structural: np.ndarray,
                       growth_rate: float, allometric_exponent: float) -> np.ndarray:
    """Section 3.1 : r_age = r0 + k.t^alpha, les racines fines gardent leur rayon primaire."""
    increment = growth_rate * ages ** allometric_exponent
    return primary_radii + np.where(is_structural, increment, 0.0)


def propagate_from_tips(parent_indices: np.ndarray, levels: list, primary_radii: np.ndarray,
                        exponent: float, coefficient: float) -> np.ndarray:
    """Sections 3.2 a 3.4 : remontee niveau par niveau, des pointes vers le collet.

    Une pointe n'accumule rien et garde son rayon primaire, sans traitement particulier.
    """
    radii = primary_radii.copy()
    accumulated = np.zeros(len(primary_radii))

    for level in reversed(levels):
        radii[level] = np.maximum(primary_radii[level], coefficient * accumulated[level] ** (1.0 / exponent))
        linked = level[parent_indices[level] != NO_PARENT]
        np.add.at(accumulated, parent_indices[linked], radii[linked] ** exponent)

    return radii


def compute_pipe_radius(tree, pipe_efficiency: float) -> np.ndarray:
    return propagate_from_tips(tree.parent_indices, tree.levels, tree.primary_radii,
                               PIPE_EXPONENT, np.sqrt(pipe_efficiency))


def compute_da_vinci_radius(tree, area_conservation: float) -> np.ndarray:
    return propagate_from_tips(tree.parent_indices, tree.levels, tree.primary_radii,
                               DA_VINCI_EXPONENT, area_conservation)


def compute_murray_radius(tree, murray_exponent: float) -> np.ndarray:
    return propagate_from_tips(tree.parent_indices, tree.levels, tree.primary_radii,
                               murray_exponent, NEUTRAL_COEFFICIENT)
