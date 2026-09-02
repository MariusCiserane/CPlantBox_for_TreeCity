"""Fusion des modeles et contraintes de continuite, section 4 du document.

La fusion est litterale : r_final = max(r_age, r_pipe) sur les racines structurales,
r0 sur les racines fines. da Vinci et Murray sont calcules et ecrits dans le vtp
mais n'entrent pas dans la fusion, ils servent a calibrer SGC et Delta.
"""
import numpy as np


def fuse_radii(primary_radii: np.ndarray, age_radii: np.ndarray, pipe_radii: np.ndarray,
               is_structural: np.ndarray) -> np.ndarray:
    return np.where(is_structural, np.maximum(age_radii, pipe_radii), primary_radii)


def enforce_continuity(radii: np.ndarray, parent_indices: np.ndarray, levels: list,
                       primary_radii: np.ndarray, maximum_radius_cm: float) -> np.ndarray:
    """Aucune racine plus fine que son rayon primaire, ni plus epaisse que son porteur.

    Les deux contraintes se contredisent quand CPlantBox tire une laterale plus epaisse
    que le segment qui la porte. La continuite l'emporte alors sur le plancher primaire.
    """
    constrained = np.clip(radii, primary_radii, maximum_radius_cm)

    # Descente du collet vers les pointes : le parent est deja contraint quand on traite ses fils.
    for level in levels[1:]:
        constrained[level] = np.minimum(constrained[level], constrained[parent_indices[level]])

    return constrained


def measure_taper_ratio(radii: np.ndarray, levels: list) -> float:
    """Validation interne section 4 : r(pointe) / r(base) attendu sous 0.5 sur un arbre age."""
    return float(radii[levels[-1]].mean() / radii[levels[0]].mean())
