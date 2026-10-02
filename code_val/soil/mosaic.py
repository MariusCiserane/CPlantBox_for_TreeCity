"""Plusieurs archetypes dans un seul domaine, melanges par patches horizontales.

Un remblai urbain n'est pas un profil unique recopie sur toute la scene : c'est un assemblage de
materiaux deverses cote a cote, chacun avec ses horizons et sa propre profondeur de transition.

Le poids d'une patche en un point est un softmax de la distance a son germe. Le softmax donne
des poids qui somment a 1 partout, sans renormalisation, et sa temperature est directement la
largeur de transition en centimetres. La distance est perturbee par un champ correle, sinon les
frontieres seraient les bissectrices d'un diagramme de Voronoi, bien trop droites pour une
fouille. Un quart des scenes prend une transition courte, qui represente un joint de fouille
vertical : le dataset contient les deux regimes de contact au lieu d'un seul.

Les poids ne sont pas stockes. La graine de frontiere et les trois longueurs les reproduisent.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from grid import VoxelGrid
from soil.archetypes import ARCHETYPES, SoilArchetype
from soil.profile import SoilProfile, draw_soil_profile

# Une patche unique, soit un sol lateralement uniforme, tombe dans une scene sur quatre : le contrat
# l'accepte en entree, le reseau doit donc l'avoir vu.
PATCH_COUNT_RANGE = (1, 4)
PRIMARY_SEED_RADIUS_CM = 200.0
SEED_MINIMUM_SEPARATION_CM = 250.0
SEED_ATTEMPTS = 30
MINIMUM_PATCH_AREA_FRACTION = 0.08
TRANSITION_LENGTH_RANGE_CM = (40.0, 90.0)
SHARP_TRANSITION_LENGTH_CM = 20.0
SHARP_CONTACT_PROBABILITY = 0.25
BOUNDARY_WANDER_CM = 120.0
BOUNDARY_WANDER_LENGTH_CM = 150.0


@dataclass(frozen=True)
class SoilPatch:
    profile: SoilProfile
    centre_row_cm: float
    centre_column_cm: float


@dataclass(frozen=True)
class SoilMosaic:
    patches: tuple[SoilPatch, ...]
    transition_length_cm: float
    boundary_wander_cm: float
    boundary_wander_length_cm: float
    boundary_seed: int


def single_patch_mosaic(profile: SoilProfile) -> SoilMosaic:
    return SoilMosaic((SoilPatch(profile, 0.0, 0.0),), SHARP_TRANSITION_LENGTH_CM,
                      BOUNDARY_WANDER_CM, BOUNDARY_WANDER_LENGTH_CM, 0)


def draw_seed_cm(grid: VoxelGrid, taken: list, rng: np.random.Generator) -> tuple:
    """Assez loin des germes deja poses pour qu'aucune patche ne soit avalee."""
    for _ in range(SEED_ATTEMPTS):
        seed = (float(rng.uniform(-grid.half_length_cm, grid.half_length_cm)),
                float(rng.uniform(-grid.half_width_cm, grid.half_width_cm)))
        if all(np.hypot(seed[0] - row, seed[1] - column) >= SEED_MINIMUM_SEPARATION_CM
               for row, column in taken):
            return seed
    return seed


def draw_transition_length_cm(rng: np.random.Generator) -> float:
    if rng.random() < SHARP_CONTACT_PROBABILITY:
        return SHARP_TRANSITION_LENGTH_CM
    return float(rng.uniform(*TRANSITION_LENGTH_RANGE_CM))


def draw_soil_mosaic(primary: SoilArchetype, grid: VoxelGrid,
                     rng: np.random.Generator) -> SoilMosaic:
    count = int(rng.integers(PATCH_COUNT_RANGE[0], PATCH_COUNT_RANGE[1] + 1))
    others = [archetype for archetype in ARCHETYPES if archetype.identifier != primary.identifier]
    companions = rng.choice(len(others), size=count - 1, replace=False)

    # La patche principale germe pres du tronc : le champ proche porte l'essentiel de la masse
    # racinaire, donc il doit rester l'archetype que la scene revendique.
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.0, PRIMARY_SEED_RADIUS_CM))
    seeds = [(radius * np.sin(angle), radius * np.cos(angle))]
    patches = [SoilPatch(draw_soil_profile(primary, rng), *seeds[0])]

    for index in companions:
        seed = draw_seed_cm(grid, seeds, rng)
        seeds.append(seed)
        patches.append(SoilPatch(draw_soil_profile(others[index], rng), *seed))

    return SoilMosaic(tuple(patches), draw_transition_length_cm(rng), BOUNDARY_WANDER_CM,
                      BOUNDARY_WANDER_LENGTH_CM, int(rng.integers(1, 2 ** 31)))


def build_wander_field(mosaic: SoilMosaic, grid: VoxelGrid) -> np.ndarray:
    rng = np.random.default_rng(mosaic.boundary_seed)
    sigma = mosaic.boundary_wander_length_cm / grid.cell_width_cm
    shape = (len(mosaic.patches), grid.row_cells, grid.column_cells)
    field = gaussian_filter(rng.standard_normal(shape), sigma=(0.0, sigma, sigma))
    return field / field.std(axis=(1, 2), keepdims=True)


def build_patch_weights(mosaic: SoilMosaic, grid: VoxelGrid) -> np.ndarray:
    if len(mosaic.patches) == 1:
        return np.ones((1, grid.row_cells, grid.column_cells))

    rows = (np.arange(grid.row_cells) + 0.5) * grid.cell_width_cm - grid.half_length_cm
    columns = (np.arange(grid.column_cells) + 0.5) * grid.cell_width_cm - grid.half_width_cm
    centres = np.array([(patch.centre_row_cm, patch.centre_column_cm) for patch in mosaic.patches])

    distances = np.hypot(rows[None, :, None] - centres[:, 0, None, None],
                         columns[None, None, :] - centres[:, 1, None, None])
    distances = distances + mosaic.boundary_wander_cm * build_wander_field(mosaic, grid)

    exponents = np.exp(-(distances - distances.min(axis=0)) / mosaic.transition_length_cm)
    return exponents / exponents.sum(axis=0)
