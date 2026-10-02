"""Le prior de la phase B, et le Dice doux qui dit s'il vaut la peine.

prior_v2 = prior geometrique x f_se x (1 - lumen) x (1 + B x panache) : le nuage moyen de Gale et
Grigal, eteint la ou le sol arrete la croissance et dans les conduites, renforce dans les fuites.
f_se sert ici et aux masques, jamais comme canal d'entree.

Le Dice est celui de la loss d'entrainement : par scene (2 sum(pt) + 1) / (sum(p) + sum(t) + 1),
puis moyenne. On en garde les trois sommes par scene, ce qui rend la calibration k exacte apres
coup sans relire le dataset, tant que k x p ne depasse 1 nulle part.
"""
import numpy as np

# B du document, qu'il ne fixe pas ; a arreter d'apres measure_hybrid_readiness.py.
LEAK_PRIOR_BOOST = 1.0
LEAK_PRIOR_BOOSTS = (0.0, 0.5, 1.0, 2.0)
DICE_SMOOTHING = 1.0


def build_prior_v2(prior: np.ndarray, growth_scale: np.ndarray, pipe_fraction: np.ndarray,
                   leak_plume: np.ndarray, boost: float) -> np.ndarray:
    return (prior * growth_scale * (1.0 - pipe_fraction) * (1.0 + boost * leak_plume)).astype(np.float32)


def measure_dice_terms(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([np.sum(prediction * target, dtype=np.float64),
                     np.sum(prediction, dtype=np.float64),
                     np.sum(target, dtype=np.float64)])


def calibrate_prior(terms: np.ndarray) -> float:
    """k = taux de base / moyenne du prior ; toutes les scenes ont le meme nombre de voxels."""
    return float(terms[:, 2].sum() / terms[:, 1].sum())


def compute_soft_dice(terms: np.ndarray, scale: float) -> float:
    overlap, predicted, observed = terms.T
    return float(np.mean((2.0 * scale * overlap + DICE_SMOOTHING)
                         / (scale * predicted + observed + DICE_SMOOTHING)))
