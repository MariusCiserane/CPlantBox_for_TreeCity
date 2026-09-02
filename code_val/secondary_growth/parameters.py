"""Parametres de la croissance secondaire, un champ par constante du modele.

Les valeurs par defaut sont le prior Picea abies de secondary_root_growth_math.md
section 5, a recalibrer sur les donnees GPR/excavation du projet TreeCity.
"""
from dataclasses import dataclass

from dataset.occupancy import STRUCTURAL_SUBTYPES

PICEA_SECONDARY_GROWTH_RATE = 0.022
PICEA_ALLOMETRIC_EXPONENT = 0.5
PICEA_PIPE_EFFICIENCY = 0.8
PICEA_AREA_CONSERVATION = 0.80
PICEA_MURRAY_EXPONENT = 2.1
MAXIMUM_RADIUS_CM = 15.0


@dataclass(frozen=True)
class SecondaryGrowthParameters:
    secondary_growth_rate: float = PICEA_SECONDARY_GROWTH_RATE      # k, cm.day^-alpha
    allometric_exponent: float = PICEA_ALLOMETRIC_EXPONENT          # alpha
    pipe_efficiency: float = PICEA_PIPE_EFFICIENCY                  # kappa
    area_conservation: float = PICEA_AREA_CONSERVATION              # SGC
    murray_exponent: float = PICEA_MURRAY_EXPONENT                  # Delta
    maximum_radius_cm: float = MAXIMUM_RADIUS_CM
    structural_subtypes: frozenset = STRUCTURAL_SUBTYPES
