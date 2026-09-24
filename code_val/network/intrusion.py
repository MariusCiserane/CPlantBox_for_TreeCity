"""Verifier apres coup qu'aucune racine n'a fini dans un tuyau.

CPlantBox confine par rejet et renonce apres un nombre fixe d'essais : le mur est une preference
forte, pas une garantie. On mesure donc le resultat au lieu de lui faire confiance, contre le
rayon physique et non le rayon gonfle qu'a utilise la geometrie.

La profondeur de penetration est rapportee plutot qu'un booleen : un frolement de deux dixiemes
de millimetre et une poussee de trois centimetres ne sont pas le meme bug, et seul le second dit
que la marge d'exclusion est trop courte pour le pas de croissance du fichier de parametres.

Seuls les noeuds sont testes, pas l'interieur des segments. Un segment fait au plus un dx, soit
3 cm, et le plus fin des lumens fait 9 cm de diametre : une corde de 3 cm dans un cercle de
4,5 cm n'entame que 2,6 mm, hors d'atteinte derriere une marge d'exclusion de 4 cm.
"""
from xml.etree import ElementTree

import numpy as np

from network.primitives import measure_signed_distance


def measure_largest_growth_step_cm(xml_path) -> float:
    root = ElementTree.parse(str(xml_path)).getroot()
    return max(float(node.get("value")) for node in root.iter() if node.get("name") == "dx")


def find_intrusions(segments: np.ndarray, lumens: tuple) -> dict:
    if len(segments) == 0:
        return {}
    nodes = np.vstack([segments[:, :3], segments[:, 3:6]])
    depths = {lumen.name: -float(measure_signed_distance(lumen, nodes).min()) for lumen in lumens}
    return {name: depth for name, depth in depths.items() if depth > 0.0}
