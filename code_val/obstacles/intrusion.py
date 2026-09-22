"""Check afterwards that no root ended up inside a pipe.

CPlantBox confines roots by rejection sampling and gives up after a fixed number of tries,
so the wall is a strong preference rather than a guarantee. This measures the result instead
of trusting it, against the physical radius rather than the inflated one the geometry used.

The penetration depth is reported rather than a boolean: a two tenths of a millimetre graze
and a three centimetre push are different bugs, and only the second one means the exclusion
margin is too small for the growth step in the root parameter file.
"""
from xml.etree import ElementTree

import numpy as np

from obstacles.layouts import ALONG_Y
from obstacles.shapes import Cylinder, StreetShapes


def measure_largest_growth_step_cm(xml_path) -> float:
    root = ElementTree.parse(str(xml_path)).getroot()
    return max(float(node.get("value")) for node in root.iter() if node.get("name") == "dx")


def measure_clearance_to_cylinder_cm(segments: np.ndarray, cylinder: Cylinder) -> np.ndarray:
    """Exact: in the plane cutting the pipe the axis is a point, so this is point to segment."""
    across = 0 if cylinder.axis == ALONG_Y else 1
    start = np.stack([segments[:, across] - cylinder.offset_cm,
                      segments[:, 2] + cylinder.depth_cm], axis=1)
    end = np.stack([segments[:, across + 3] - cylinder.offset_cm,
                    segments[:, 5] + cylinder.depth_cm], axis=1)
    edge = end - start
    length_squared = np.maximum(np.square(edge).sum(axis=1), np.finfo(float).tiny)
    position = np.clip(-(start * edge).sum(axis=1) / length_squared, 0.0, 1.0)
    closest = start + position[:, None] * edge
    return np.hypot(closest[:, 0], closest[:, 1]) - cylinder.radius_cm


def find_intrusions(segments: np.ndarray, shapes: StreetShapes) -> dict[str, float]:
    if len(segments) == 0:
        return {}
    depths = {lumen.name: -measure_clearance_to_cylinder_cm(segments, lumen).min()
              for lumen in shapes.lumens}
    return {name: float(depth) for name, depth in depths.items() if depth > 0.0}
