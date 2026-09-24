"""Les solides d'une scene de reseaux, et les distances qui les definissent.

Le mur CPlantBox, la penalite de sol et le controle d'intrusion doivent s'accorder sur ou se
trouve un tuyau. La scene est donc decrite une seule fois, ici, et chaque consommateur lit la
distance dont il a besoin. Aucune forme n'est liee a un axe : un tuyau est un segment entre deux
points, une conduite coudee une polyligne.

Deux familles, parce qu'elles ne se mesurent pas pareil :

- les tuyaux sont des capsules a bouts plats en trois dimensions. Leur distance signee reprend
  la formule de SDF_PlantContainer::getDist exprimee en coordonnees monde, ce qui permet de
  confronter numpy et CPlantBox au centimetre pres et pas seulement en signe.
- les tranchees et la fosse sont des prismes : une forme de plan extrudee de la surface vers le
  bas. On les decrit par leur distance en plan et leur profondeur, ce qui rend leur couverture
  separable donc exacte en profondeur.
"""
from dataclasses import dataclass

import numpy as np

SURFACE_CM = 0.0
ELBOW_OVERLAP_RADII = 1.0  # un coude jusqu'a 90 degres se referme si chaque segment deborde d'un rayon


@dataclass(frozen=True)
class PipeSegment:
    name: str
    start_cm: tuple[float, float, float]
    end_cm: tuple[float, float, float]
    radius_cm: float


@dataclass(frozen=True)
class PipeRun:
    name: str
    nodes_cm: tuple[tuple[float, float, float], ...]
    radius_cm: float


@dataclass(frozen=True)
class TrenchSlab:
    """Excavee depuis la surface, donc une saignee et non un fourreau."""
    name: str
    path_cm: tuple[tuple[float, float], ...]
    half_width_cm: float
    bottom_cm: float


@dataclass(frozen=True)
class PitColumn:
    name: str
    radius_cm: float
    depth_cm: float


@dataclass(frozen=True)
class Intersection:
    name: str
    parts: tuple


def build_run_segments(run: PipeRun) -> tuple:
    nodes = np.asarray(run.nodes_cm, float)
    directions = nodes[1:] - nodes[:-1]
    units = directions / np.linalg.norm(directions, axis=1)[:, None]
    overlap = ELBOW_OVERLAP_RADII * run.radius_cm

    segments = []
    for index, unit in enumerate(units):
        start = nodes[index] - (overlap * unit if index > 0 else 0.0)
        end = nodes[index + 1] + (overlap * unit if index + 1 < len(units) else 0.0)
        segments.append(PipeSegment(f"{run.name}_{index}", tuple(start), tuple(end), run.radius_cm))
    return tuple(segments)


def measure_segment_distance(points: np.ndarray, segment: PipeSegment) -> np.ndarray:
    start = np.asarray(segment.start_cm, float)
    direction = np.asarray(segment.end_cm, float) - start
    length = float(np.linalg.norm(direction))
    unit = direction / length

    offsets = points - start
    along = offsets @ unit
    radial = np.linalg.norm(offsets - along[:, None] * unit, axis=1)
    return np.maximum(radial - segment.radius_cm, np.abs(along - 0.5 * length) - 0.5 * length)


def measure_signed_distance(shape, points: np.ndarray) -> np.ndarray:
    """Tuyaux seulement : les prismes se mesurent en plan et en profondeur."""
    if isinstance(shape, PipeSegment):
        return measure_segment_distance(points, shape)
    return np.min([measure_segment_distance(points, segment)
                   for segment in build_run_segments(shape)], axis=0)


def measure_pipe_bounds_cm(shape) -> tuple:
    nodes = np.asarray([shape.start_cm, shape.end_cm] if isinstance(shape, PipeSegment)
                       else shape.nodes_cm, float)
    return nodes.min(axis=0) - shape.radius_cm, nodes.max(axis=0) + shape.radius_cm


def measure_path_distance(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    distances = None
    for start, end in zip(path[:-1], path[1:]):
        direction = end - start
        offsets = points - start
        along = np.clip((offsets @ direction) / (direction @ direction), 0.0, 1.0)
        distance = np.linalg.norm(offsets - along[:, None] * direction, axis=1)
        distances = distance if distances is None else np.minimum(distances, distance)
    return distances


def measure_plan_distance(shape, points: np.ndarray) -> np.ndarray:
    """Prismes seulement, sur des points (x, y)."""
    if isinstance(shape, TrenchSlab):
        return measure_path_distance(points, np.asarray(shape.path_cm, float)) - shape.half_width_cm
    if isinstance(shape, PitColumn):
        return np.linalg.norm(points, axis=1) - shape.radius_cm
    return np.max([measure_plan_distance(part, points) for part in shape.parts], axis=0)


def measure_bottom_cm(shape) -> float:
    if isinstance(shape, TrenchSlab):
        return shape.bottom_cm
    if isinstance(shape, PitColumn):
        return -shape.depth_cm
    return max(measure_bottom_cm(part) for part in shape.parts)
