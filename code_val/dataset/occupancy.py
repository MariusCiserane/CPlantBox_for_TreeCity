"""Turn root segments into a binary occupancy grid, then into a probability grid.

Only the centreline of each root is rasterized. Root radii from the XML are one
to two orders of magnitude too small (CPlantBox has no secondary thickening), so
dilating by radius would propagate a known error into the training target.
"""
from typing import NamedTuple

import numpy as np

from grid import VoxelGrid

STRUCTURAL_SUBTYPES = frozenset({1, 2, 4, 6, 7})
SAMPLE_STEP_CM = 1.0
MINIMUM_SAMPLES_PER_SEGMENT = 2


class Occupancy(NamedTuple):
    grid: np.ndarray
    outside_samples: int


class OccupancyEnsemble(NamedTuple):
    probability: np.ndarray
    hit_count: np.ndarray
    run_count: int


def select_structural_segments(segments: np.ndarray, subtypes: np.ndarray) -> np.ndarray:
    keep = np.isin(subtypes, list(STRUCTURAL_SUBTYPES))
    return segments[keep]


def sample_points_along(segments: np.ndarray, step_cm: float) -> np.ndarray:
    """Evenly spaced points along every segment, returned as one (M, 3) array."""
    starts, ends = segments[:, :3], segments[:, 3:6]
    lengths = np.linalg.norm(ends - starts, axis=1)
    per_segment = np.maximum(MINIMUM_SAMPLES_PER_SEGMENT, np.ceil(lengths / step_cm).astype(int) + 1)

    segment_index = np.repeat(np.arange(len(segments)), per_segment)
    first_sample = np.repeat(np.cumsum(per_segment) - per_segment, per_segment)
    position = np.arange(per_segment.sum()) - first_sample
    fraction = position / np.repeat(per_segment - 1, per_segment)

    return starts[segment_index] + fraction[:, None] * (ends - starts)[segment_index]


def voxelize_structural_roots(
    segments: np.ndarray,
    subtypes: np.ndarray,
    grid: VoxelGrid,
    step_cm: float = SAMPLE_STEP_CM,
) -> Occupancy:
    occupancy = np.zeros(grid.shape, dtype=bool)
    structural = select_structural_segments(segments, subtypes)
    if len(structural) == 0:
        return Occupancy(occupancy, 0)

    points = sample_points_along(structural, step_cm)
    depth_index, row_index, column_index, inside = grid.locate_points(points)
    occupancy[depth_index[inside], row_index[inside], column_index[inside]] = True
    return Occupancy(occupancy, int((~inside).sum()))


def compute_occupancy_probability(occupancies: list[np.ndarray]) -> OccupancyEnsemble:
    hit_count = np.sum(occupancies, axis=0, dtype=np.uint16)
    run_count = len(occupancies)
    return OccupancyEnsemble(hit_count.astype(np.float32) / run_count, hit_count, run_count)
