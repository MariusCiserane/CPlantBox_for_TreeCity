"""Grow one root system in CPlantBox and return its segments.

The soil acts through f_se, the elongation scale CPlantBox reads at every apex.
It is pushed into a native EquidistantGrid3D so the hot loop stays in C++;
a Python SoilLookUp subclass would be called once per apex per time step.
"""
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import plantbox as pb

from grid import VoxelGrid
from tree.attributes import DAYS_PER_YEAR

TIME_STEP_DAYS = 10.0
XML_PATH = Path(__file__).resolve().parent.parent / "parameters" / "Picea_Abies_combined_v6.xml"


class RootSystem(NamedTuple):
    segments: np.ndarray
    subtypes: np.ndarray


def build_elongation_grid(growth_scale: np.ndarray, grid: VoxelGrid) -> pb.EquidistantGrid3D:
    """CPlantBox maps a coordinate onto n-1 intervals, so the lookup needs one extra node
    per axis to land on exactly the same cells as the voxel grid. Its depth index runs
    upward, hence the flip."""
    lookup = pb.EquidistantGrid3D(
        -grid.half_width_cm, grid.half_width_cm, grid.column_cells + 1,
        -grid.half_length_cm, grid.half_length_cm, grid.row_cells + 1,
        -grid.depth_cm, 0.0, grid.depth_cells + 1)

    padded = np.empty((grid.depth_cells + 1, grid.row_cells + 1, grid.column_cells + 1))
    padded[:-1, :-1, :-1] = growth_scale[::-1, :, :]
    padded[-1, :, :] = padded[-2, :, :]
    padded[:, -1, :] = padded[:, -2, :]
    padded[:, :, -1] = padded[:, :, -2]
    lookup.data = padded.ravel().tolist()
    return lookup


def assign_elongation_grid(plant: pb.Plant, lookup: pb.EquidistantGrid3D) -> None:
    for parameter in plant.getOrganRandomParameter(pb.OrganTypes.root):
        parameter.f_se = lookup


def extract_root_system(plant: pb.Plant) -> RootSystem:
    edges = np.asarray([[int(s.x), int(s.y)] for s in plant.getSegments()], dtype=int)
    if edges.size == 0:
        return RootSystem(np.zeros((0, 6)), np.zeros(0, dtype=int))
    nodes = np.asarray([[n.x, n.y, n.z] for n in plant.getNodes()], dtype=float)
    segments = np.hstack([nodes[edges[:, 0]], nodes[edges[:, 1]]])
    subtypes = np.rint(pb.SegmentAnalyser(plant).getParameter("subType")).astype(int)
    return RootSystem(segments, subtypes)


def grow_root_system(geometry, elongation_grid, seed: int, age_years: float,
                     vtp_path: Optional[Path] = None,
                     time_step_days: float = TIME_STEP_DAYS) -> RootSystem:
    plant = pb.Plant()
    plant.readParameters(str(XML_PATH))
    plant.setSeed(seed)
    plant.setGeometry(geometry)
    assign_elongation_grid(plant, elongation_grid)
    plant.initialize(False)

    for _ in range(round(age_years * DAYS_PER_YEAR / time_step_days)):
        plant.simulate(time_step_days, False)

    if vtp_path is not None:
        vtp_path.parent.mkdir(parents=True, exist_ok=True)
        plant.write(str(vtp_path))
    return extract_root_system(plant)
