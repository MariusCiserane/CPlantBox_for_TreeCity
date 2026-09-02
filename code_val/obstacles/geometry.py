"""Compose the CPlantBox growth domain: the soil box minus every solid obstacle.

Pipes and the pavement slab are hard: roots cannot enter them. The soil box itself
confines the root system to the voxel grid extent.
"""
import plantbox as pb

from grid import VoxelGrid
from obstacles.layouts import ALONG_Y, SIDEWALK, ObstacleLayout, Pavement, Pipe

QUARTER_TURN_DEGREES = 90.0
ROTATION_AXIS_X = 0
ROTATION_AXIS_Y = 1
ROUND_CROSS_SECTION = 0.0


def build_pipe_solid(pipe: Pipe, grid: VoxelGrid) -> pb.SignedDistanceFunction:
    """A vertical cylinder tipped a quarter turn so it runs horizontally across the domain."""
    if pipe.axis == ALONG_Y:
        length_cm = 2.0 * grid.half_length_cm
        origin = pb.Vector3d(pipe.offset_cm, -grid.half_length_cm, -pipe.depth_cm)
        rotation_axis = ROTATION_AXIS_X
    else:
        length_cm = 2.0 * grid.half_width_cm
        origin = pb.Vector3d(grid.half_width_cm, pipe.offset_cm, -pipe.depth_cm)
        rotation_axis = ROTATION_AXIS_Y
    cylinder = pb.SDF_PlantContainer(pipe.radius_cm, pipe.radius_cm, length_cm, ROUND_CROSS_SECTION)
    return pb.SDF_RotateTranslate(cylinder, QUARTER_TURN_DEGREES, rotation_axis, origin)


def build_pavement_solid(pavement: Pavement, grid: VoxelGrid) -> pb.SignedDistanceFunction:
    top = pb.Vector3d(grid.half_width_cm, grid.half_length_cm, 0.0)
    if pavement.style == SIDEWALK:
        start = pb.Vector3d(pavement.boundary_cm, -grid.half_length_cm, -pavement.slab_depth_cm)
        return pb.SDF_Cuboid(start, top)

    whole_surface = pb.SDF_Cuboid(
        pb.Vector3d(-grid.half_width_cm, -grid.half_length_cm, -pavement.slab_depth_cm), top)
    tree_pit = pb.SDF_Cuboid(
        pb.Vector3d(-pavement.boundary_cm, -pavement.boundary_cm, -pavement.slab_depth_cm),
        pb.Vector3d(pavement.boundary_cm, pavement.boundary_cm, 0.0))
    return pb.SDF_Difference(whole_surface, tree_pit)


def build_growth_geometry(layout: ObstacleLayout, grid: VoxelGrid) -> pb.SignedDistanceFunction:
    soil_box = pb.SDF_PlantBox(2.0 * grid.half_width_cm, 2.0 * grid.half_length_cm, grid.depth_cm)
    solids = [build_pipe_solid(pipe, grid) for pipe in layout.pipes]
    if layout.pavement is not None:
        solids.append(build_pavement_solid(layout.pavement, grid))
    if not solids:
        return soil_box
    return pb.SDF_Difference([soil_box] + solids)
