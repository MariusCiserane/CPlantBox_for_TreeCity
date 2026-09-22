"""The absolute barriers: the soil box minus every pipe a root may not enter.

CPlantBox enforces this by rejection sampling, and it gives up after twenty angle steps and
takes the least bad direction, so the wall is not quite absolute on its own. The margin
below covers the real hole, which is elsewhere: Organ::createSegments slides the tip node
forward along its existing heading by up to one dx without testing the geometry at all, so
a tip stopped against a pipe can be pushed into it every time step. The margin therefore
has to exceed the largest dx in the root parameter file, not the sagitta of a growth step.

Only this module inflates a pipe. The fractional masks and the intrusion check both read
the physical radius from shapes.lumens.
"""
import plantbox as pb

from grid import VoxelGrid
from obstacles.geometry import build_pipe_solid
from obstacles.layouts import Pipe
from obstacles.shapes import StreetShapes

PIPE_EXCLUSION_MARGIN_CM = 4.0


def build_street_geometry(shapes: StreetShapes, grid: VoxelGrid) -> pb.SignedDistanceFunction:
    """A flat difference: nesting a union inside it is the same expression and one dispatch more."""
    soil_box = pb.SDF_PlantBox(2.0 * grid.half_width_cm, 2.0 * grid.half_length_cm, grid.depth_cm)
    pipes = [build_pipe_solid(Pipe(lumen.axis, lumen.offset_cm, lumen.depth_cm,
                                   lumen.radius_cm + PIPE_EXCLUSION_MARGIN_CM), grid)
             for lumen in shapes.lumens]
    return pb.SDF_Difference([soil_box] + pipes)
