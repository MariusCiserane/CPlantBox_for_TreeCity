"""Water escaping a pipe joint, and the plume roots actually follow.

A leak is at a joint, not along the whole pipe: concrete comes in 2.5 m lengths and ductile
iron in 6 m, and spreading the same water over the full run would dilute the very thing that
moves root mass. The plume is not a sphere either. Water leaves the joint into the bedding
sand, which conducts it along the trench and, under gravity, downward; the trench wall holds
it back sideways. So it is an ellipsoid, long along the pipe and short across it.

Reach is a budget, not a free parameter: bias per growth step times reach is roughly constant
at 5.9 in this model, so a wider plume is a weaker one. Sixty centimetres across the trench
puts the pull at about half the gravitropic dive, which is the point of the exercise.
"""
from dataclasses import dataclass

import numpy as np

from grid import VoxelGrid
from obstacles.layouts import ALONG_Y
from obstacles.street import MAJOR_LEAK, NO_LEAK, Utility

PLUME_REACH_ALONG_TRENCH_CM = 150.0
PLUME_REACH_DOWNWARD_CM = 120.0
PLUME_REACH_UPWARD_CM = 50.0
PLUME_REACH_LATERAL_CM = 60.0
# A plume brings the soil back to field capacity; beyond that the water drains away.
MINOR_LEAK_WATER_FRACTION = 0.75
MAJOR_LEAK_WATER_FRACTION = 1.0
MINOR_LEAK_REACH_SCALE = 0.6
MAJOR_LEAK_REACH_SCALE = 1.0


@dataclass(frozen=True)
class JointLeak:
    name: str
    axis: str
    across_cm: float
    along_cm: float
    depth_cm: float
    available_water_fraction: float
    reach_scale: float


def draw_joint_leaks(utility: Utility, rng: np.random.Generator) -> tuple[JointLeak, ...]:
    """The joint nearest the trunk, since a leak beyond the scene teaches the network nothing."""
    if utility.leakage_status == NO_LEAK:
        return ()
    spacing = utility.joint_spacing_cm
    phase = float(rng.uniform(0.0, spacing))
    along_cm = phase - spacing * round(phase / spacing)
    is_major = utility.leakage_status == MAJOR_LEAK
    return (JointLeak(
        name=f"{utility.name}_leak",
        axis=utility.axis,
        across_cm=utility.offset_cm,
        along_cm=along_cm,
        depth_cm=utility.depth_cm,
        available_water_fraction=MAJOR_LEAK_WATER_FRACTION if is_major else MINOR_LEAK_WATER_FRACTION,
        reach_scale=MAJOR_LEAK_REACH_SCALE if is_major else MINOR_LEAK_REACH_SCALE,
    ),)


def build_plume_ramp(leak: JointLeak, grid: VoxelGrid) -> np.ndarray:
    """1 at the joint, fading to 0 on an ellipsoid stretched along the trench and downward."""
    depths, rows, columns = grid.measure_cell_centres_cm()
    if leak.axis == ALONG_Y:
        along, lateral = rows - leak.along_cm, columns - leak.across_cm
    else:
        along, lateral = columns - leak.along_cm, rows - leak.across_cm
    vertical = depths + leak.depth_cm
    upward = np.where(vertical >= 0.0, PLUME_REACH_UPWARD_CM, PLUME_REACH_DOWNWARD_CM)
    distance = np.sqrt(np.square(along / (PLUME_REACH_ALONG_TRENCH_CM * leak.reach_scale))
                       + np.square(lateral / (PLUME_REACH_LATERAL_CM * leak.reach_scale))
                       + np.square(vertical / (upward * leak.reach_scale)))
    return np.clip(1.0 - distance, 0.0, 1.0).astype(np.float32)


def build_scene_leaks(utilities: tuple[Utility, ...], rng: np.random.Generator) -> tuple[JointLeak, ...]:
    return tuple(leak for utility in utilities for leak in draw_joint_leaks(utility, rng))
