# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
r"""Six welding robots weld a car body along Cartesian seams, coordinated by interlocks.

A simplified car body (body-in-white, built from cuboids at about 1:3 scale) sits on a
skid. Six KUKA KR10 R1100-2 robots with MIG welding torches stand around it: two on each
side, one in front of and one behind the car. Together they weld 28 seams: straight
seams on sills, pillars, roof, aprons, and the front and rear panels, and arc-shaped
seams around the wheel arches. Neighboring robots reach into each other's space at the
roof, the pillars, and the car's corners.

The demo shows three things cuRobo does for a multi-robot welding cell:

- **Cartesian seam constraints.** The torch tip follows each seam at a constant travel
  speed. The torch axis is held at the work angle (along the bisector of the joint) and
  tilted 10 deg into the travel direction (push angle). Rotation about the torch axis is
  left free (``ToolPoseCriteria`` with zero weight on that axis), and cuRobo uses this
  freedom to keep the wrist and torch away from the car body. The torch approaches and
  leaves each seam in a straight line along its axis. The points of a seam are solved
  with warm-started, batched IK, so the joint path is continuous and checked for
  collisions with the car body and the robot itself.
- **Collision-free transfers.** Between seams, cuRobo plans joint-space motions around
  the car body.
- **Interlocks between robots.** Each robot runs its own program. Before a robot starts a
  job (a transfer and a weld), an interlock checks the space the robot will sweep against
  the space the other robots hold: their running motion, or the pose they stand in.
  Space is measured on cuRobo's collision spheres in a voxel grid with a 2 cm margin. A
  robot reserves the space of a whole job before it starts and releases the space of the
  transfer as soon as it reaches the seam. A robot whose job is blocked waits. If robots
  block each other, one of them retreats to its home pose. Afterwards the whole schedule
  is checked again with the collision spheres to get the smallest robot-to-robot
  distance.

By default every robot welds a fixed program, as in a production cell. The left and
right programs are mirrored, so the two rear side robots both start at the roof bow,
whose halves overlap at the car's center line; the interlock lets one of them wait.
Planning the same programs without interlocks shows the collisions the interlocks
prevent. A program step a robot cannot reach is handed to another robot that can. With
``--dynamic`` (or the viewer checkbox) seams are assigned while the cell runs instead:
an idle robot takes the best seam it can reach, prefers seams only it can weld, and
picks another seam when an interlock blocks its first choice.

All robots share one cuRobo robot model (``kr10_r1100_2_welding.yml``). They differ
only in their base pose, so the planner loads the obstacles in the base frame of the
robot it plans for.

Run the demo
------------

From the repository root:

.. code-block:: bash

   python -m curobo.examples.reference.multi_robot_welding

Open http://localhost:8080. The cell is planned (about 20 s after warm-up) and played
back. While a robot waits, the space it needs but another robot holds is shown as red
cubes, and the label above the robot says whom it waits for. The *Interlock log* lists
every wait as it happens; *Show reserved space* draws the space each robot holds as dots
in its color. Click
**Plan without interlocks** to see the robots collide (red marker), and **Plan with
interlocks** to go back. Other options:

.. code-block:: bash

   # Assign seams at run time instead of fixed programs.
   python -m curobo.examples.reference.multi_robot_welding --dynamic
   # Four side robots only; the front and rear seams go to them where they can reach.
   python -m curobo.examples.reference.multi_robot_welding \
       --robots left_front left_rear right_front right_rear
   # Plan without the viewer and print the schedule.
   python -m curobo.examples.reference.multi_robot_welding --headless

Use ``--test`` to plan the cell and check the result.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

#: KUKA KR10 R1100-2 with a MIG welding torch; the tool frame is the wire tip.
ROBOT_FILE = "kr10_r1100_2_welding.yml"
TOOL_FRAME = "torch_tcp"
#: Robot stations of the default cell.
ROBOTS = ("left_front", "left_rear", "right_front", "right_rear", "front", "rear")

#: Time step of all joint trajectories (s). Equal to the planner's interpolation step.
DT = 0.025
#: Seam points solved together in one batched IK call.
TRACK_CHUNK = 16
#: IK solutions of a seam's first pose from which seam tracking is tried.
NUM_START_SOLUTIONS = 6
#: Rounds of start IK (with new random seeds) before a seam counts as unreachable.
PLAN_WELD_ATTEMPTS = 2
#: Largest joint change between neighboring seam points (rad). Larger steps mean the
#: IK jumped to another arm configuration.
MAX_JOINT_STEP = 0.3
#: Fraction of the joint velocity limits used on seams and approach moves.
WELD_JOINT_SPEED_SCALE = 0.5
#: Ticks over which ``check_plan`` measures the tip speed (0.25 s, 1 cm of seam).
SPEED_WINDOW = 10

#: Torch pose criteria: position and torch axis are fixed; rotation about the torch axis
#: (z of the goal frame) is free.
FREE_SPIN = ToolPoseCriteria(
    terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    project_distance_to_goal=True,
)

#: Motion kinds in a robot's timeline.
TRANSFER, WELD, HOME = "transfer", "weld", "home"
#: Phases of a tool path.
APPROACH, SEAM, RETRACT = 0, 1, 2


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def unit(v: Sequence[float]) -> np.ndarray:
    """Normalized copy of a vector."""
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def rot_z(yaw: float) -> np.ndarray:
    """Rotation matrix ``(3, 3)`` about the z-axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix ``(3, 3)`` of a wxyz quaternion."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> List[float]:
    """Quaternion (wxyz) of a rotation matrix ``(3, 3)``."""
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return list(unit(q))


def _box(lower: Sequence[float], upper: Sequence[float]) -> Dict:
    """Axis-aligned cuboid from its lower and upper corner."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return {"dims": list(upper - lower), "pose": list((lower + upper) / 2) + [1.0, 0.0, 0.0, 0.0]}


def _beam(
    start: Sequence[float],
    end: Sequence[float],
    width: float,
    depth: float,
    outward: Sequence[float],
) -> Dict:
    """Cuboid from ``start`` to ``end``; its ``depth`` side faces ``outward``."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = unit(end - start)
    side = unit(np.asarray(outward, float) - np.dot(outward, axis) * axis)
    rotation = np.stack([axis, side, np.cross(axis, side)], axis=1)
    return {
        "dims": [float(np.linalg.norm(end - start)), depth, width],
        "pose": list((start + end) / 2) + matrix_to_quat(rotation),
    }


# --------------------------------------------------------------------------------------
# Work cell and seams
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotStation:
    """Where a robot stands. The robot base sits on a pedestal."""

    #: Robot name, also used as the station name.
    name: str
    #: Base position in the world frame (m).
    position: Tuple[float, float, float]
    #: Rotation of the base about the world z-axis (rad). At 0 the robot faces +x.
    yaw: float

    @property
    def rotation(self) -> np.ndarray:
        """Base orientation ``(3, 3)`` in the world frame."""
        return rot_z(self.yaw)

    def to_base(self, points: np.ndarray) -> np.ndarray:
        """World points ``(..., 3)`` expressed in the base frame."""
        return (np.asarray(points) - np.asarray(self.position)) @ self.rotation

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Base-frame points ``(..., 3)`` expressed in the world frame."""
        return np.asarray(points) @ self.rotation.T + np.asarray(self.position)


@dataclass
class Seam:
    """A weld seam on the car body."""

    #: Unique name.
    name: str
    #: Polyline of the seam in the world frame, shape ``(K, 3)`` with ``K >= 2``.
    points: np.ndarray
    #: Direction from the joint towards the torch side, shape ``(3,)``. For a fillet weld
    #: this is the bisector of the two plates; for a lap weld on a face, the face normal.
    normal: np.ndarray

    @property
    def length(self) -> float:
        """Seam length (m)."""
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def resample(self, step: float) -> Tuple[np.ndarray, np.ndarray]:
        """Points ``(N, 3)`` about ``step`` apart along the seam, and unit tangents ``(N, 3)``."""
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        s_knots = np.concatenate([[0.0], np.cumsum(seg)])
        num = max(2, int(math.ceil(s_knots[-1] / step)) + 1)
        s = np.linspace(0.0, s_knots[-1], num)
        points = np.stack([np.interp(s, s_knots, self.points[:, k]) for k in range(3)], axis=1)
        tangents = np.gradient(points, axis=0)
        return points, tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def _line_seam(
    name: str, start: Sequence[float], end: Sequence[float], normal: Sequence[float]
) -> Seam:
    return Seam(name, np.array([start, end], dtype=np.float64), unit(normal))


def _arc_seam(
    name: str, center: Sequence[float], radius: float, normal: Sequence[float], num: int = 25
) -> Seam:
    """Half circle above ``center`` in the x-z plane, from the front (+x) to the back."""
    angles = np.linspace(0.0, math.pi, num)
    center = np.asarray(center, dtype=np.float64)
    points = center + radius * np.stack([np.cos(angles), np.zeros(num), np.sin(angles)], axis=1)
    return Seam(name, points, unit(normal))


@dataclass(frozen=True)
class WeldCell:
    """Car body, skid, and robot stations. The floor is at ``z = 0``, the car faces +x."""

    #: Robot stations.
    stations: Tuple[RobotStation, ...] = (
        RobotStation("left_front", (0.45, 0.95, 0.30), -math.pi / 2),
        RobotStation("left_rear", (-0.45, 0.95, 0.30), -math.pi / 2),
        RobotStation("right_front", (0.45, -0.95, 0.30), math.pi / 2),
        RobotStation("right_rear", (-0.45, -0.95, 0.30), math.pi / 2),
        RobotStation("front", (1.30, 0.0, 0.30), math.pi),
        RobotStation("rear", (-1.30, 0.0, 0.30), 0.0),
    )
    #: Half width of the car body (m).
    half_width: float = 0.32
    #: Side length of the square robot pedestals (m).
    pedestal_size: float = 0.30

    def station(self, name: str) -> RobotStation:
        """Station of the robot ``name``."""
        return next(s for s in self.stations if s.name == name)

    def car_body(self) -> Dict[str, Dict]:
        """Cuboids of the skid and the car body in the world frame."""
        hw = self.half_width
        parts = {
            "skid": _box([-0.60, -0.18, 0.0], [0.60, 0.18, 0.37]),
            "floor": _box([-0.78, -hw + 0.02, 0.37], [0.78, hw - 0.02, 0.40]),
            "dash": _box([0.40, -hw + 0.01, 0.40], [0.42, hw - 0.01, 0.62]),
            "front_cross": _box([0.76, -hw + 0.01, 0.40], [0.78, hw - 0.01, 0.58]),
            "rear_panel": _box([-0.80, -hw + 0.01, 0.37], [-0.78, hw - 0.01, 0.60]),
            "roof_front": _box([0.16, -hw + 0.03, 0.79], [0.20, hw - 0.03, 0.83]),
            "roof_bow": _box([-0.07, -hw + 0.03, 0.79], [-0.03, hw - 0.03, 0.83]),
            "roof_rear": _box([-0.28, -hw + 0.03, 0.79], [-0.24, hw - 0.03, 0.83]),
        }
        for s, tag in ((1.0, "l"), (-1.0, "r")):

            def y(a: float, b: float) -> Tuple[float, float]:
                return min(s * a, s * b), max(s * a, s * b)

            sill, plate, pillar, rail = (
                y(hw - 0.06, hw),
                y(hw - 0.03, hw - 0.01),
                y(hw - 0.05, hw - 0.01),
                y(hw - 0.07, hw - 0.03),
            )
            parts[f"sill_{tag}"] = _box([-0.40, sill[0], 0.36], [0.42, sill[1], 0.46])
            parts[f"apron_{tag}"] = _box([0.42, plate[0], 0.40], [0.78, plate[1], 0.58])
            parts[f"quarter_{tag}"] = _box([-0.78, plate[0], 0.40], [-0.40, plate[1], 0.60])
            parts[f"b_pillar_{tag}"] = _box([-0.07, pillar[0], 0.46], [-0.03, pillar[1], 0.81])
            parts[f"roof_rail_{tag}"] = _box([-0.28, rail[0], 0.79], [0.20, rail[1], 0.83])
            parts[f"a_pillar_{tag}"] = _beam(
                [0.42, s * (hw - 0.03), 0.58], [0.18, s * (hw - 0.05), 0.81], 0.04, 0.04, [0, s, 0]
            )
            parts[f"c_pillar_{tag}"] = _beam(
                [-0.42, s * (hw - 0.03), 0.60],
                [-0.26, s * (hw - 0.05), 0.81],
                0.05,
                0.04,
                [0, s, 0],
            )
        return parts

    def pedestals(self) -> Dict[str, Dict]:
        """Cuboids of the robot pedestals in the world frame."""
        h = self.pedestal_size / 2
        return {
            f"pedestal_{st.name}": _box(
                [st.position[0] - h, st.position[1] - h, 0.0],
                [st.position[0] + h, st.position[1] + h, st.position[2]],
            )
            for st in self.stations
        }

    def obstacles(self) -> Dict[str, Dict]:
        """All static cuboids in the world frame."""
        return {**self.car_body(), **self.pedestals()}

    def scene_cfg(self, robot: str) -> Dict[str, Dict[str, Dict]]:
        """Collision scene in the base frame of ``robot``."""
        station = self.station(robot)
        rotation = station.rotation
        cuboids = {}
        for name, cuboid in self.obstacles().items():
            pose = cuboid["pose"]
            position = station.to_base(np.array(pose[:3]))
            orientation = rotation.T @ quat_to_matrix(pose[3:])
            cuboids[name] = {
                "dims": cuboid["dims"],
                "pose": list(position) + matrix_to_quat(orientation),
            }
        return {"cuboid": cuboids}

    def programs(self) -> Dict[str, List[str]]:
        """Default robot programs: the seams each robot welds, in order.

        The programs are mirrored between the left and right side and start in shared
        space on purpose: the two rear side robots both begin at the roof bow, whose
        halves overlap at the car's center line, while the front side robots weld the
        B-pillars right next to them and the end robots weld the roof headers. Later the
        side robots meet the end robots at the wheel arches, aprons, and rear quarters.
        """
        front_side = ["b_pillar", "a_pillar", "apron", "front_arch", "sill"]
        rear_side = ["roof_bow", "roof_rail", "c_pillar", "quarter", "rear_arch"]
        return {
            "left_front": [f"{name}_L" for name in front_side],
            "left_rear": [f"{name}_L" for name in rear_side],
            "right_front": [f"{name}_R" for name in front_side],
            "right_rear": [f"{name}_R" for name in rear_side],
            "front": ["front_cross_top", "front_cross_low", "roof_front_L", "roof_front_R"],
            "rear": ["rear_panel_top", "rear_panel_low", "roof_rear_L", "roof_rear_R"],
        }

    def seams(self) -> List[Seam]:
        """The weld seams on the car body."""
        hw = self.half_width
        seams = []
        for s, tag in ((1.0, "L"), (-1.0, "R")):
            face = s * (hw - 0.01)
            seams += [
                _line_seam(f"sill_{tag}", [-0.30, s * hw, 0.46], [0.32, s * hw, 0.46], [0, s, 1]),
                _line_seam(
                    f"a_pillar_{tag}",
                    [0.37, face, 0.63],
                    [0.22, s * (hw - 0.03), 0.77],
                    [0, s, 0.4],
                ),
                _line_seam(
                    f"b_pillar_{tag}", [-0.05, face, 0.50], [-0.05, face, 0.75], [0, s, 0.3]
                ),
                _line_seam(
                    f"c_pillar_{tag}",
                    [-0.40, face, 0.63],
                    [-0.29, s * (hw - 0.03), 0.77],
                    [0, s, 0.3],
                ),
                _line_seam(
                    f"roof_rail_{tag}",
                    [-0.22, s * (hw - 0.03), 0.83],
                    [0.14, s * (hw - 0.03), 0.83],
                    [0, s, 1],
                ),
                _arc_seam(f"front_arch_{tag}", [0.60, face, 0.40], 0.11, [0, s, 0]),
                _arc_seam(f"rear_arch_{tag}", [-0.59, face, 0.40], 0.11, [0, s, 0]),
                # The two halves of the roof bow overlap by 6 cm at the car's center line.
                _line_seam(
                    f"roof_bow_{tag}", [-0.05, s * 0.25, 0.83], [-0.05, -s * 0.03, 0.83], [0, 0, 1]
                ),
                _line_seam(
                    f"roof_rear_{tag}",
                    [-0.28, s * 0.25, 0.83],
                    [-0.28, s * 0.02, 0.83],
                    [-1, 0, 1],
                ),
                _line_seam(
                    f"roof_front_{tag}",
                    [0.20, s * 0.25, 0.83],
                    [0.20, s * 0.02, 0.83],
                    [1, 0, 1],
                ),
                _line_seam(f"apron_{tag}", [0.46, face, 0.58], [0.74, face, 0.58], [0, s, 1]),
                _line_seam(f"quarter_{tag}", [-0.74, face, 0.60], [-0.46, face, 0.60], [0, s, 1]),
            ]
        seams += [
            _line_seam("rear_panel_top", [-0.80, 0.26, 0.60], [-0.80, -0.26, 0.60], [-1, 0, 1]),
            _line_seam("rear_panel_low", [-0.80, 0.24, 0.46], [-0.80, -0.24, 0.46], [-1, 0, 0.7]),
            _line_seam("front_cross_top", [0.78, 0.26, 0.58], [0.78, -0.26, 0.58], [1, 0, 1]),
            _line_seam("front_cross_low", [0.78, 0.24, 0.46], [0.78, -0.24, 0.46], [1, 0, 0.7]),
        ]
        return seams


# --------------------------------------------------------------------------------------
# Cartesian tool paths and timing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WeldProcess:
    """Torch angles, speeds, and path resolution of the welding process."""

    #: Torch tip speed along the seam while welding (m/s).
    travel_speed: float = 0.04
    #: Torch tip speed of the straight approach and retract moves (m/s).
    approach_speed: float = 0.08
    #: Length of the approach and retract moves along the torch axis (m).
    approach_distance: float = 0.06
    #: Torch tilt towards the travel direction (rad).
    push_angle: float = math.radians(10.0)
    #: Torch tip acceleration at the start and end of each straight move (m/s^2).
    acceleration: float = 0.25
    #: Pause at the seam start (arc ignition) and end (crater fill) (s).
    arc_dwell: float = 0.3
    #: Distance between neighboring IK points along the path (m).
    path_step: float = 0.005


def torch_rotation(tangent: np.ndarray, normal: np.ndarray, push_angle: float) -> np.ndarray:
    """Torch orientation ``(3, 3)`` at a seam point.

    The torch z-axis points from the torch into the joint: along ``-normal``, tilted by
    ``push_angle`` towards the travel direction ``tangent``. The x-axis is the travel
    direction projected onto the plane normal to the torch axis. The rotation about the
    torch axis is not constrained while planning, so x is only a reference.
    """
    n = unit(normal - np.dot(normal, tangent) * tangent)
    z = unit(-math.cos(push_angle) * n + math.sin(push_angle) * tangent)
    x = unit(tangent - np.dot(tangent, z) * z)
    return np.stack([x, np.cross(z, x), z], axis=1)


@dataclass
class ToolPath:
    """Torch tip poses for welding one seam: approach, seam, retract."""

    #: Tip positions in the world frame, shape ``(N, 3)``.
    positions: np.ndarray
    #: Torch orientations in the world frame, shape ``(N, 3, 3)``.
    rotations: np.ndarray
    #: Phase of each point: ``APPROACH``, ``SEAM``, or ``RETRACT``. The seam's first and
    #: last points belong to ``SEAM``.
    phase: np.ndarray

    @property
    def arc_length(self) -> np.ndarray:
        """Cumulative tip path length at each point (m), shape ``(N,)``."""
        return np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1))]
        )

    @property
    def seam_range(self) -> Tuple[int, int]:
        """Indices of the first and last seam point."""
        idx = np.flatnonzero(self.phase == SEAM)
        return int(idx[0]), int(idx[-1])


def seam_tool_path(seam: Seam, process: WeldProcess, reverse: bool = False) -> ToolPath:
    """Torch tip path that welds ``seam``, in either direction.

    The torch moves in along its axis to the seam start, follows the seam with the torch
    angles of ``torch_rotation``, and moves back out along its axis at the seam end.
    """
    points, tangents = seam.resample(process.path_step)
    if reverse:
        points, tangents = points[::-1].copy(), -tangents[::-1]
    rotations = np.stack([torch_rotation(t, seam.normal, process.push_angle) for t in tangents])
    num_approach = max(2, int(math.ceil(process.approach_distance / (2 * process.path_step))))
    back = np.linspace(1.0, 0.0, num_approach + 1)[:-1, None] * process.approach_distance
    approach = points[0] - back * rotations[0][:, 2]
    retract = points[-1] - back[::-1] * rotations[-1][:, 2]
    positions = np.concatenate([approach, points, retract])
    all_rotations = np.concatenate(
        [
            np.repeat(rotations[:1], num_approach, axis=0),
            rotations,
            np.repeat(rotations[-1:], num_approach, axis=0),
        ]
    )
    phase = np.concatenate(
        [
            np.full(num_approach, APPROACH),
            np.full(len(points), SEAM),
            np.full(num_approach, RETRACT),
        ]
    )
    return ToolPath(positions, all_rotations, phase)


def trapezoid_times(s: np.ndarray, speed: float, acceleration: float) -> np.ndarray:
    """Times at path positions ``s`` (from 0) for a rest-to-rest trapezoidal speed profile."""
    length = float(s[-1])
    if length <= 0.0:
        return np.zeros_like(s)
    ramp = speed * speed / acceleration
    if length < ramp:  # triangular profile
        peak = math.sqrt(acceleration * length)
        ramp, speed = length, peak
    ramp_s = ramp / 2
    ramp_t = speed / acceleration
    total = 2 * ramp_t + (length - 2 * ramp_s) / speed
    t = np.where(
        s < ramp_s, np.sqrt(2 * np.clip(s, 0, None) / acceleration), ramp_t + (s - ramp_s) / speed
    )
    rest = length - s
    return np.where(rest < ramp_s, total - np.sqrt(2 * np.clip(rest, 0, None) / acceleration), t)


def time_tool_path(
    q: np.ndarray, path: ToolPath, process: WeldProcess, joint_speed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Time a seam's joint path and resample it every ``DT``.

    Approach, seam, and retract are each a rest-to-rest move along the path with a
    trapezoidal tip speed profile (``approach_speed``, ``travel_speed``,
    ``approach_speed``). The torch pauses for ``arc_dwell`` at both ends of the seam.
    Where a joint would exceed ``joint_speed``, the motion slows down.

    Args:
        q: Joint positions at the path points, shape ``(N, dof)``.
        path: The torch tip path.
        process: Speeds and dwell times.
        joint_speed: Joint velocity limits (rad/s), shape ``(dof,)``.

    Returns:
        Joint positions every ``DT``, shape ``(T, dof)``, and for every sample the welded
        fraction of the seam, shape ``(T,)``: -1 before the arc starts, then 0 to 1.
    """
    s = path.arc_length
    first, last = path.seam_range
    sections = (
        (0, first, process.approach_speed),
        (first, last, process.travel_speed),
        (last, len(s) - 1, process.approach_speed),
    )
    knot_t: List[float] = []
    knot_index: List[int] = []
    clock = 0.0
    arc_on = 0.0
    for k, (i, j, speed) in enumerate(sections):
        local = trapezoid_times(s[i : j + 1] - s[i], speed, process.acceleration)
        # Slow down where a joint would move faster than allowed.
        needed = np.max(np.abs(np.diff(q[i : j + 1], axis=0)) / joint_speed, axis=1)
        local = np.concatenate([[0.0], np.cumsum(np.maximum(np.diff(local), needed))])
        knot_t.extend(clock + local)
        knot_index.extend(range(i, j + 1))
        clock += local[-1]
        if k == 0:
            arc_on = clock
        if k < 2:
            # Hold at the seam start (arc ignition) and end (crater fill). The next
            # section starts with the same point, so the pause is a repeated knot.
            clock += process.arc_dwell
    knot_t = np.array(knot_t)
    knot_q = q[knot_index]
    progress = np.clip((s - s[first]) / max(s[last] - s[first], 1e-9), 0.0, 1.0)[knot_index]
    sample_t = np.arange(0.0, knot_t[-1] + DT, DT)
    sample_t[-1] = min(sample_t[-1], knot_t[-1])
    positions = np.stack(
        [np.interp(sample_t, knot_t, knot_q[:, k]) for k in range(q.shape[1])], axis=1
    )
    bead = np.interp(sample_t, knot_t, progress)
    bead[sample_t < arc_on - 1e-9] = -1.0
    return positions.astype(np.float32), bead.astype(np.float32)


# --------------------------------------------------------------------------------------
# Planning with cuRobo
# --------------------------------------------------------------------------------------


@dataclass
class WeldMotion:
    """A robot's joint motion that welds one seam: approach, seam, retract."""

    #: Robot that welds.
    robot: str
    #: Seam name.
    seam: str
    #: True if the seam is welded from its last point to its first.
    reverse: bool
    #: Joint positions every ``DT``, shape ``(T, dof)``.
    positions: np.ndarray
    #: Welded fraction of the seam at each sample, shape ``(T,)``; -1 before the arc starts.
    bead: np.ndarray
    #: Torch tip path the motion follows, in the world frame.
    tool_path: ToolPath


class OccupancyGrid:
    """Voxel grid that marks the space taken by robot collision spheres.

    A voxel is marked when it intersects a sphere grown by ``margin``. Two sets of
    spheres that mark no common voxel are therefore at least ``2 * margin`` apart.
    """

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        voxel_size: float = 0.03,
        margin: float = 0.02,
        device: torch.device = torch.device("cuda"),
    ):
        """Create an empty grid covering the box from ``lower`` to ``upper`` (m)."""
        self.voxel_size = voxel_size
        self.margin = margin
        self.device = device
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=device)
        extent = torch.as_tensor(upper, dtype=torch.float32, device=device) - self.lower
        self.shape = tuple(int(math.ceil(float(e) / voxel_size)) for e in extent)
        self.num_voxels = self.shape[0] * self.shape[1] * self.shape[2]
        self._offsets: Dict[int, torch.Tensor] = {}

    def empty(self) -> torch.Tensor:
        """Grid with no voxel marked, shape ``(num_voxels,)``."""
        return torch.zeros(self.num_voxels, dtype=torch.bool, device=self.device)

    def mark(self, spheres: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """Voxels touched by spheres ``(M, 4)`` (center, radius) grown by the margin."""
        grid = self.empty()
        spheres = spheres[spheres[:, 3] > 0]
        if spheres.shape[0] == 0:
            return grid
        reach = spheres[:, 3] + self.margin
        half = int(math.ceil(float(reach.max()) / self.voxel_size))
        offsets = self._offset_cube(half)
        dims = torch.tensor(self.shape, device=self.device)
        for start in range(0, spheres.shape[0], chunk):
            center = spheres[start : start + chunk, :3]
            base = torch.floor((center - self.lower) / self.voxel_size).to(torch.int32)
            index = base[:, None, :] + offsets[None]
            box_lower = self.lower + index.to(torch.float32) * self.voxel_size
            gap = (box_lower - center[:, None]).clamp(min=0) + (
                center[:, None] - box_lower - self.voxel_size
            ).clamp(min=0)
            hit = torch.linalg.norm(gap, dim=-1) <= reach[start : start + chunk, None]
            hit &= ((index >= 0) & (index < dims)).all(dim=-1)
            index = index[hit].to(torch.int64)
            grid[(index[:, 0] * self.shape[1] + index[:, 1]) * self.shape[2] + index[:, 2]] = True
        return grid

    def centers(self, grid: torch.Tensor) -> np.ndarray:
        """Centers ``(K, 3)`` of the marked voxels."""
        flat = torch.nonzero(grid).view(-1)
        nz, ny = self.shape[2], self.shape[1]
        index = torch.stack([flat // (ny * nz), (flat // nz) % ny, flat % nz], dim=-1)
        return (self.lower + (index.to(torch.float32) + 0.5) * self.voxel_size).cpu().numpy()

    def _offset_cube(self, half: int) -> torch.Tensor:
        if half not in self._offsets:
            r = torch.arange(-half, half + 1, device=self.device, dtype=torch.int32)
            self._offsets[half] = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).view(
                -1, 3
            )
        return self._offsets[half]


class CellPlanner:
    """cuRobo planning for every robot of a ``WeldCell``.

    The robots share one robot model and differ only in their base pose. Planning for a
    robot loads the cell's obstacles in that robot's base frame (``use_robot``); results
    are converted to and from the world frame.
    """

    def __init__(
        self,
        cell: WeldCell = WeldCell(),
        process: WeldProcess = WeldProcess(),
        robot_file: str = ROBOT_FILE,
        grid: Optional[OccupancyGrid] = None,
    ):
        """Create and warm up the planners.

        Args:
            cell: Work cell with car body, stations, and seams.
            process: Welding process parameters.
            robot_file: Robot config with the tool frame ``TOOL_FRAME``.
            grid: Voxel grid for interlocks; by default it covers every robot's reach.
        """
        self.cell = cell
        self.process = process
        self.robot_file = robot_file
        first = cell.stations[0].name
        num_obstacles = len(cell.obstacles())
        self.planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=robot_file,
                scene_model=cell.scene_cfg(first),
                collision_cache={"cuboid": num_obstacles},
                interpolation_dt=DT,
            )
        )
        self.planner.update_tool_pose_criteria({TOOL_FRAME: FREE_SPIN})
        print(f"Warming up cuRobo for {robot_file} (the first run compiles CUDA kernels)...")
        self.planner.warmup(enable_graph=True, num_warmup_iterations=2)
        # Seam tracker: IK for a batch of neighboring seam points, warm-started from the
        # previous solution. Its LM seed stage starts only from that solution (no random
        # seeds), so it converges to the nearby solution on the same arm configuration.
        self.tracker = InverseKinematics(
            InverseKinematicsCfg.create(
                robot=robot_file,
                scene_model=cell.scene_cfg(first),
                collision_cache={"cuboid": num_obstacles},
                num_seeds=1,
                seed_solver_num_seeds=1,
                max_batch_size=TRACK_CHUNK,
                optimizer_configs=["ik/lbfgs_ik.yml"],
                position_tolerance=0.001,
                orientation_tolerance=0.02,
            ),
            self.planner.scene_collision_checker,
        )
        self.tracker.config.exit_early = False
        self.tracker.update_tool_pose_criteria({TOOL_FRAME: FREE_SPIN})
        self.kinematics: Kinematics = self.planner.kinematics
        self.device = self.planner.default_joint_state.position.device
        #: Home configuration of every robot, shape ``(dof,)``.
        self.home = (
            self.planner.default_joint_state.position.view(-1).cpu().numpy().astype(np.float32)
        )
        limits = self.kinematics.get_joint_limits()
        #: Joint velocity limits (rad/s), shape ``(dof,)``.
        self.joint_speed = limits.velocity[1].view(-1).cpu().numpy()
        #: Joint acceleration limits (rad/s^2), shape ``(dof,)``.
        self.max_acceleration = limits.acceleration[1].view(-1).cpu().numpy()
        self._robot = first
        if grid is None:
            positions = np.array([s.position for s in cell.stations])
            reach = 1.45
            grid = OccupancyGrid(
                [positions[:, 0].min() - reach, positions[:, 1].min() - reach, 0.0],
                [
                    positions[:, 0].max() + reach,
                    positions[:, 1].max() + reach,
                    positions[:, 2].max() + reach,
                ],
                device=self.device,
            )
        self.grid = grid

    @property
    def joint_names(self) -> List[str]:
        """Joint names of a single robot."""
        return list(self.planner.joint_names)

    def destroy(self) -> None:
        """Release CUDA graph resources."""
        self.tracker.destroy()
        self.planner.destroy()

    def use_robot(self, robot: str) -> None:
        """Load the cell's obstacles in the base frame of ``robot``."""
        if robot != self._robot:
            self.planner.update_world(Scene.create(self.cell.scene_cfg(robot)))
            self._robot = robot

    # ---------------------------------------------------------------- seams

    def plan_weld(self, robot: str, seam: Seam, reverse: bool = False) -> Optional[WeldMotion]:
        """Joint motion for ``robot`` to weld ``seam``, or ``None`` if it cannot.

        IK for the first tool pose (the start of the approach) gives several arm
        configurations, tried closest to home first. From each, the whole tool path is
        tracked with warm-started IK. The first configuration from which every point is
        reached, without collision and without a jump between arm configurations, is
        used. IK starts from random seeds, so a seam that fails is tried once more with
        new seeds.
        """
        self.use_robot(robot)
        path = seam_tool_path(seam, self.process, reverse)
        position, rotation = self._path_in_base(robot, path)
        start_goal = self._goal(position[:1], rotation[:1])
        home = torch.as_tensor(self.home, device=self.device)
        for _attempt in range(PLAN_WELD_ATTEMPTS):
            start = self.planner.ik_solver.solve_pose(start_goal, return_seeds=NUM_START_SOLUTIONS)
            success = start.success.view(-1)
            if not success.any():
                continue
            candidates = start.solution.view(-1, len(self.home))[success]
            order = torch.argsort((candidates - home).abs().sum(dim=-1))
            for k in order.tolist():
                q = self._track(position, rotation, candidates[k])
                if q is not None:
                    positions, bead = time_tool_path(
                        q.cpu().numpy(),
                        path,
                        self.process,
                        self.joint_speed * WELD_JOINT_SPEED_SCALE,
                    )
                    return WeldMotion(robot, seam.name, reverse, positions, bead, path)
        return None

    def _track(
        self, position: torch.Tensor, rotation: torch.Tensor, q0: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Joint positions ``(N, dof)`` that follow tool poses from ``q0``.

        Args:
            position: Tool positions in the base frame, shape ``(N, 3)``.
            rotation: Tool orientations in the base frame, shape ``(N, 3, 3)``. Only the
                torch axis (z) is used; the rotation about it is free.
            q0: Joint configuration near the first pose, shape ``(dof,)``.
        """
        solutions = []
        last = q0.view(1, -1)
        for start in range(0, position.shape[0], TRACK_CHUNK):
            stop = min(start + TRACK_CHUNK, position.shape[0])
            # The rotation error is measured with a log map, which is ill-conditioned when
            # the free spin about the torch axis is far from the goal's. The goals of this
            # chunk therefore keep the spin the torch has at the previous solution.
            goal = self._goal(position[start:stop], self._keep_spin(rotation[start:stop], last))
            result = self.tracker.solve_pose(
                goal, seed_config=last.view(1, 1, -1).repeat(stop - start, 1, 1)
            )
            if not result.success.view(-1).all():
                return None
            solution = result.solution.view(stop - start, -1)
            solutions.append(solution)
            last = solution[-1:].clone()
        q = torch.cat([q0.view(1, -1)] + solutions)
        if float((q[1:] - q[:-1]).abs().max()) > MAX_JOINT_STEP:
            return None
        return q[1:]

    def _keep_spin(self, rotation: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Orientations ``(m, 3, 3)`` with their torch axes, spun like the tool at ``q``."""
        state = JointState.from_position(q.view(1, -1), joint_names=self.joint_names)
        tool = self.kinematics.compute_kinematics(state).tool_poses[TOOL_FRAME]
        current_x = Pose(
            position=tool.position.view(-1, 3), quaternion=tool.quaternion.view(-1, 4)
        ).get_rotation()[0, :, 0]
        z = rotation[:, :, 2]
        x = current_x - (z @ current_x)[:, None] * z
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        return torch.stack([x, torch.cross(z, x, dim=-1), z], dim=-1)

    def _path_in_base(self, robot: str, path: ToolPath) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tool path in the base frame: positions ``(N, 3)``, orientations ``(N, 3, 3)``."""
        station = self.cell.station(robot)
        position = torch.as_tensor(
            station.to_base(path.positions), device=self.device, dtype=torch.float32
        )
        rotation = torch.as_tensor(
            np.einsum("ji,njk->nik", station.rotation, path.rotations),
            device=self.device,
            dtype=torch.float32,
        )
        return position, rotation

    def _goal(self, position: torch.Tensor, rotation: torch.Tensor) -> GoalToolPose:
        pose = Pose(position=position.contiguous(), rotation=rotation.contiguous())
        return GoalToolPose.from_poses({TOOL_FRAME: pose}, num_goalset=1)

    # ---------------------------------------------------------------- transfers

    def plan_transfer(
        self, robot: str, start: np.ndarray, goal: np.ndarray
    ) -> Optional[np.ndarray]:
        """Collision-free joint motion ``(T, dof)`` every ``DT`` from ``start`` to ``goal``.

        A single trajectory optimization is tried first; the full planner with graph
        search is the fallback.
        """
        self.use_robot(robot)
        names = self.joint_names
        current = JointState.from_position(
            torch.as_tensor(start, device=self.device).view(1, -1), joint_names=names
        )
        target = JointState.from_position(
            torch.as_tensor(goal, device=self.device).view(1, -1), joint_names=names
        )
        result = self.planner.trajopt_solver.solve_cspace(target, current, finetune_attempts=1)
        if result is None or not result.success.any():
            result = self.planner.plan_cspace(target, current, max_attempts=3)
        if result is None or not result.success.any():
            return None
        traj = result.get_interpolated_plan()
        return traj.position.reshape(-1, traj.position.shape[-1]).cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------------- geometry of robots

    def spheres(self, robot: str, positions: np.ndarray) -> torch.Tensor:
        """World-frame collision spheres ``(T, S, 4)`` at joint positions ``(T, dof)``."""
        q = torch.as_tensor(positions, device=self.device, dtype=torch.float32).view(
            -1, len(self.home)
        )
        state = JointState.from_position(q, joint_names=self.joint_names)
        spheres = (
            self.kinematics.compute_kinematics(state).robot_spheres.view(q.shape[0], -1, 4).clone()
        )
        station = self.cell.station(robot)
        rotation = torch.as_tensor(station.rotation, device=self.device, dtype=torch.float32)
        offset = torch.as_tensor(station.position, device=self.device, dtype=torch.float32)
        spheres[..., :3] = spheres[..., :3] @ rotation.T + offset
        return spheres

    def tool_poses(self, robot: str, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Torch tip positions ``(T, 3)`` and orientations ``(T, 3, 3)`` in the world frame."""
        q = torch.as_tensor(positions, device=self.device, dtype=torch.float32).view(
            -1, len(self.home)
        )
        pose = self.kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self.joint_names)
        ).tool_poses[TOOL_FRAME]
        station = self.cell.station(robot)
        position = station.to_world(pose.position.view(-1, 3).cpu().numpy())
        rotation = Pose(
            position=pose.position.view(-1, 3), quaternion=pose.quaternion.view(-1, 4)
        ).get_rotation()
        rotation = np.einsum("ij,njk->nik", station.rotation, rotation.cpu().numpy())
        return position, rotation

    def occupancy(self, robot: str, positions: np.ndarray, step: float = 0.01) -> torch.Tensor:
        """Voxels swept by ``robot`` along joint positions ``(T, dof)``.

        The motion is resampled so that no sphere moves more than ``step`` between two
        samples; the grid margin covers the space in between.
        """
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, len(self.home))
        spheres = self.spheres(robot, positions)
        if positions.shape[0] > 1:
            moved = (
                torch.linalg.norm(spheres[1:, :, :3] - spheres[:-1, :, :3], dim=-1)
                .max(dim=-1)
                .values
            )
            travel = np.concatenate([[0.0], torch.cumsum(moved, 0).cpu().numpy()])
            samples = np.linspace(0.0, travel[-1], max(2, int(math.ceil(travel[-1] / step)) + 1))
            # Interpolate in time, then in joint space, at evenly spaced sphere travel.
            index = np.interp(samples, travel, np.arange(len(travel)))
            low = np.floor(index).astype(int)
            high = np.minimum(low + 1, len(travel) - 1)
            alpha = (index - low)[:, None]
            dense = (1 - alpha) * positions[low] + alpha * positions[high]
            spheres = torch.cat([spheres, self.spheres(robot, dense)], dim=0)
        return self.grid.mark(spheres.view(-1, 4))


# --------------------------------------------------------------------------------------
# Interlocks and scheduling
# --------------------------------------------------------------------------------------


@dataclass
class Motion:
    """A piece of one robot's timeline."""

    #: ``TRANSFER``, ``WELD``, or ``HOME``.
    kind: str
    #: Tick at which the motion starts; one tick is ``DT``.
    start: int
    #: Joint positions every tick, shape ``(T, dof)``. The motion lasts ``T - 1`` ticks.
    positions: np.ndarray
    #: Seam that is welded (``WELD``) or approached (``TRANSFER``).
    seam: Optional[str] = None
    #: Welded fraction of the seam at each sample (``WELD`` only), shape ``(T,)``.
    bead: Optional[np.ndarray] = None
    #: Torch tip path of a ``WELD`` motion.
    tool_path: Optional[ToolPath] = None

    @property
    def end(self) -> int:
        """Tick at which the motion ends."""
        return self.start + len(self.positions) - 1


@dataclass
class InterlockWait:
    """A span of time in which a robot waited for space held by other robots."""

    #: Robot that waited.
    robot: str
    #: First tick of the wait.
    start: int
    #: Tick at which the robot started moving again.
    end: int
    #: Robots that held the space.
    blocked_by: Tuple[str, ...]
    #: Seam the robot wanted to weld next; ``None`` if it wanted to go home.
    seam: Optional[str]
    #: Centers of the voxels both robots needed, shape ``(K, 3)``.
    shared_space: np.ndarray


@dataclass
class Deferral:
    """A robot welded another seam first because its preferred seam was blocked."""

    #: Robot that changed its plan.
    robot: str
    #: Tick of the decision.
    tick: int
    #: Seam the robot preferred.
    wanted: str
    #: Seam the robot welded instead.
    taken: str
    #: Robots that held the space of the preferred seam.
    blocked_by: Tuple[str, ...]


@dataclass
class Reservation:
    """Space a robot holds for a span of time: a running job or the pose it stands in."""

    #: Robot that holds the space.
    robot: str
    #: First tick.
    start: int
    #: Tick at which the robot released the space.
    end: int
    #: Centers of the held voxels, shape ``(K, 3)``.
    voxels: np.ndarray


@dataclass
class CellPlan:
    """Planned programs of all robots in a cell."""

    #: Robots in the plan.
    robots: List[str]
    #: Timeline of each robot. Between motions a robot stands still.
    motions: Dict[str, List[Motion]]
    #: Interlock waits of all robots.
    waits: List[InterlockWait]
    #: Seams welded out of preference order because of interlocks.
    deferrals: List[Deferral]
    #: Robot that welded each seam.
    assignment: Dict[str, str]
    #: Seams no robot could weld.
    leftover: List[str]
    #: Home configuration of the robots, shape ``(dof,)``.
    home: np.ndarray
    #: Wall-clock planning time (s).
    planning_time: float
    #: Space held by each robot over time; an interlock admits a job only if its space
    #: does not overlap the space other robots hold.
    reservations: List[Reservation] = field(default_factory=list)
    #: ``(tick, robot)`` of every retreat home that resolved a mutual block.
    retreats: List[Tuple[int, str]] = field(default_factory=list)
    #: Program steps a robot could not weld, handed to another robot: ``(seam, from, to)``.
    handovers: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def yields(self) -> int:
        """Number of retreats home."""
        return len(self.retreats)

    @property
    def num_ticks(self) -> int:
        """Duration of the plan in ticks."""
        return max((m.end for ms in self.motions.values() for m in ms), default=0)

    @property
    def cycle_time(self) -> float:
        """Duration of the plan (s)."""
        return self.num_ticks * DT

    def weld_order(self, robot: str) -> List[str]:
        """Seams welded by ``robot`` in order."""
        return [m.seam for m in self.motions[robot] if m.kind == WELD]

    def joint_trajectory(self, robot: str) -> np.ndarray:
        """Joint positions of ``robot`` at every tick, shape ``(num_ticks + 1, dof)``."""
        out = np.repeat(self.home[None], self.num_ticks + 1, axis=0)
        for motion in self.motions[robot]:
            out[motion.start : motion.end + 1] = motion.positions
            out[motion.end + 1 :] = motion.positions[-1]
        return out

    def wait_ticks(self, robot: str) -> int:
        """Ticks ``robot`` spent waiting for interlocks."""
        return sum(w.end - w.start for w in self.waits if w.robot == robot)


@dataclass
class _RobotState:
    """Book-keeping of one robot while the schedule is built."""

    q: np.ndarray
    occupancy: torch.Tensor
    busy_until: int = 0
    idle_since: int = 0
    at_home: bool = True
    finished: bool = False
    wait: Optional[InterlockWait] = None
    yields: int = 0
    reservation: Optional[Reservation] = None
    #: Upcoming ``(tick, space)`` changes of the held space while a job runs.
    releases: List[Tuple[int, torch.Tensor]] = field(default_factory=list)


class WeldScheduler:
    """Builds a cell program: which robot welds which seam, when, and who waits for whom.

    The robots' programs run as a discrete-event simulation. Whenever a robot is idle it
    picks a job: a transfer to a seam followed by the weld. With fixed programs the seam
    is the next one in the robot's program. Otherwise it is the best unwelded seam the
    robot can reach; seams only this robot can weld are preferred, and seams that several
    robots can reach go to whichever robot takes them first.

    An interlock admits a job only if the space it sweeps (``CellPlanner.occupancy``) is
    disjoint from the space every other robot holds: the motion it is executing, or the
    pose it stands in. A robot whose jobs are all blocked waits. Robots that block each
    other while standing still resolve this by one of them retreating home.
    """

    def __init__(
        self,
        planner: CellPlanner,
        seams: Sequence[Seam],
        robots: Sequence[str],
        programs: Optional[Dict[str, Sequence[str]]] = None,
        interlocks: bool = True,
        max_tries: int = 4,
        shared_penalty: float = 2.0,
        max_yields: int = 6,
    ):
        """Set up the scheduler.

        Args:
            planner: Plans welds and transfers and measures occupied space.
            seams: Seams to weld.
            robots: Robots in the cell.
            programs: Seams each robot welds, in order. ``None`` assigns seams while the
                program runs: an idle robot takes the best unwelded seam it can reach.
            interlocks: Check robots against each other. Without interlocks every robot
                runs its program as if it were alone, to show what would collide.
            max_tries: Seams tried per decision, in order of preference.
            shared_penalty: Time (s) added to a seam's cost for every other robot that
                could also weld it.
            max_yields: Retreats home allowed per robot.
        """
        self.interlocks = interlocks
        self.programs = (
            None if programs is None else {r: list(programs.get(r, [])) for r in robots}
        )
        #: Robot whose program lists each seam, including robots missing from the cell.
        self._owner = {n: r for r, program in (programs or {}).items() for n in program}
        self.planner = planner
        self.seams = list(seams)
        self.robots = list(robots)
        self.max_tries = max_tries
        self.shared_penalty = shared_penalty
        self.max_yields = max_yields

    def plan(self, progress: Optional[Callable[[str], None]] = None) -> CellPlan:
        """Plan every robot's welds and build the interlocked schedule.

        Args:
            progress: Optional callback ``progress(message)`` for status updates.

        Returns:
            The cell program.
        """
        report = progress or (lambda message: None)
        start_time = time.time()
        self._options: Dict[str, List[WeldMotion]] = {r: [] for r in self.robots}
        for k, seam in enumerate(self.seams):
            report(f"Planning seam {k + 1}/{len(self.seams)}: {seam.name}")
            for robot in self.robots:
                if self.programs is not None and seam.name not in self.programs[robot]:
                    continue
                for reverse in (False, True):
                    motion = self.planner.plan_weld(robot, seam, reverse)
                    if motion is not None:
                        self._options[robot].append(motion)
        self._capable = {
            s.name: {r for r in self.robots if any(o.seam == s.name for o in self._options[r])}
            for s in self.seams
        }
        self._queue = None
        self._handovers: List[Tuple[str, str, str]] = []
        if self.programs is not None:
            self._queue = {
                r: [n for n in self.programs[r] if r in self._capable[n]] for r in self.robots
            }
            self._hand_over(report)
        self._occupancy_cache: Dict[Tuple[str, str, bool], torch.Tensor] = {}
        self._transfer_cache: Dict[Tuple, Optional[Tuple[np.ndarray, torch.Tensor]]] = {}
        self._rest_cache: Dict[Tuple[str, bytes], torch.Tensor] = {}
        self._motions: Dict[str, List[Motion]] = {r: [] for r in self.robots}
        self._waits: List[InterlockWait] = []
        self._deferrals: List[Deferral] = []
        self._reservations: List[Reservation] = []
        self._retreats: List[Tuple[int, str]] = []
        self._assignment: Dict[str, str] = {}
        home = self.planner.home
        self._state = {}
        for robot in self.robots:
            space = self._rest(robot, home)
            self._state[robot] = _RobotState(q=home.copy(), occupancy=space)
            self._hold(robot, 0, space)

        tick = 0
        while True:
            report(
                f"Scheduling: t = {tick * DT:.1f} s, "
                f"{len(self._assignment)}/{len(self.seams)} seams assigned"
            )
            idle = [
                r
                for r in self.robots
                if self._state[r].busy_until <= tick and not self._state[r].finished
            ]
            for robot in sorted(idle, key=lambda r: self._state[r].idle_since):
                self._try_start(robot, tick)
            busy = [r for r in self.robots if self._state[r].busy_until > tick]
            if not busy:
                if all(self._state[r].finished for r in self.robots):
                    break
                if not self._yield(tick, [r for r in self.robots if not self._state[r].finished]):
                    break
                continue
            stuck = self._blocked_by_each_other(tick)
            if stuck and self._yield(tick, stuck):
                continue
            releases = [t for r in busy for t, _ in self._state[r].releases]
            tick = min([self._state[r].busy_until for r in busy] + releases)
            for robot in busy:
                state = self._state[robot]
                while state.releases and state.releases[0][0] <= tick:
                    self._hold(robot, tick, state.releases.pop(0)[1])
                if state.busy_until == tick:
                    self._hold(robot, tick, self._rest(robot, state.q))
                    state.idle_since = tick

        for state in self._state.values():
            if state.wait is not None:
                self._close_wait(state, tick)
            state.reservation.end = tick
            self._reservations.append(state.reservation)
        welded = set(self._assignment)
        return CellPlan(
            robots=self.robots,
            motions=self._motions,
            waits=self._waits,
            deferrals=self._deferrals,
            assignment=dict(self._assignment),
            leftover=[s.name for s in self.seams if s.name not in welded],
            home=home.copy(),
            planning_time=time.time() - start_time,
            reservations=self._reservations,
            retreats=self._retreats,
            handovers=self._handovers,
        )

    def _hand_over(self, report: Callable[[str], None]) -> None:
        """Give seams that no program can weld to a robot that can reach them.

        A program step the robot cannot weld, or a seam in no robot's program (for
        example when a robot is missing from the cell), goes to the end of the shortest
        program among the robots that can weld it.
        """
        queued = {n for queue in self._queue.values() for n in queue}
        for seam in self.seams:
            if seam.name in queued:
                continue
            report(f"Finding another robot for {seam.name}")
            for robot in sorted(self.robots, key=lambda r: len(self._queue[r])):
                if robot not in self._capable[seam.name]:
                    for reverse in (False, True):
                        motion = self.planner.plan_weld(robot, seam, reverse)
                        if motion is not None:
                            self._options[robot].append(motion)
                            self._capable[seam.name].add(robot)
                if robot in self._capable[seam.name]:
                    if robot == self._owner.get(seam.name):
                        # Planned on the second try: keep it at its place in the program.
                        program = self.programs[robot]
                        later = program[program.index(seam.name) + 1 :]
                        queue = self._queue[robot]
                        index = next((queue.index(n) for n in later if n in queue), len(queue))
                        queue.insert(index, seam.name)
                    else:
                        self._queue[robot].append(seam.name)
                        self._handovers.append((seam.name, self._owner.get(seam.name, "-"), robot))
                    break

    # ---------------------------------------------------------------- decisions

    def _try_start(self, robot: str, tick: int) -> bool:
        """Start the robot's best admissible job at ``tick``; return False if it has to wait."""
        state = self._state[robot]
        if self._queue is not None:
            queue = self._queue[robot]
            options = [o for o in self._options[robot] if queue and o.seam == queue[0]]
        else:
            options = [o for o in self._options[robot] if o.seam not in self._assignment]
        if not options:
            return self._go_home(robot, tick, finish=True)
        ranked = sorted(options, key=lambda o: self._cost(robot, o))
        blockers: Set[str] = set()
        shared: Optional[torch.Tensor] = None
        for option in ranked[: self.max_tries]:
            weld_space = self._weld_occupancy(robot, option)
            hits = self._conflicts(robot, weld_space)
            if hits:
                blockers.update(hits)
                shared = (
                    shared if shared is not None else torch.stack(list(hits.values())).any(dim=0)
                )
                continue
            planned = self._transfer(robot, state.q, option.positions[0])
            if planned is None:
                continue
            transfer, transfer_space = planned
            hits = self._conflicts(robot, transfer_space)
            if hits:
                blockers.update(hits)
                shared = (
                    shared if shared is not None else torch.stack(list(hits.values())).any(dim=0)
                )
                continue
            self._assignment[option.seam] = robot
            if self._queue is not None:
                self._queue[robot].pop(0)
            if blockers and option.seam != ranked[0].seam:
                self._deferrals.append(
                    Deferral(robot, tick, ranked[0].seam, option.seam, tuple(sorted(blockers)))
                )
            transfer_motion = Motion(TRANSFER, tick, transfer, seam=option.seam)
            weld = Motion(
                WELD,
                transfer_motion.end,
                option.positions,
                option.seam,
                option.bead,
                option.tool_path,
            )
            self._start(robot, tick, [transfer_motion, weld], [transfer_space, weld_space])
            return True
        if blockers:
            self._open_wait(robot, tick, blockers, ranked[0].seam, shared)
        elif not state.at_home:
            # No job could be planned from here; try again from home.
            return self._go_home(robot, tick, finish=False)
        else:
            # Nothing reachable from home either: give up the remaining seams.
            self._options[robot] = []
        return False

    def _go_home(self, robot: str, tick: int, finish: bool) -> bool:
        """Move the robot home; it stays there for good if ``finish``."""
        state = self._state[robot]
        if state.at_home:
            state.finished = finish
            if state.wait is not None:
                self._close_wait(state, tick)
            return False
        planned = self._transfer(robot, state.q, self.planner.home)
        if planned is None:
            return False
        path, space = planned
        hits = self._conflicts(robot, space)
        if hits:
            self._open_wait(
                robot, tick, set(hits), None, torch.stack(list(hits.values())).any(dim=0)
            )
            return False
        self._start(robot, tick, [Motion(HOME, tick, path)], [space])
        state.at_home = True
        state.finished = finish
        return True

    def _yield(self, tick: int, candidates: Sequence[str]) -> bool:
        """Send one waiting robot home to break a mutual block; False if none can go."""
        order = sorted(candidates, key=lambda r: -self._state[r].idle_since)
        for robot in order:
            state = self._state[robot]
            if state.at_home or state.yields >= self.max_yields:
                continue
            if self._go_home(robot, tick, finish=False):
                state.yields += 1
                self._retreats.append((tick, robot))
                return True
        return False

    def _blocked_by_each_other(self, tick: int) -> List[str]:
        """Idle robots that wait for each other (a cycle of waits); empty if there is none."""
        waiting = {
            r: set(self._state[r].wait.blocked_by)
            for r in self.robots
            if self._state[r].wait is not None and self._state[r].busy_until <= tick
        }
        for robot in waiting:
            seen, frontier = set(), [robot]
            while frontier:
                current = frontier.pop()
                for other in waiting.get(current, ()):
                    if other == robot:
                        return [r for r in waiting if r in seen or r == robot]
                    if other in waiting and other not in seen:
                        seen.add(other)
                        frontier.append(other)
        return []

    def _cost(self, robot: str, option: WeldMotion) -> float:
        """Estimated time to reach a seam, plus a penalty if other robots could weld it."""
        others = len(
            self._capable[option.seam]
            - {robot}
            - {r for r in self.robots if self._state[r].finished}
        )
        return (
            self._transfer_estimate(self._state[robot].q, option.positions[0])
            + self.shared_penalty * others
        )

    def _transfer_estimate(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Rest-to-rest time of the slowest joint at full acceleration (s)."""
        return float(np.max(2.0 * np.sqrt(np.abs(goal - start) / self.planner.max_acceleration)))

    # ---------------------------------------------------------------- book-keeping

    def _start(
        self, robot: str, tick: int, motions: List[Motion], spaces: List[torch.Tensor]
    ) -> None:
        """Start a job: consecutive ``motions`` that sweep ``spaces``.

        The robot reserves the space of the whole job up front, so it never has to stop
        halfway, and releases the space of each motion once that motion is done.
        """
        state = self._state[robot]
        if state.wait is not None:
            self._close_wait(state, tick)
        self._motions[robot].extend(motions)
        state.busy_until = motions[-1].end
        state.q = motions[-1].positions[-1].copy()
        remaining = [torch.stack(spaces[k:]).any(dim=0) for k in range(len(spaces))]
        self._hold(robot, tick, remaining[0])
        state.releases = [(m.end, space) for m, space in zip(motions[:-1], remaining[1:])]
        state.at_home = False

    def _hold(self, robot: str, tick: int, space: torch.Tensor) -> None:
        """Make ``space`` the space ``robot`` holds from ``tick`` on."""
        state = self._state[robot]
        if state.reservation is not None:
            state.reservation.end = tick
            if tick > state.reservation.start:
                self._reservations.append(state.reservation)
        state.occupancy = space
        state.reservation = Reservation(robot, tick, tick, self.planner.grid.centers(space))

    def _conflicts(self, robot: str, space: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Other robots whose held space overlaps ``space``, with the overlap."""
        hits = {}
        if not self.interlocks:
            return hits
        for other in self.robots:
            if other != robot:
                overlap = space & self._state[other].occupancy
                if overlap.any():
                    hits[other] = overlap
        return hits

    def _open_wait(
        self, robot: str, tick: int, blockers: Set[str], seam: Optional[str], shared: torch.Tensor
    ) -> None:
        state = self._state[robot]
        if state.wait is None:
            state.wait = InterlockWait(robot, tick, tick, (), seam, np.zeros((0, 3)))
        state.wait.blocked_by = tuple(sorted(blockers))
        state.wait.seam = seam
        state.wait.shared_space = self.planner.grid.centers(shared)

    def _close_wait(self, state: _RobotState, tick: int) -> None:
        state.wait.end = tick
        if state.wait.end > state.wait.start:
            self._waits.append(state.wait)
        state.wait = None

    def _weld_occupancy(self, robot: str, option: WeldMotion) -> torch.Tensor:
        key = (robot, option.seam, option.reverse)
        if key not in self._occupancy_cache:
            self._occupancy_cache[key] = self.planner.occupancy(robot, option.positions)
        return self._occupancy_cache[key]

    def _rest(self, robot: str, q: np.ndarray) -> torch.Tensor:
        key = (robot, np.asarray(q, dtype=np.float32).tobytes())
        if key not in self._rest_cache:
            self._rest_cache[key] = self.planner.occupancy(robot, np.asarray(q)[None])
        return self._rest_cache[key]

    def _transfer(
        self, robot: str, start: np.ndarray, goal: np.ndarray
    ) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
        """Transfer motion ``(T, dof)`` and the space it sweeps, or ``None``."""
        key = (
            robot,
            np.asarray(start, dtype=np.float32).tobytes(),
            np.asarray(goal, dtype=np.float32).tobytes(),
        )
        if key not in self._transfer_cache:
            path = self.planner.plan_transfer(robot, start, goal)
            space = None if path is None else self.planner.occupancy(robot, path)
            self._transfer_cache[key] = None if path is None else (path, space)
        return self._transfer_cache[key]


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


@dataclass
class PlanCheck:
    """Independent checks of a cell program."""

    #: Smallest distance between collision spheres of two robots over the plan (m).
    min_clearance: float
    #: Time at which ``min_clearance`` occurs (s).
    min_clearance_time: float
    #: Largest distance of the torch tip from the seam while welding (m).
    max_seam_deviation: float
    #: Largest angle between the torch axis and its goal direction while welding (rad).
    max_torch_angle_error: float
    #: Mean and largest relative deviation of the tip speed from the travel speed while
    #: welding at full speed, measured over ``SPEED_WINDOW`` ticks.
    speed_error: Tuple[float, float]
    #: Collisions between robots: ``(first tick, robot, robot, depth (m))``.
    collisions: List[Tuple[int, str, str, float]] = field(default_factory=list)


@dataclass
class ClearanceProfile:
    """Smallest robot-to-robot distance at every tick, measured on the collision spheres."""

    #: Clearance (m), shape ``(num_ticks + 1,)``; negative where two robots intersect.
    clearance: np.ndarray
    #: Midpoint between the two closest spheres, shape ``(num_ticks + 1, 3)``.
    contact: np.ndarray
    #: The two closest robots at every tick.
    pair: List[Tuple[str, str]]


def clearance_profile(planner: CellPlanner, plan: CellPlan, chunk: int = 400) -> ClearanceProfile:
    """Measure the smallest robot-to-robot distance at every tick of a plan."""
    trajectories = {r: plan.joint_trajectory(r) for r in plan.robots}
    num = plan.num_ticks + 1
    clearance = np.full(num, np.inf)
    contact = np.zeros((num, 3))
    pair_index = np.zeros(num, dtype=int)
    pairs = [(a, b) for i, a in enumerate(plan.robots) for b in plan.robots[i + 1 :]]
    for start in range(0, num, chunk):
        spheres = {}
        for robot in plan.robots:
            s = planner.spheres(robot, trajectories[robot][start : start + chunk])
            spheres[robot] = s[:, s[0, :, 3] > 0]
        for k, (a, b) in enumerate(pairs):
            sa, sb = spheres[a], spheres[b]
            gap = torch.cdist(sa[..., :3], sb[..., :3]) - sa[..., 3:4] - sb[..., 3].unsqueeze(1)
            values, flat = gap.flatten(1).min(dim=1)
            ia, ib = flat // sb.shape[1], flat % sb.shape[1]
            ticks = torch.arange(len(values), device=gap.device)
            mid = 0.5 * (sa[ticks, ia, :3] + sb[ticks, ib, :3])
            values, mid = values.cpu().numpy(), mid.cpu().numpy()
            window = slice(start, start + len(values))
            closer = values < clearance[window]
            clearance[window] = np.where(closer, values, clearance[window])
            contact[window] = np.where(closer[:, None], mid, contact[window])
            pair_index[window] = np.where(closer, k, pair_index[window])
    return ClearanceProfile(clearance, contact, [pairs[k] for k in pair_index] if pairs else [])


def collision_episodes(profile: ClearanceProfile) -> List[Tuple[int, str, str, float]]:
    """Runs of ticks in which robots intersect: ``(first tick, robot, robot, depth (m))``."""
    episodes = []
    colliding = profile.clearance < 0.0
    tick = 0
    while tick < len(colliding):
        if not colliding[tick]:
            tick += 1
            continue
        end = tick
        while end + 1 < len(colliding) and colliding[end + 1]:
            end += 1
        a, b = profile.pair[tick]
        episodes.append((tick, a, b, float(-profile.clearance[tick : end + 1].min())))
        tick = end + 1
    return episodes


def check_plan(planner: CellPlanner, plan: CellPlan) -> PlanCheck:
    """Measure robot-to-robot clearance and Cartesian weld accuracy of a plan."""
    profile = clearance_profile(planner, plan)
    clearance = profile.clearance
    best_tick = int(np.argmin(clearance))
    best = float(clearance[best_tick])
    deviation, angle, speed_errors = 0.0, 0.0, []
    travel = planner.process.travel_speed
    for robot in plan.robots:
        for motion in plan.motions[robot]:
            if motion.kind != WELD:
                continue
            arc = (motion.bead >= 0) & (motion.bead < 1)
            position, rotation = planner.tool_poses(robot, motion.positions[arc])
            path = motion.tool_path
            first, last = path.seam_range
            seam_points = path.positions[first : last + 1]
            nearest = np.argmin(
                np.linalg.norm(position[:, None] - seam_points[None], axis=-1), axis=1
            )
            # Distance to the seam polyline: to the nearest point's neighboring segments.
            deviation = max(deviation, float(_polyline_distance(position, seam_points).max()))
            goal_z = path.rotations[first + nearest][:, :, 2]
            cos = np.clip(np.sum(rotation[:, :, 2] * goal_z, axis=-1), -1.0, 1.0)
            angle = max(angle, float(np.arccos(cos).max()))
            # Tip speed over windows of SPEED_WINDOW ticks, away from the seam ends.
            bead = motion.bead[arc]
            w = SPEED_WINDOW
            cruise = (bead[:-w] > 0.05) & (bead[w:] < 0.95)
            speed = np.linalg.norm(position[w:] - position[:-w], axis=-1) / (w * DT)
            speed_errors.extend(np.abs(speed[cruise] / travel - 1.0))
    speed_errors = np.array(speed_errors) if speed_errors else np.zeros(1)
    return PlanCheck(
        best,
        best_tick * DT,
        deviation,
        angle,
        (float(speed_errors.mean()), float(speed_errors.max())),
        collision_episodes(profile),
    )


def _polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance of points ``(N, 3)`` to a polyline ``(K, 3)``."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    t = np.clip(
        np.einsum("nkc,kc->nk", points[:, None] - a[None], ab)
        / np.maximum(np.sum(ab * ab, axis=-1), 1e-12),
        0,
        1,
    )
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None] - closest, axis=-1).min(axis=1)


# --------------------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------------------

ROBOT_COLORS = {
    "left_front": (66, 133, 244),
    "left_rear": (0, 172, 193),
    "right_front": (244, 140, 46),
    "right_rear": (214, 180, 20),
    "front": (156, 39, 176),
    "rear": (52, 168, 83),
}
#: Color of the space a waiting robot needs but another robot holds.
CONFLICT_COLOR = (230, 40, 40)
BODY_COLOR = (196, 202, 210)
SEAM_COLOR = (90, 90, 96)
ARC_COLOR = (255, 236, 140)


@dataclass
class _Playback:
    """A planned cell program prepared for playback."""

    #: The program.
    plan: CellPlan
    #: Its checks.
    check: PlanCheck
    #: True if the program was planned with interlocks.
    interlocks: bool
    #: Joint positions per robot at every tick, shape ``(num_ticks + 1, dof)``.
    joints: Dict[str, np.ndarray]
    #: Robot-to-robot clearance at every tick.
    profile: ClearanceProfile
    #: ``(tick, text)`` of interlock events in time order, for the event log.
    events: List[Tuple[int, str]]


#: Lines shown in the viewer's interlock log.
EVENT_LOG_LINES = 8


def _interlock_events(plan: CellPlan, profile: ClearanceProfile) -> List[Tuple[int, str]]:
    """``(tick, text)`` of the plan's interlock events and its collisions, in time order."""
    events = [
        (0, f"t=0.0 s: {seam} from {owner}'s program goes to {robot}")
        for seam, owner, robot in plan.handovers
    ]
    for w in plan.waits:
        events.append(
            (
                w.start,
                f"t={w.start * DT:.1f} s: **{w.robot}** waits for {'+'.join(w.blocked_by)} "
                f"({w.seam or 'home'}), {(w.end - w.start) * DT:.1f} s",
            )
        )
    for d in plan.deferrals:
        events.append(
            (
                d.tick,
                f"t={d.tick * DT:.1f} s: **{d.robot}** welds {d.taken} first, "
                f"{d.wanted} is held by {'+'.join(d.blocked_by)}",
            )
        )
    for tick, robot in plan.retreats:
        events.append((tick, f"t={tick * DT:.1f} s: **{robot}** retreats home to unblock"))
    for tick, a, b, depth in collision_episodes(profile):
        text = f"t={tick * DT:.1f} s: **{a} and {b} collide** ({depth * 1000:.0f} mm)"
        events.append((tick, f"<span style='color:#d00'>{text}</span>"))
    return sorted(events, key=lambda e: e[0])


class WeldingViewer:
    """Viser front end: draws the cell, plans on request, and plays back the program."""

    def __init__(
        self, planner: CellPlanner, robots: Sequence[str], port: int, dynamic: bool = False
    ):
        """Start the Viser server and draw the cell.

        Args:
            planner: Cell planner (already warmed up).
            robots: Robots in the cell.
            port: Viser server port.
            dynamic: Initial state of the "assign seams at run time" checkbox.
        """
        import viser
        import yourdfpy

        self.planner = planner
        self.robots = list(robots)
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.add_grid("/ground", width=5, height=5, cell_size=0.25)
        self._seams = {s.name: s for s in planner.cell.seams()}
        self._draw_cell()

        kin = KinematicsCfg.from_robot_yaml_file(planner.robot_file).generator_config
        root = kin.asset_root_path
        urdf = yourdfpy.URDF.load(
            kin.urdf_path,
            load_meshes=True,
            build_scene_graph=True,
            filename_handler=lambda fname: f"{root}/{fname.replace('package://', '')}",
        )
        from viser.extras import ViserUrdf

        self._urdfs = {}
        for robot in self.robots:
            station = planner.cell.station(robot)
            self.server.scene.add_frame(
                f"/robot_{robot}",
                show_axes=False,
                position=station.position,
                wxyz=matrix_to_quat(station.rotation),
            )
            self._urdfs[robot] = ViserUrdf(self.server, urdf, root_node_name=f"/robot_{robot}")
        self._joint_order = [
            planner.joint_names.index(n)
            for n in self._urdfs[self.robots[0]].get_actuated_joint_names()
        ]
        self._set_joints({r: planner.home for r in self.robots})

        scene = self.server.scene
        self._sparks = {
            r: scene.add_icosphere(f"/spark_{r}", radius=0.012, color=ARC_COLOR, visible=False)
            for r in self.robots
        }
        self._collision = scene.add_icosphere(
            "/collision", radius=0.06, color=(255, 0, 0), opacity=0.7, visible=False
        )
        self._spaces: Dict[str, object] = {}
        self._shown_wait: Dict[str, Optional[InterlockWait]] = {r: None for r in self.robots}
        self._held: Dict[str, object] = {}
        self._shown_reservation: Dict[str, Optional[Reservation]] = {r: None for r in self.robots}
        self._labels: Dict[str, Tuple[str, object]] = {}
        self._beads: Dict[str, object] = {}
        self._playback: Optional[_Playback] = None
        self._busy = threading.Lock()
        self._stop = threading.Event()
        gui = self.server.gui
        gui.add_markdown(
            f"**Multi-robot car body welding** ({len(self.robots)} robots)  \n"
            "Red cubes: space a waiting robot needs but another robot holds. "
            "Beads grow in the color of the robot that welds them. "
            "*Show reserved space* draws the space each robot holds as dots in its color."
        )
        self.plan_status = gui.add_markdown("Starting...")
        self.robot_status = gui.add_markdown("")
        self.progress = gui.add_progress_bar(0.0)
        with gui.add_folder("Interlock log"):
            self.event_log = gui.add_markdown("")
        self.stats = gui.add_markdown("")
        self.speed = gui.add_slider(
            "Playback speed", min=0.5, max=8.0, step=0.5, initial_value=2.0
        )
        self.show_held = gui.add_checkbox(
            "Show reserved space",
            False,
            hint="Space each robot holds through the interlock, including its own arm.",
        )
        self.show_space = gui.add_checkbox("Show blocked space", True)
        self.dynamic = gui.add_checkbox(
            "Assign seams at run time",
            dynamic,
            hint="Off: every robot welds its fixed program in order.",
        )
        self.replay_button = gui.add_button("Replay", color="blue")
        self.plan_button = gui.add_button("Plan with interlocks", color="green")
        self.unsafe_button = gui.add_button("Plan without interlocks", color="red")
        self.replay_button.on_click(lambda _: self._run(self._replay))
        self.plan_button.on_click(lambda _: self._run(self._plan_and_play, interlocks=True))
        self.unsafe_button.on_click(lambda _: self._run(self._plan_and_play, interlocks=False))

    def run(self) -> None:
        """Plan the cell, play it back, and serve the viewer until Ctrl+C."""
        self._run(self._plan_and_play, interlocks=True)
        print(f"\nWelding cell running at http://localhost:{self.server.get_port()}")
        print("Press Ctrl+C to exit.\n")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nShutting down...")

    # ---------------------------------------------------------------- actions

    def _run(self, fn: Callable[..., None], **kwargs) -> None:
        """Stop the current playback and run ``fn`` on a worker thread."""
        self._stop.set()

        def worker() -> None:
            with self._busy:
                self._stop.clear()
                fn(**kwargs)

        threading.Thread(target=worker, daemon=True).start()

    def _plan_and_play(self, interlocks: bool) -> None:
        mode = "with" if interlocks else "**without**"
        self.stats.content = ""
        self._reset_scene()

        def progress(message: str) -> None:
            self.plan_status.content = f"Planning {mode} interlocks: {message}"

        programs = None if self.dynamic.value else self.planner.cell.programs()
        scheduler = WeldScheduler(
            self.planner,
            list(self._seams.values()),
            self.robots,
            programs=programs,
            interlocks=interlocks,
        )
        plan = scheduler.plan(progress)
        profile = clearance_profile(self.planner, plan)
        check = check_plan(self.planner, plan)
        joints = {r: plan.joint_trajectory(r) for r in self.robots}
        events = _interlock_events(plan, profile)
        self._playback = _Playback(plan, check, interlocks, joints, profile, events)
        self.plan_status.content = f"**Planned {mode} interlocks** in {plan.planning_time:.1f} s"
        self.stats.content = self._format_stats(self._playback)
        self._replay()

    def _replay(self) -> None:
        """Play the program in real time, scaled by the speed slider."""
        pb = self._playback
        if pb is None:
            return
        self._reset_scene()
        plan = pb.plan
        welds = {r: [m for m in plan.motions[r] if m.kind == WELD] for r in self.robots}
        drawn: Dict[str, int] = {}
        tick = 0.0
        last = time.time()
        while not self._stop.is_set():
            k = min(int(tick), plan.num_ticks)
            self._set_joints({r: pb.joints[r][k] for r in self.robots})
            texts = [self._robot_state(robot, k, welds[robot], drawn) for robot in self.robots]
            self.robot_status.content = "  \n".join(texts)
            self._show_conflicts(pb, k)
            self._show_reserved(pb, k)
            log = [text for t, text in pb.events if t <= k][-EVENT_LOG_LINES:]
            self.event_log.content = "  \n".join(log) or "No interlock event yet."
            self.progress.value = 100.0 * k / max(plan.num_ticks, 1)
            if k == plan.num_ticks:
                self.plan_status.content += "  \n**Done.** Click *Replay* or plan again."
                return
            time.sleep(1.0 / 30.0)
            now = time.time()
            tick += (now - last) * self.speed.value / DT
            last = now

    # ---------------------------------------------------------------- drawing

    def _robot_state(
        self, robot: str, tick: int, welds: List[Motion], drawn: Dict[str, int]
    ) -> str:
        """Status line of a robot at ``tick``; also grows its weld beads and arc spark."""
        plan = self._playback.plan
        color = "#%02x%02x%02x" % ROBOT_COLORS[robot]
        name = f"<span style='color:{color}'>**{robot}**</span>"
        for motion in welds:
            if motion.start <= tick:
                index = min(tick - motion.start, len(motion.bead) - 1)
                self._draw_bead(robot, motion, float(motion.bead[index]), drawn)
        spark = self._sparks[robot]
        spark.visible = False
        for motion in plan.motions[robot]:
            if motion.start <= tick < motion.end:
                if motion.kind == WELD:
                    progress = float(motion.bead[tick - motion.start])
                    if 0 <= progress < 1:
                        position, _ = self.planner.tool_poses(
                            robot, motion.positions[tick - motion.start][None]
                        )
                        spark.position = tuple(position[0])
                        spark.visible = True
                        self._label(robot, f"{robot}: welding {motion.seam}")
                        return f"{name}: welding {motion.seam} ({progress * 100:.0f} %)"
                    self._label(robot, f"{robot}: {motion.seam}")
                    return (
                        f"{name}: approaching {motion.seam}"
                        if progress < 0
                        else f"{name}: leaving {motion.seam}"
                    )
                if motion.kind == TRANSFER:
                    self._label(robot, f"{robot}: to {motion.seam}")
                    return f"{name}: moving to {motion.seam}"
                self._label(robot, f"{robot}: going home")
                return f"{name}: returning home"
        for wait in plan.waits:
            if wait.robot == robot and wait.start <= tick < wait.end:
                target = wait.seam or "home"
                blockers = "+".join(wait.blocked_by)
                self._label(robot, f"{robot}: WAIT for {blockers}")
                return (
                    f"{name}: <span style='color:#d33'>**waiting**</span> for {blockers} "
                    f"(interlock, next: {target})"
                )
        done = sum(1 for m in welds if m.end <= tick)
        self._label(robot, f"{robot}: idle")
        return f"{name}: idle ({done} seams welded)"

    def _label(self, robot: str, text: str) -> None:
        """Show ``text`` above the robot; only sends an update when the text changes."""
        if robot in self._labels and self._labels[robot][0] == text:
            return
        station = self.planner.cell.station(robot)
        handle = self.server.scene.add_label(
            f"/labels/{robot}",
            text,
            position=(station.position[0], station.position[1], station.position[2] + 1.25),
            font_screen_scale=0.7,
            anchor="bottom-center",
        )
        self._labels[robot] = (text, handle)

    def _show_reserved(self, pb: _Playback, tick: int) -> None:
        """Space each robot holds at ``tick``, as dots in the robot's color."""
        for robot in self.robots:
            held = None
            if self.show_held.value:
                held = next(
                    (
                        r
                        for r in pb.plan.reservations
                        if r.robot == robot and r.start <= tick < max(r.end, r.start + 1)
                    ),
                    None,
                )
            if held is self._shown_reservation[robot]:
                continue
            self._shown_reservation[robot] = held
            if robot in self._held:
                self._held.pop(robot).remove()
            if held is not None and len(held.voxels):
                self._held[robot] = self.server.scene.add_point_cloud(
                    f"/reserved/{robot}",
                    held.voxels.astype(np.float32),
                    ROBOT_COLORS[robot],
                    point_size=self.planner.grid.voxel_size * 0.3,
                    point_shape="circle",
                )

    def _draw_bead(
        self, robot: str, motion: Motion, progress: float, drawn: Dict[str, int]
    ) -> None:
        if progress < 0:
            return
        first, last = motion.tool_path.seam_range
        points = motion.tool_path.positions[first : last + 1]
        count = int(round(progress * (len(points) - 1))) + 1
        if count < 2 or drawn.get(motion.seam, 0) == count:
            return
        drawn[motion.seam] = count
        segments = np.stack([points[: count - 1], points[1:count]], axis=1)
        self._beads[motion.seam] = self.server.scene.add_line_segments(
            f"/beads/{motion.seam}", segments, ROBOT_COLORS[robot], thickness=0.008
        )

    def _show_conflicts(self, pb: _Playback, tick: int) -> None:
        """Shared space of waiting robots, and the contact point where robots collide."""
        for robot in self.robots:
            wait = next(
                (w for w in pb.plan.waits if w.robot == robot and w.start <= tick < w.end), None
            )
            if not self.show_space.value or (wait is not None and len(wait.shared_space) == 0):
                wait = None
            if wait is self._shown_wait[robot]:
                continue
            self._shown_wait[robot] = wait
            if robot in self._spaces:
                self._spaces.pop(robot).remove()
            if wait is not None:
                self._spaces[robot] = self.server.scene.add_point_cloud(
                    f"/interlock/{robot}",
                    wait.shared_space.astype(np.float32),
                    CONFLICT_COLOR,
                    point_size=self.planner.grid.voxel_size * 0.8,
                    point_shape="square",
                )
        colliding = bool(pb.profile.clearance[tick] < 0.0)
        self._collision.visible = colliding
        if colliding:
            self._collision.position = tuple(pb.profile.contact[tick])
            overlap = -pb.profile.clearance[tick] * 1000
            a, b = pb.profile.pair[tick]
            warning = f"**COLLISION**: {a} and {b} overlap by {overlap:.0f} mm"
            self.robot_status.content += f"  \n<span style='color:#d00'>{warning}</span>"

    def _set_joints(self, joints: Dict[str, np.ndarray]) -> None:
        for robot, q in joints.items():
            self._urdfs[robot].update_cfg(np.asarray(q)[self._joint_order])

    def _reset_scene(self) -> None:
        handles = (
            list(self._beads.values()) + list(self._spaces.values()) + list(self._held.values())
        )
        for handle in handles:
            handle.remove()
        self._beads, self._spaces, self._held = {}, {}, {}
        self._shown_wait = {r: None for r in self.robots}
        self._shown_reservation = {r: None for r in self.robots}
        self.event_log.content = ""
        for spark in self._sparks.values():
            spark.visible = False
        self._collision.visible = False
        self._set_joints({r: self.planner.home for r in self.robots})

    def _draw_cell(self) -> None:
        for name, cuboid in self.planner.cell.obstacles().items():
            if name.startswith("pedestal"):
                color = (80, 80, 86)
            elif name == "skid":
                color = (120, 110, 100)
            else:
                color = BODY_COLOR
            self.server.scene.add_box(
                f"/cell/{name}",
                color=color,
                dimensions=tuple(cuboid["dims"]),
                position=tuple(cuboid["pose"][:3]),
                wxyz=tuple(cuboid["pose"][3:]),
            )
        for seam in self._seams.values():
            points = seam.resample(0.01)[0]
            segments = np.stack([points[:-1], points[1:]], axis=1)
            self.server.scene.add_line_segments(
                f"/seams/{seam.name}", segments, SEAM_COLOR, thickness=0.004
            )

    @staticmethod
    def _format_stats(pb: _Playback) -> str:
        plan, check = pb.plan, pb.check
        lines = [
            f"Interlocks: **{'on' if pb.interlocks else 'off'}**",
            f"Cycle time: **{plan.cycle_time:.1f} s**, seams welded: {len(plan.assignment)}"
            + (f", left over: {', '.join(plan.leftover)}" if plan.leftover else ""),
        ]
        for robot in plan.robots:
            busy = sum(len(m.positions) - 1 for m in plan.motions[robot] if m.kind == WELD) * DT
            color = "#%02x%02x%02x" % ROBOT_COLORS.get(robot, (0, 0, 0))
            lines.append(
                f"<span style='color:{color}'>**{robot}**</span>: "
                f"{len(plan.weld_order(robot))} seams, welding {busy:.0f} s, "
                f"waiting {plan.wait_ticks(robot) * DT:.1f} s"
            )
        lines.append(
            f"Interlock waits: {len(plan.waits)}, seams deferred: {len(plan.deferrals)}, "
            f"retreats: {plan.yields}, handed over: {len(plan.handovers)}"
        )
        color = "#080" if check.min_clearance > 0 else "#d00"
        lines.append(
            f"Min. robot-to-robot distance: "
            f"<span style='color:{color}'>**{check.min_clearance * 1000:.0f} mm**</span>"
        )
        lines.append(
            f"Torch on seam: within {check.max_seam_deviation * 1000:.2f} mm, axis within "
            f"{math.degrees(check.max_torch_angle_error):.1f} deg"
        )
        return "  \n".join(lines)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def summarize(plan: CellPlan, check: PlanCheck) -> str:
    """Multi-line text summary of a plan and its checks."""
    lines = [
        f"cycle time {plan.cycle_time:.1f} s, planning {plan.planning_time:.1f} s, "
        f"{len(plan.assignment)} seams welded, {len(plan.leftover)} left over "
        f"{plan.leftover or ''}",
    ]
    for robot in plan.robots:
        welds = plan.weld_order(robot)
        lines.append(
            f"  {robot:11s} {len(welds):2d} seams, waited {plan.wait_ticks(robot) * DT:5.1f} s: "
            f"{', '.join(welds)}"
        )
    for seam, owner, robot in plan.handovers:
        lines.append(f"  handover: {seam} ({owner}'s program) welded by {robot}")
    for d in plan.deferrals:
        lines.append(
            f"  interlock: {d.robot} welded {d.taken} first at {d.tick * DT:5.1f} s; "
            f"{d.wanted} was blocked by {'+'.join(d.blocked_by)}"
        )
    for wait in sorted(plan.waits, key=lambda w: w.start):
        lines.append(
            f"  interlock: {wait.robot} waited {wait.start * DT:5.1f}-{wait.end * DT:5.1f} s for "
            f"{'+'.join(wait.blocked_by)} (next: {wait.seam or 'home'})"
        )
    for tick, robot in plan.retreats:
        lines.append(f"  retreat: {robot} went home at {tick * DT:5.1f} s to clear a mutual block")
    for tick, a, b, depth in check.collisions:
        lines.append(
            f"  COLLISION: {a} and {b} at {tick * DT:5.1f} s ({depth * 1000:.0f} mm deep)"
        )
    lines.append(
        f"  min robot clearance {check.min_clearance * 1000:.0f} mm "
        f"at {check.min_clearance_time:.1f} s, "
        f"retreats {plan.yields}"
    )
    lines.append(
        f"  seam deviation max {check.max_seam_deviation * 1000:.2f} mm, torch angle error max "
        f"{math.degrees(check.max_torch_angle_error):.2f} deg, travel speed error mean "
        f"{check.speed_error[0] * 100:.1f} % / max {check.speed_error[1] * 100:.1f} %"
    )
    return "\n".join(lines)


def plan_cell(
    planner: CellPlanner,
    robots: Sequence[str] = ROBOTS,
    interlocks: bool = True,
    dynamic: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[CellPlan, PlanCheck]:
    """Plan and check the welding program of a cell.

    Args:
        planner: Cell planner.
        robots: Robots in the cell.
        interlocks: Keep robots apart with interlocks.
        dynamic: Assign seams at run time instead of using the cell's fixed programs.
        progress: Optional callback for status messages.
    """
    programs = None if dynamic else planner.cell.programs()
    scheduler = WeldScheduler(
        planner, planner.cell.seams(), robots, programs=programs, interlocks=interlocks
    )
    plan = scheduler.plan(progress)
    return plan, check_plan(planner, plan)


def test() -> None:
    """Plan the default cell and check the result."""
    planner = CellPlanner()
    plan, check = plan_cell(planner)
    print(summarize(plan, check))
    assert not plan.leftover, f"seams left over: {plan.leftover}"
    assert all(plan.weld_order(r) for r in plan.robots), "every robot should weld"
    assert check.min_clearance > 0.0, f"robots touch: {check.min_clearance:.4f} m"
    assert (
        check.max_seam_deviation < 0.003
    ), f"torch left the seam by {check.max_seam_deviation * 1000:.1f} mm"
    assert check.max_torch_angle_error < math.radians(2.0), "torch angle off"
    assert check.speed_error[1] < 0.05, f"travel speed off by {check.speed_error[1] * 100:.1f} %"
    planner.destroy()


def main() -> None:
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Multi-robot car body welding with cuRobo")
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=ROBOTS,
        default=list(ROBOTS),
        help="Robots in the cell (default: all three)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Assign seams at run time instead of fixed robot programs",
    )
    parser.add_argument(
        "--no-interlocks",
        action="store_true",
        help="Headless: plan without interlocks to see the collisions",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    parser.add_argument("--headless", action="store_true", help="Plan once without the viewer")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)
    planner = CellPlanner()
    if args.headless:
        plan, check = plan_cell(
            planner, args.robots, interlocks=not args.no_interlocks, dynamic=args.dynamic
        )
        print(summarize(plan, check))
        return
    WeldingViewer(planner, args.robots, args.port, dynamic=args.dynamic).run()


if __name__ == "__main__":
    main()
