# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Two robot arms empty a bin of randomly piled plates without colliding with each other.

Two arms stand on opposite sides of a work table. The table holds a *source* bin
with a random pile of thin plates and an empty *target* bin. The arms move every
plate from the source bin into neat stacks in the target bin. By default the arms
are KUKA LBR iisy3 R760 robots without a gripper (``dual_lbr_iisy3_r760.yml``): a
virtual tool point 0.10 m below the flange holds each plate, so plates hang below the
flange. ``--robot dual_franka.yml`` uses two Franka Panda arms with Franka hands.

The demo shows four things cuRobo does for coordinated multi-arm work:

- **Each arm works at its own pace.** A pick or a place has three moves: a free
  transit to 12 cm above the plate or stack, a straight-down move (grasp or
  release), and a straight-up move. An arm plans its next move as soon as it has
  finished the previous one. It does not wait for the other arm to reach a matching
  point in its cycle.
- **One robot model, one planner per arm.** The robot config models both arms as one
  robot with two tool frames, so self-collision checking keeps the arms (and the
  plates they carry) apart. Each arm has its own planner in which the other arm's
  joints are locked. Before each plan, the locked arm is set to the pose where it
  rests once its current move ends (``_lock_other_arm``).
- **Timing check against the other arm.** A planned move is then checked against
  the other arm's timed motion, which may still be under way, and starts later if
  the arms would come too close (``_earliest_start``). When the other arm is in the
  way even at rest, the arm waits for the other arm's next move; an arm without a
  plate waits at its home pose, out of the way. When both arms wait for each other,
  one of them yields: it gives up its plate or steps back to its home pose.
- **Task sequencing on top of motion planning.** Plates can lie on top of each
  other. The sequencer only offers plates whose covering plates have been lifted,
  prefers plates close to the arm, keeps the tools of the two arms apart, and falls
  back to another plate when a move cannot be planned.

cuRobo keeps the fastest of its candidate trajectories, not the shortest. For a 6-axis
arm that is often an elbow/wrist flip that swings the arm far around, so moves are
first planned from IK solutions next to the current configuration
(``_plan_near_branch``).

Run the demo
------------

From the repository root:

.. code-block:: bash

   python -m curobo.examples.reference.dual_arm_bin_picking

Open http://localhost:8080. The first scene is planned and played back
automatically. Click **New random scene** to sample a new pile. Click
**Replay** to watch the last plan again. By default the right arm starts once the
left arm has lifted its first plate. Use ``--schedule synchronous`` to start both
arms at the same time:

.. code-block:: bash

   python -m curobo.examples.reference.dual_arm_bin_picking --schedule synchronous --num-parts 16

Use ``--headless`` to plan without the viewer and print a summary. Use ``--test``
to check that a few random scenes are fully cleared.

Any dual-arm config works when it follows the same naming: arms prefixed ``left_``
(base at y = +0.5 m) and ``right_`` (base at y = -0.5 m), each with an
``attached_object`` link and a tool frame listed in ``END_EFFECTORS`` (``panda_hand``
for a Franka hand, ``tool0`` for an arm without a gripper).
"""

from __future__ import annotations

import argparse
import copy
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from curobo.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.trajectory_optimizer import TrajectoryOptimizerResult
from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria

#: Default robot: two KUKA LBR iisy3 R760 arms without grippers.
ROBOT_FILE = "dual_lbr_iisy3_r760.yml"
ARMS = ("left", "right")

#: Base position of each arm in the world frame. Must match the robot URDF.
ARM_BASE_XY = {"left": np.array([0.0, 0.5]), "right": np.array([0.0, -0.5])}

ATTACH_LINK = {arm: f"{arm}_attached_object" for arm in ARMS}


def _other(arm: str) -> str:
    """The other arm's name."""
    return ARMS[1 - ARMS.index(arm)]


@dataclass(frozen=True)
class EndEffector:
    """How an arm's tool frame holds a plate. Link and joint names omit the arm prefix."""

    #: Tool frame of the arm in the robot config.
    frame: str
    #: Height of the tool frame above the top face of a held plate (m).
    standoff: float
    #: Half extents of the tool footprint seen from above, along the tool x and y axes,
    #: including 2 cm clearance (m). Plates the two arms pick at the same time must not
    #: have overlapping footprints.
    footprint_half: Tuple[float, float]
    #: Top-down tool yaws, relative to the plate yaw, that hold a plate equally well.
    grasp_yaws: Tuple[float, ...]
    #: Finger links. Their collision is off during straight grasp and release moves.
    finger_links: Tuple[str, ...] = ()
    #: Finger joints, animated when the gripper opens and closes.
    finger_joints: Tuple[str, ...] = ()
    #: Finger joint value of an open gripper (m).
    finger_open: float = 0.0


#: Franka hand. The fingertips are 0.112 m below the hand frame and stop 2 mm above the
#: bottom of the 8 mm plate. The fingers close across the plate width, so the hand can
#: only grasp at the plate yaw or turned by pi.
PANDA_HAND = EndEffector(
    frame="panda_hand",
    standoff=0.112 + 0.002 - 0.008,
    footprint_half=(0.052, 0.124),
    grasp_yaws=(0.0, math.pi),
    finger_links=("panda_leftfinger", "panda_rightfinger"),
    finger_joints=("panda_finger_joint1", "panda_finger_joint2"),
    finger_open=0.04,
)
#: Arm without a gripper. A virtual tool point 0.10 m below the flange (``tool0``) holds
#: the plate; there is no tool geometry, so the plate hangs below the flange. The wrist
#: then stays above the bin walls, as it would with a real gripper of similar length.
#: Any of four tool yaws holds the plate, so the wrist can turn its cable lug away
#: from the other arm.
VIRTUAL_TOOL = EndEffector(
    frame="tool0",
    standoff=0.10,
    footprint_half=(0.075, 0.075),
    grasp_yaws=(0.0, math.pi / 2, math.pi, -math.pi / 2),
)
END_EFFECTORS = {e.frame: e for e in (PANDA_HAND, VIRTUAL_TOOL)}

#: Height of the pre-grasp / pre-place pose above the grasp / place pose (m).
APPROACH_HEIGHT = 0.12
#: IK solutions whose summed joint distance from the current configuration is within
#: this margin (rad) of the nearest one seed the first planning attempt.
NEAR_BRANCH_MARGIN = 1.0
#: Noise (rad) added to the current configuration to form the IK seeds of that attempt.
NEAR_BRANCH_SEED_NOISE = 0.1
#: A motion is a detour when a tool travels this much farther than a straight line (m).
#: Normal moves stay below about 6 cm; elbow/wrist flips add about 2 m.
MAX_TOOL_DETOUR = 0.15
#: Planning passes as ``(thorough, strict)`` (see ``DualArmBinPicking._plan_goalset``).
#: Near-branch planning without detours:
QUICK_PASS = ((False, True),)
#: Adds cuRobo's standard planning, still without detours:
THOROUGH_PASS = ((True, True),)
#: Accepts detours, as a last resort:
ANY_PASS = ((True, False),)
#: Smallest gap between the two arms, including carried plates, when a move is checked
#: against the other arm's timed motion (m). The planner itself keeps 2 cm (a 1 cm
#: sphere buffer on each arm) from the other arm at its resting pose.
MIN_ARM_GAP = 0.01
#: Step between the start delays tried when a move would come too close to the other
#: arm's motion (s).
DELAY_STEP = 0.1
#: Duration of a gripper open / close (or suction on / off) pause (s).
GRIPPER_TIME = 0.3
#: Upper bound on planning iterations, to stop instead of looping forever.
MAX_PLANNING_STEPS = 5000

#: Move types: a free transit and straight vertical moves.
TRANSIT, DOWN, UP = "transit", "down", "up"

#: One goal option of an arm: tool position ``(3,)`` and orientation (wxyz).
GoalOption = Tuple[np.ndarray, List[float]]


# --------------------------------------------------------------------------------------
# Work cell and parts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellLayout:
    """Geometry of the work cell. The table top is at ``z = 0``. Units are meters."""

    #: Table top size along x and y.
    table_size: Tuple[float, float] = (0.70, 0.66)
    #: Height of the table and robot pedestals below the table top.
    table_height: float = 0.75
    #: Footprint of each robot pedestal.
    pedestal_size: Tuple[float, float] = (0.24, 0.24)
    #: Inner size of each bin along x and y.
    bin_inner_size: Tuple[float, float] = (0.24, 0.28)
    #: Wall height above the bin floor. Low enough for the wrist to clear it at grasp height.
    bin_wall_height: float = 0.045
    #: Thickness of bin walls and floor.
    bin_wall_thickness: float = 0.01
    #: Center (x, y) of the source bin.
    source_bin_xy: Tuple[float, float] = (-0.16, 0.0)
    #: Center (x, y) of the target bin.
    target_bin_xy: Tuple[float, float] = (0.16, 0.0)
    #: Plate length, width, and thickness. Grippers close across the width.
    part_size: Tuple[float, float, float] = (0.07, 0.04, 0.008)
    #: Stack centers in the target bin, relative to the bin center. Each arm fills the
    #: stacks on its own side (left: y > 0, right: y < 0). The two sides are 0.18 m
    #: apart so both wrists fit above the bin when the arms place at the same time,
    #: and a held plate's collision spheres stay 6 mm clear of the bin wall.
    stack_offsets: Tuple[Tuple[float, float], ...] = ((-0.05, 0.09), (0.05, 0.09))
    #: Plate yaw in the target stacks. At pi/2 the Franka fingers spread along x, so
    #: both hands fit into the target bin at the same time.
    stack_yaw: float = math.pi / 2

    @property
    def bin_floor_z(self) -> float:
        """Height of the bin floor surface."""
        return self.bin_wall_thickness

    def stack_xy(self, arm: str) -> List[np.ndarray]:
        """Stack centers in world (x, y) used by ``arm``, nearest first."""
        sign = 1.0 if arm == "left" else -1.0
        offsets = sorted(self.stack_offsets, key=lambda o: o[1])
        return [np.array(self.target_bin_xy) + np.array([dx, sign * dy]) for dx, dy in offsets]

    def scene_cfg(self) -> Dict[str, Dict[str, Dict]]:
        """Collision scene with the table, pedestals, and both bins as cuboids."""
        h = self.table_height
        cuboids = {
            "table": {
                "dims": [self.table_size[0], self.table_size[1], h],
                "pose": [0.0, 0.0, -h / 2, 1, 0, 0, 0],
            },
        }
        for arm in ARMS:
            x, y = ARM_BASE_XY[arm]
            cuboids[f"{arm}_pedestal"] = {
                "dims": [self.pedestal_size[0], self.pedestal_size[1], h],
                "pose": [float(x), float(y), -h / 2, 1, 0, 0, 0],
            }
        for name, (cx, cy) in (("source", self.source_bin_xy), ("target", self.target_bin_xy)):
            cuboids.update(self._bin_cuboids(f"{name}_bin", cx, cy))
        return {"cuboid": cuboids}

    def _bin_cuboids(self, name: str, cx: float, cy: float) -> Dict[str, Dict]:
        t = self.bin_wall_thickness
        ix, iy = self.bin_inner_size
        ox, oy = ix + 2 * t, iy + 2 * t
        wall_h = self.bin_floor_z + self.bin_wall_height
        wall_z = wall_h / 2
        return {
            f"{name}_floor": {"dims": [ox, oy, t], "pose": [cx, cy, t / 2, 1, 0, 0, 0]},
            f"{name}_wall_xn": {
                "dims": [t, oy, wall_h],
                "pose": [cx - (ix + t) / 2, cy, wall_z, 1, 0, 0, 0],
            },
            f"{name}_wall_xp": {
                "dims": [t, oy, wall_h],
                "pose": [cx + (ix + t) / 2, cy, wall_z, 1, 0, 0, 0],
            },
            f"{name}_wall_yn": {
                "dims": [ix, t, wall_h],
                "pose": [cx, cy - (iy + t) / 2, wall_z, 1, 0, 0, 0],
            },
            f"{name}_wall_yp": {
                "dims": [ix, t, wall_h],
                "pose": [cx, cy + (iy + t) / 2, wall_z, 1, 0, 0, 0],
            },
        }


@dataclass
class Part:
    """A plate lying flat in the source bin."""

    #: Index of the plate in the generated pile.
    index: int
    #: Center position in the world frame, shape ``(3,)``.
    position: np.ndarray
    #: Rotation about the world z-axis (rad).
    yaw: float
    #: Indices of plates that lie on top of this plate and must be removed first.
    blockers: Set[int] = field(default_factory=set)


def _rectangle_corners(center: np.ndarray, yaw: float, half: np.ndarray) -> np.ndarray:
    """Corners ``(4, 2)`` of a rectangle with half extents ``half`` rotated by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    return center + (signs * half) @ rot.T


def _rectangles_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quads given as corners ``(4, 2)``."""
    for quad in (a, b):
        for i in range(4):
            edge = quad[(i + 1) % 4] - quad[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def sample_parts(
    layout: CellLayout, num_parts: int, rng: np.random.Generator, max_layers: int = 3
) -> List[Part]:
    """Drop plates at random poses into the source bin.

    A plate that overlaps plates already in the bin rests on the highest of them, so
    the result is a pile with up to ``max_layers`` layers. Each plate records which
    plates lie on top of it.

    Args:
        layout: Work cell geometry.
        num_parts: Number of plates to drop.
        rng: Random number generator.
        max_layers: Maximum pile height in plates.

    Returns:
        Plates in drop order. Later plates are never below earlier plates.
    """
    length, width, thick = layout.part_size
    half = np.array([length, width]) / 2
    inner_half = np.array(layout.bin_inner_size) / 2 - 0.01
    center = np.array(layout.source_bin_xy)
    parts: List[Part] = []
    corners: List[np.ndarray] = []
    for _ in range(num_parts):
        for _attempt in range(1000):
            yaw = rng.uniform(-math.pi, math.pi)
            xy = center + rng.uniform(-inner_half, inner_half)
            quad = _rectangle_corners(xy, yaw, half)
            if np.any(np.abs(quad - center) > inner_half):
                continue
            below = [p for p, q in zip(parts, corners) if _rectangles_overlap(quad, q)]
            layer = max(
                (round((p.position[2] - layout.bin_floor_z) / thick + 0.5) for p in below),
                default=0,
            )
            if layer >= max_layers:
                continue
            part = Part(
                len(parts),
                np.array([xy[0], xy[1], layout.bin_floor_z + (layer + 0.5) * thick]),
                yaw,
            )
            for p in below:
                p.blockers.add(part.index)
            parts.append(part)
            corners.append(quad)
            break
    if len(parts) < num_parts:
        print(
            f"Only {len(parts)} of {num_parts} plates fit into the bin ({max_layers} layers max)."
        )
    return parts


def top_down_quaternion(yaw: float) -> List[float]:
    """Tool orientation (wxyz) pointing straight down, tool x-axis along ``yaw``.

    This is ``Rz(yaw) * Rx(pi)``: the tool z-axis points down. At the plate yaw the tool
    x-axis is along the plate length, so Franka fingers close across the plate width.
    """
    return [0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]


def _path_detour(positions: torch.Tensor) -> float:
    """Path length of points ``(T, 3)`` minus the straight distance from first to last (m)."""
    path = torch.linalg.norm(positions[1:] - positions[:-1], dim=-1).sum()
    return float(path - torch.linalg.norm(positions[-1] - positions[0]))


def tool_footprint(position: np.ndarray, yaw: float, half: Tuple[float, float]) -> np.ndarray:
    """Corners ``(4, 2)`` of a tool footprint with half extents ``half`` above a plate."""
    return _rectangle_corners(np.asarray(position)[:2], yaw, np.asarray(half))


# --------------------------------------------------------------------------------------
# Task sequencing
# --------------------------------------------------------------------------------------


class PickSequencer:
    """Decides which plate each arm picks next.

    A plate is offered only when every plate on top of it has been lifted. Offered
    plates are ranked by distance to the arm base, with a bonus for plates that
    uncover others. A plate an arm failed to reach ``max_failures`` times is no longer
    offered to that arm but stays available to the other arm.
    """

    def __init__(
        self,
        parts: Sequence[Part],
        max_failures: int = 2,
        footprint_half: Tuple[float, float] = VIRTUAL_TOOL.footprint_half,
    ):
        """Track the pile.

        Args:
            parts: Plates in the source bin.
            max_failures: Failed attempts after which an arm stops trying a plate.
            footprint_half: Tool footprint half extents, see ``EndEffector.footprint_half``.
        """
        self._footprint_half = footprint_half
        self._parts = {p.index: p for p in parts}
        self._remaining: Set[int] = set(self._parts)
        self._lifted_at: Dict[int, float] = {}
        self._failures: Dict[str, Dict[int, int]] = {arm: {} for arm in ARMS}
        self._rejected: Dict[str, Set[int]] = {arm: set() for arm in ARMS}
        self._max_failures = max_failures
        self._uncovers = {i: sum(i in p.blockers for p in parts) for i in self._parts}

    @property
    def remaining(self) -> Set[int]:
        """Plates still in the source bin and not assigned to an arm."""
        return set(self._remaining)

    def has_work(self, arm: str) -> bool:
        """True while some remaining plate could still be picked by ``arm``."""
        return bool(self._remaining - self._rejected[arm])

    def candidates(
        self, arm: str, time: float, keep_away_from: Optional[np.ndarray] = None
    ) -> List[int]:
        """Plates ``arm`` may pick when it starts moving to them at ``time``, best first.

        Args:
            arm: Arm name.
            time: Time at which the arm starts moving to the plate (s).
            keep_away_from: Tool footprint corners ``(4, 2)`` of the other arm's pick.
                Plates whose tool footprint overlaps it are skipped.

        Returns:
            Plate indices ordered by preference.
        """
        options = []
        base = ARM_BASE_XY[arm]
        for i in self._remaining - self._rejected[arm]:
            part = self._parts[i]
            if any(self._lifted_at.get(b, math.inf) > time for b in part.blockers):
                continue
            if keep_away_from is not None and _rectangles_overlap(
                tool_footprint(part.position, part.yaw, self._footprint_half), keep_away_from
            ):
                continue
            # Prefer near plates on the arm's own half of the bin (fewer path crossings)
            # and plates that uncover others (more options later).
            other_side = part.position[1] * base[1] < 0.0
            score = (
                np.linalg.norm(part.position[:2] - base)
                + 0.1 * other_side
                - 0.05 * self._uncovers[i]
            )
            options.append((score, i))
        return [i for _, i in sorted(options)]

    def commit(self, part_index: int) -> None:
        """Assign a plate to an arm; it no longer counts as remaining."""
        self._remaining.discard(part_index)

    def mark_lifted(self, part_index: int, time: float) -> None:
        """Record that an assigned plate has been lifted clear of the pile at ``time`` (s)."""
        self._lifted_at[part_index] = time

    def release(self, part_index: int) -> None:
        """Return an assigned plate to the pile (its pick was cancelled before grasping)."""
        self._lifted_at.pop(part_index, None)
        self._remaining.add(part_index)

    def record_failure(self, arm: str, part_index: int) -> None:
        """Count a failed attempt of ``arm`` to plan a motion to a plate."""
        count = self._failures[arm].get(part_index, 0) + 1
        self._failures[arm][part_index] = count
        if count >= self._max_failures:
            self._rejected[arm].add(part_index)


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


@dataclass
class Waypoint:
    """One goal of an arm."""

    #: Move type: ``TRANSIT`` (free tool path), ``DOWN`` or ``UP`` (straight vertical path).
    kind: str
    #: Target tool position, shape ``(3,)``.
    position: np.ndarray
    #: Allowed tool orientations (wxyz). Several entries form a goal set; ``None``
    #: keeps the orientation chosen in the previous transit.
    orientations: Optional[List[List[float]]] = None
    #: Yaw of the plate handled in this pick or place, used for the tool footprint.
    yaw: Optional[float] = None
    #: What happens when the move ends: ``"grasp"``, ``"release"``, ``"lift"`` (the
    #: plate is clear of the pile), or ``None``.
    event: Optional[str] = None
    #: Plate of ``event``.
    part: Optional[int] = None
    #: Short description shown in the viewer.
    label: str = ""


@dataclass
class Segment:
    """A piece of the final timeline to play back."""

    #: Short description shown in the viewer.
    label: str
    #: Full joint positions including fingers, shape ``(T, num_joints)``.
    positions: np.ndarray
    #: Time between rows of ``positions`` (s).
    dt: float
    #: Plate poses ``part -> (positions (T, 3), wxyz (T, 4))`` of plates that move.
    part_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: Arm carrying each plate in ``part_poses``.
    carried_by: Dict[int, str] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Result of planning the full bin-emptying task."""

    #: Timeline to play back.
    segments: List[Segment]
    #: Joint names of ``Segment.positions`` columns.
    joint_names: List[str]
    #: Plate indices in pick order for each arm.
    pick_order: Dict[str, List[int]]
    #: Plates left in the source bin.
    leftover: List[int]
    #: Time at which each moved plate leaves the pile (s).
    pick_times: Dict[int, float]
    #: Number of planned arm moves.
    num_motions: int
    #: Wall-clock planning time (s).
    planning_time: float
    #: Duration of the planned motion (s).
    execution_time: float
    #: Smallest distance between the two arms, including carried plates, over the
    #: whole timeline (m), measured on the collision spheres.
    min_arm_clearance: float
    #: Largest extra distance a tool travels in one move compared to a straight
    #: line (m). Large values mean an arm swung far around.
    max_tool_detour: float


@dataclass
class _ArmState:
    """Book-keeping of one arm at the end of its planned timeline."""

    queue: List[Waypoint] = field(default_factory=list)
    orientation: Optional[List[float]] = None
    held_part: Optional[int] = None
    tool_to_part: Optional[Pose] = None
    placed: int = 0
    finished: bool = False
    finger: float = 0.0
    #: Number of moves planned so far.
    num_motions: int = 0
    #: The other arm's ``num_motions`` when this arm got stuck. The arm tries again once
    #: the other arm has made another move. ``None`` while the arm is not waiting.
    waiting_for: Optional[int] = None
    #: The arm starts once the other arm has lifted its first plate.
    starts_after_first_lift: bool = False

    @property
    def done(self) -> bool:
        """True once the arm has no work left and has returned home."""
        return self.finished and not self.queue


@dataclass
class _Track:
    """Timeline of one arm on the shared clock: one row per interpolation step."""

    #: Arm joint positions in the joint order of the arm's planner, shape ``(T, dof)``.
    q: np.ndarray
    #: Finger joint value of each row, shape ``(T,)``.
    finger: np.ndarray
    #: Plate held in each row (-1 for none), shape ``(T,)``.
    held: np.ndarray
    #: Moves and pauses as ``(first row, end row, label)``.
    spans: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def now(self) -> int:
        """Row of the arm's last planned state."""
        return len(self.q) - 1

    def extend(self, q: np.ndarray, finger: np.ndarray, held: int, label: str) -> None:
        """Append rows ``q (N, dof)`` with finger values ``(N,)`` and the held plate."""
        start = len(self.q)
        self.q = np.concatenate([self.q, q])
        self.finger = np.concatenate([self.finger, finger])
        self.held = np.concatenate([self.held, np.full(len(q), held)])
        self.spans.append((start, len(self.q), label))

    def rows(self, start: int, count: int) -> np.ndarray:
        """Indices of ``count`` rows from ``start``; the arm holds its last state after its end."""
        return np.minimum(np.arange(start, start + count), self.now)


@dataclass
class _Carry:
    """When an arm carries a plate."""

    arm: str
    #: First row in which the arm holds the plate.
    start: int
    #: Plate pose in the tool frame.
    tool_to_part: Pose
    #: First row after the release, ``None`` while carried.
    end: Optional[int] = None


@dataclass
class _Motion:
    """A planned move of one arm."""

    #: Arm joint positions at the interpolation steps, shape ``(T, dof)``. Row 0 is the
    #: start state.
    q: torch.Tensor
    #: Tool orientation (wxyz) reached, taken from the goal set.
    orientation: List[float]
    #: Extra tool path length compared to a straight line (m).
    detour: float


def _joint_links(robot_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """``(parent link, child link)`` of every active joint of each arm."""
    config = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_file)).config.kinematics_config
    link_names = {index: name for name, index in config.link_name_to_idx_map.items()}
    pairs: Dict[str, List[Tuple[str, str]]] = {arm: [] for arm in ARMS}
    for link, joint in enumerate(config.joint_map.tolist()):
        if joint >= 0:
            arm = config.joint_names[joint].split("_")[0]
            pairs[arm].append((link_names[int(config.link_map[link])], link_names[link]))
    return pairs


class DualArmBinPicking:
    """Plans a full bin-emptying task for two arms that work at their own pace."""

    def __init__(
        self,
        layout: CellLayout = CellLayout(),
        schedule: str = "staggered",
        robot_file: str = ROBOT_FILE,
    ):
        """Create and warm up one motion planner per arm.

        Args:
            layout: Work cell geometry.
            schedule: ``"staggered"`` (the right arm starts once the left arm has
                lifted its first plate) or ``"synchronous"`` (both arms start together).
            robot_file: Dual-arm robot config (see the module docstring for the
                naming it must follow).
        """
        if schedule not in ("staggered", "synchronous"):
            raise ValueError(f"Unknown schedule: {schedule}")
        self.layout = layout
        self.schedule = schedule
        self.robot_file = robot_file
        robot = load_yaml(ContentPath(robot_config_file=robot_file).get_robot_configuration_path())
        kinematics = robot["robot_cfg"]["kinematics"]
        frame = kinematics["tool_frames"][0].removeprefix(f"{ARMS[0]}_")
        if frame not in END_EFFECTORS:
            raise ValueError(f"No end effector known for tool frame {frame!r} in {robot_file}")
        #: How the arms hold plates, chosen from the robot's tool frame.
        self.tool = END_EFFECTORS[frame]
        self._tool_frame = {arm: f"{arm}_{frame}" for arm in ARMS}

        # Forward kinematics of both arms. It also reports the links on both sides of
        # every arm joint, to lock an arm at a new pose in the other arm's planner.
        joint_links = _joint_links(robot_file)
        link_frames = sorted(
            {link for pairs in joint_links.values() for pair in pairs for link in pair}
        )
        self.fk = Kinematics(
            KinematicsCfg.from_robot_yaml_file(
                robot_file, tool_frames=list(self._tool_frame.values()) + link_frames
            )
        )

        cspace = kinematics["cspace"]
        default = dict(zip(cspace["joint_names"], cspace["default_joint_position"]))
        #: One motion planner per arm. It moves only that arm; the other arm's joints are
        #: locked and set before every plan (see ``_lock_other_arm``).
        self.planners: Dict[str, MotionPlanner] = {}
        for arm in ARMS:
            config = copy.deepcopy(robot)
            kin = config["robot_cfg"]["kinematics"]
            kin["tool_frames"] = [self._tool_frame[arm]]
            prefix = f"{_other(arm)}_"
            kin["lock_joints"] = {
                **(kin.get("lock_joints") or {}),
                **{name: value for name, value in default.items() if name.startswith(prefix)},
            }
            planner = MotionPlanner(
                MotionPlannerCfg.create(
                    robot=config,
                    scene_model=layout.scene_cfg(),
                    max_goalset=len(self.tool.grasp_yaws),
                )
            )
            print(f"Warming up the {arm} arm planner (first run compiles CUDA kernels)...")
            planner.warmup(enable_graph=True, num_warmup_iterations=3)
            self.planners[arm] = planner
        #: Time between rows of the timeline (s).
        self.dt = float(self.planners[ARMS[0]].trajopt_solver.config.interpolation_dt)
        self._device = self.planners[ARMS[0]].default_joint_state.position.device

        fk_joints = list(self.fk.joint_names)
        self._columns = {
            arm: [fk_joints.index(n) for n in self.planners[arm].joint_names] for arm in ARMS
        }
        self._home_q = {
            arm: self.planners[arm].default_joint_state.position.view(-1).cpu().numpy()
            for arm in ARMS
        }
        home = self._active_state({})
        #: Names of all joints including fingers, in the column order of ``Segment.positions``.
        self.full_joint_names = list(self.fk.get_full_js(home).joint_names)
        self._home_tool = {
            arm: (pose.position[0].cpu().numpy(), pose.quaternion[0].tolist())
            for arm, pose in self._tool_poses_batch(home).items()
        }
        self._free_motion = {arm: {self._tool_frame[arm]: ToolPoseCriteria()} for arm in ARMS}
        self._linear_motion = {
            arm: {self._tool_frame[arm]: ToolPoseCriteria.linear_motion(axis="z")} for arm in ARMS
        }

        # Links whose fixed transforms hold the locked arm in each planner.
        self._locked = {}
        for arm in ARMS:
            pairs = joint_links[_other(arm)]
            link_index = self.planners[
                arm
            ].kinematics.config.kinematics_config.link_name_to_idx_map
            rows = torch.tensor([link_index[child] for _, child in pairs], device=self._device)
            self._locked[arm] = ([p for p, _ in pairs], [c for _, c in pairs], rows)

        # Collision spheres of each arm in ``self.fk``, without the attachment slots.
        kin_cfg = self.fk.config.kinematics_config
        sphere_link = kin_cfg.link_sphere_idx_map.long()
        radius = self.fk.compute_kinematics(home).robot_spheres.view(-1, 4)[:, 3]
        self._arm_spheres_index = {}
        for arm in ARMS:
            links = torch.tensor(
                [
                    i
                    for name, i in kin_cfg.link_name_to_idx_map.items()
                    if name.startswith(f"{arm}_") and name != ATTACH_LINK[arm]
                ],
                device=self._device,
            )
            self._arm_spheres_index[arm] = torch.nonzero(
                torch.isin(sphere_link, links) & (radius > 0)
            ).view(-1)

        part_len, _, part_thick = layout.part_size
        sphere_radius = 0.018
        # Raise the spheres so their bottom stays 2 mm above the plate bottom. The held
        # plate then stays collision-active even while it touches the floor or a stack.
        lift = sphere_radius - part_thick / 2 + 0.002
        offsets = np.linspace(-0.37, 0.37, 4) * part_len
        #: Collision spheres of a held plate in the plate frame, shape ``(4, 4)``.
        self._part_spheres = torch.tensor(
            [[x, 0.0, lift, sphere_radius] for x in offsets],
            device=self._device,
            dtype=torch.float32,
        )

    # ---------------------------------------------------------------- public

    def plan(
        self,
        parts: Sequence[Part],
        progress: Optional[Callable[[str], None]] = None,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> TaskPlan:
        """Plan pick-and-place motions until the source bin is empty.

        The arm whose timeline ends earliest plans its next move. A move starts when
        the previous move of the same arm ends, or later if the other arm is in the
        way.

        Args:
            parts: Plates in the source bin.
            progress: Optional callback ``progress(message)`` for status updates.
            on_segment: Optional callback that receives each timeline segment as soon
                as both arms' motion up to its end is final, so playback can start
                before planning finishes. Exceptions raised by the callback abort
                planning.

        Returns:
            The planned timeline and statistics.
        """
        start_time = time.time()
        self._parts = {p.index: p for p in parts}
        self._sequencer = PickSequencer(parts, footprint_half=self.tool.footprint_half)
        self._arms = {arm: _ArmState(finger=self.tool.finger_open) for arm in ARMS}
        self._tracks = {
            arm: _Track(
                q=self._home_q[arm][None].copy(),
                finger=np.array([self.tool.finger_open]),
                held=np.array([-1]),
            )
            for arm in ARMS
        }
        self._carries: Dict[int, _Carry] = {}
        self._pile_pose = {
            p.index: (p.position.copy(), self._part_quaternion(p.yaw)) for p in parts
        }
        self._part_pose = dict(self._pile_pose)
        self._pick_order: Dict[str, List[int]] = {arm: [] for arm in ARMS}
        self._first_lift_row: Optional[int] = None
        self._max_detour = 0.0
        self._lock_cache: Dict[str, Tuple] = {}
        self._reset_attachments()
        if self.schedule == "staggered":
            self._arms["right"].starts_after_first_lift = True
        segments: List[Segment] = []
        emitted = 0

        def emit(until: int) -> None:
            nonlocal emitted
            for segment in self._segments(emitted, until):
                segments.append(segment)
                if on_segment is not None:
                    on_segment(segment)
            emitted = max(emitted, until)

        for _ in range(MAX_PLANNING_STEPS):
            active = [arm for arm in ARMS if not self._arms[arm].done]
            if not active:
                break
            ready = [arm for arm in active if not self._waiting(arm)]
            if ready:
                self._advance(min(ready, key=lambda a: (self._tracks[a].now, ARMS.index(a))))
            else:
                self._resolve_deadlock()
            # Rows before the end of every unfinished arm's timeline are final.
            unfinished = [len(self._tracks[arm].q) for arm in ARMS if not self._arms[arm].done]
            if unfinished:
                emit(min(unfinished))
            if progress is not None:
                done = len(parts) - len(self._sequencer.remaining)
                planned = min(t.now for t in self._tracks.values()) * self.dt
                progress(
                    f"Planning: {done}/{len(parts)} plates assigned, "
                    f"{planned:.0f} s of motion planned"
                )
        else:
            raise RuntimeError("Planning did not finish.")
        end = max(len(t.q) for t in self._tracks.values())
        emit(end)

        return TaskPlan(
            segments=segments,
            joint_names=self.full_joint_names,
            pick_order=self._pick_order,
            leftover=sorted(self._sequencer.remaining),
            pick_times={part: carry.start * self.dt for part, carry in self._carries.items()},
            num_motions=sum(a.num_motions for a in self._arms.values()),
            planning_time=time.time() - start_time,
            execution_time=(end - 1) * self.dt,
            min_arm_clearance=self.min_arm_clearance(segments, self.full_joint_names),
            max_tool_detour=self._max_detour,
        )

    def min_arm_clearance(self, segments: Sequence[Segment], joint_names: List[str]) -> float:
        """Smallest gap between the two arms' collision spheres over a timeline.

        Carried plates are included as spheres attached to the arm that holds them.
        This is an independent check of the planned motion: a negative value means
        the arms (or plates) would touch.

        Args:
            segments: Timeline to check.
            joint_names: Joint names of ``Segment.positions`` columns.

        Returns:
            Minimum clearance in meters.
        """
        part_local = self._part_spheres
        best = float("inf")
        for seg in segments:
            q = torch.as_tensor(seg.positions, device=self._device, dtype=torch.float32)
            active = self.fk.get_active_js(JointState.from_position(q, joint_names=joint_names))
            spheres = self.fk.compute_kinematics(active).robot_spheres.view(q.shape[0], -1, 4)
            side = {arm: [spheres[:, self._arm_spheres_index[arm]]] for arm in ARMS}
            for index, arm in seg.carried_by.items():
                pos, quat = seg.part_poses[index]
                pose = Pose(
                    position=torch.as_tensor(pos, device=q.device, dtype=torch.float32),
                    quaternion=torch.as_tensor(quat, device=q.device, dtype=torch.float32),
                )
                centers = pose.batch_transform_points(
                    part_local[:, :3].unsqueeze(0).repeat(q.shape[0], 1, 1).contiguous()
                )
                radii = part_local[:, 3].view(1, -1, 1).repeat(q.shape[0], 1, 1)
                side[arm].append(torch.cat([centers, radii], dim=-1))
            left = torch.cat(side["left"], dim=1)
            right = torch.cat(side["right"], dim=1)
            best = min(best, self._gap(left, right))
        return best

    # ---------------------------------------------------------------- task flow

    def _waiting(self, arm: str) -> bool:
        """True while ``arm`` waits for the other arm to move on."""
        a, other = self._arms[arm], self._arms[_other(arm)]
        if a.starts_after_first_lift and self._first_lift_row is None and not other.done:
            return True
        if a.waiting_for is None:
            return False
        if other.done or other.num_motions != a.waiting_for:
            a.waiting_for = None
            return False
        return True

    def _advance(self, arm: str) -> None:
        """Plan the arm's next move from the end of its timeline, or let it wait."""
        a = self._arms[arm]
        if a.starts_after_first_lift:
            a.starts_after_first_lift = False
            if self._first_lift_row is not None:
                self._hold(arm, self._first_lift_row - self._tracks[arm].now, "wait to start")
            return
        if not a.queue:
            self._assign(arm)
            return
        wp = a.queue[0]
        other_done = self._arms[_other(arm)].done
        motion = self._plan_waypoint(
            arm, wp, QUICK_PASS + THOROUGH_PASS if other_done else QUICK_PASS
        )
        if motion is not None and self._execute(arm, motion):
            return
        if not other_done:
            self._wait(arm)
            return
        # The other arm has finished, so waiting does not help any more.
        if self._pick_pending(arm):
            self._cancel_pick(arm)
            return
        motion = self._plan_waypoint(arm, wp, ANY_PASS)
        if motion is None or not self._execute(arm, motion):
            raise RuntimeError(f"Could not plan the {arm} arm's move: {wp.label}.")

    def _assign(self, arm: str) -> None:
        """Give an idle arm its next plate and execute the transit above it.

        Tries candidate plates in order of preference. While the other arm still
        moves, only motions that keep the arm in its configuration are accepted; if
        none works, the arm waits for the other arm's next move. Once the other arm is
        done, failures count against the plate.
        """
        a, sequencer = self._arms[arm], self._sequencer
        if not sequencer.has_work(arm):
            self._send_home(arm, finished=True)
            return
        other_done = self._arms[_other(arm)].done
        candidates = sequencer.candidates(
            arm, self._tracks[arm].now * self.dt, keep_away_from=self._pick_footprint(_other(arm))
        )[:4]
        passes = QUICK_PASS + THOROUGH_PASS if other_done else QUICK_PASS
        for index in candidates:
            a.queue = self._pick_place_waypoints(arm, a, self._parts[index])
            motion = self._plan_waypoint(arm, a.queue[0], passes)
            if (
                motion is not None
                and self._descent_feasible(arm, motion)
                and self._execute(arm, motion)
            ):
                sequencer.commit(index)
                self._pick_order[arm].append(index)
                a.placed += 1
                return
            a.queue = []
            if other_done:
                sequencer.record_failure(arm, index)
        if not other_done:
            # Wait out of the other arm's way: an arm resting above a bin often blocks
            # the other arm and its own way to the next plate.
            if self._at_home(arm):
                self._wait(arm)
            else:
                self._send_home(arm, finished=False)
        elif not candidates:
            # Plates are left, but none can be uncovered any more.
            self._send_home(arm, finished=True)

    def _wait(self, arm: str) -> None:
        """Hold the arm until the other arm's timeline ends, then retry after its next move."""
        other = _other(arm)
        self._arms[arm].waiting_for = self._arms[other].num_motions
        self._hold(arm, self._tracks[other].now - self._tracks[arm].now, "wait for the other arm")

    def _resolve_deadlock(self) -> None:
        """Both arms wait for each other: let one of them yield.

        An idle arm yields first, then an arm on its way to a pick, then an arm
        holding a plate.
        """
        for arm in ARMS:
            if self._arms[arm].starts_after_first_lift:
                self._arms[arm].starts_after_first_lift = False
                return

        def rank(arm: str) -> int:
            a = self._arms[arm]
            return 0 if not a.queue else 1 if a.held_part is None else 2

        for arm in sorted(ARMS, key=rank):
            if self._yield(arm):
                self._arms[arm].waiting_for = None
                return
        raise RuntimeError("The arms block each other and neither can yield.")

    def _yield(self, arm: str) -> bool:
        """Make way for the other arm, which stands still. Returns True on progress.

        The arm first tries its own next move with cuRobo's standard planning. Then it
        gives up a pending pick and steps back to its home pose.
        """
        a = self._arms[arm]
        if a.queue:
            motion = self._plan_waypoint(arm, a.queue[0], THOROUGH_PASS)
            if motion is not None and self._execute(arm, motion):
                return True
        if self._pick_pending(arm):
            self._cancel_pick(arm)
            if self._at_home(arm):
                return True
        if self._at_home(arm):
            return False
        home = Waypoint(
            TRANSIT, self._home_tool[arm][0], [self._home_tool[arm][1]], label="step back home"
        )
        a.queue.insert(0, home)
        motion = self._plan_waypoint(arm, home, QUICK_PASS + THOROUGH_PASS + ANY_PASS)
        if motion is not None and self._execute(arm, motion):
            return True
        a.queue.pop(0)
        return False

    def _execute(self, arm: str, motion: _Motion) -> bool:
        """Append a planned move of the arm's next waypoint to its timeline.

        The move starts at the end of the arm's timeline, or later if it would come
        too close to the other arm's motion. Returns False if no start time works.
        """
        delay = self._earliest_start(arm, motion)
        if delay is None:
            return False
        a, track = self._arms[arm], self._tracks[arm]
        wp = a.queue.pop(0)
        self._hold(arm, delay, "wait for a gap")
        q = motion.q[1:].cpu().numpy()
        track.extend(
            q, np.full(len(q), a.finger), -1 if a.held_part is None else a.held_part, wp.label
        )
        a.orientation = motion.orientation
        a.num_motions += 1
        self._max_detour = max(self._max_detour, motion.detour)
        if wp.event == "grasp":
            self._grasp(arm, wp.part)
        elif wp.event == "release":
            self._release(arm, wp.part)
        elif wp.event == "lift":
            self._sequencer.mark_lifted(wp.part, track.now * self.dt)
            if self._first_lift_row is None:
                self._first_lift_row = track.now
        return True

    def _hold(self, arm: str, rows: int, label: str) -> None:
        """Keep the arm still for ``rows`` rows."""
        if rows <= 0:
            return
        a, track = self._arms[arm], self._tracks[arm]
        track.extend(
            np.repeat(track.q[-1:], rows, axis=0),
            np.full(rows, a.finger),
            -1 if a.held_part is None else a.held_part,
            label,
        )

    def _gripper(self, arm: str, target: float, label: str) -> None:
        """Pause while the gripper moves its fingers to ``target`` (or the suction switches)."""
        a, track = self._arms[arm], self._tracks[arm]
        rows = max(2, int(GRIPPER_TIME / self.dt))
        track.extend(
            np.repeat(track.q[-1:], rows, axis=0),
            np.linspace(a.finger, target, rows),
            -1 if a.held_part is None else a.held_part,
            label,
        )
        a.finger = target

    def _grasp(self, arm: str, part: int) -> None:
        """Close the gripper on the plate below the tool, then carry it."""
        a = self._arms[arm]
        position, quat = self._part_pose[part]
        tool = self._tool_pose_now(arm)
        a.tool_to_part = tool.inverse().multiply(Pose.from_list(list(position) + list(quat)))
        self._gripper(arm, self.layout.part_size[1] / 2, f"grasp plate {part}")
        a.held_part = part
        self._carries[part] = _Carry(
            arm, start=len(self._tracks[arm].q), tool_to_part=a.tool_to_part
        )

    def _release(self, arm: str, part: int) -> None:
        """Put the carried plate down where it is and open the gripper."""
        a = self._arms[arm]
        placed = self._tool_pose_now(arm).multiply(a.tool_to_part)
        self._part_pose[part] = (
            placed.position.view(-1).cpu().numpy(),
            placed.quaternion.view(-1).tolist(),
        )
        self._carries[part].end = len(self._tracks[arm].q)
        a.held_part = None
        a.tool_to_part = None
        self._gripper(arm, self.tool.finger_open, f"release plate {part}")

    def _pick_pending(self, arm: str) -> bool:
        """True if the arm is on its way to a plate it has not grasped yet."""
        a = self._arms[arm]
        return a.held_part is None and any(wp.event == "grasp" for wp in a.queue)

    def _cancel_pick(self, arm: str) -> None:
        """Give a plate the arm has not grasped yet back to the pile."""
        a = self._arms[arm]
        index = next(wp.part for wp in a.queue if wp.event == "grasp")
        self._sequencer.release(index)
        self._sequencer.record_failure(arm, index)
        self._pick_order[arm].remove(index)
        a.placed -= 1
        a.queue = []

    def _pick_footprint(self, arm: str) -> Optional[np.ndarray]:
        """Tool footprint over the plate ``arm`` is about to pick or lift, if any."""
        for wp in self._arms[arm].queue:
            if wp.event in ("grasp", "lift"):
                part = self._parts[wp.part]
                return tool_footprint(part.position, part.yaw, self.tool.footprint_half)
        return None

    def _descent_feasible(self, arm: str, motion: _Motion) -> bool:
        """Check that the arm can reach the bottom of its next straight-down move.

        The hover pose can be free while the descended pose is not: lower down, the
        forearm may hit the other arm. One IK solve (fingers off, as in the descent)
        catches this before a plate is committed, while another plate can still be
        chosen.
        """
        planner = self.planners[arm]
        wp = self._arms[arm].queue[1]
        end = JointState.from_position(motion.q[-1:].clone(), joint_names=planner.joint_names)
        self._set_finger_collision(arm, enable=False)
        try:
            result = planner.ik_solver.solve_pose(
                self._goal_tool_pose(arm, [(wp.position, motion.orientation)]), current_state=end
            )
        finally:
            self._set_finger_collision(arm, enable=True)
        return bool(result.success.any())

    def _at_home(self, arm: str) -> bool:
        position = self._tool_pose_now(arm).position.view(-1).cpu().numpy()
        return bool(np.linalg.norm(position - self._home_tool[arm][0]) < 0.01)

    def _pick_place_waypoints(self, arm: str, arm_state: _ArmState, part: Part) -> List[Waypoint]:
        """Six waypoints: transit, grasp, lift, transit, release, retreat."""
        layout = self.layout
        thick = layout.part_size[2]
        standoff = self.tool.standoff
        grasp = np.array(
            [part.position[0], part.position[1], part.position[2] + thick / 2 + standoff]
        )
        stacks = layout.stack_xy(arm)
        slot = arm_state.placed
        stack_xy = stacks[slot % len(stacks)]
        layer = slot // len(stacks)
        place = np.array(
            [stack_xy[0], stack_xy[1], layout.bin_floor_z + (layer + 1) * thick + standoff]
        )
        above = np.array([0.0, 0.0, APPROACH_HEIGHT])
        i = part.index
        return [
            Waypoint(
                TRANSIT,
                grasp + above,
                self._grasp_orientations(part.yaw),
                yaw=part.yaw,
                label=f"to plate {i}",
            ),
            Waypoint(DOWN, grasp, event="grasp", part=i, label=f"down to plate {i}"),
            Waypoint(UP, grasp + above, event="lift", part=i, label=f"lift plate {i}"),
            Waypoint(
                TRANSIT,
                place + above,
                self._grasp_orientations(layout.stack_yaw),
                yaw=layout.stack_yaw,
                label=f"carry plate {i}",
            ),
            Waypoint(DOWN, place, event="release", part=i, label=f"place plate {i}"),
            Waypoint(UP, place + above, label="retreat"),
        ]

    def _send_home(self, arm: str, finished: bool) -> None:
        """Queue a move to the arm's home pose, out of the shared workspace.

        Args:
            arm: Arm name.
            finished: True when the arm has no plates left; otherwise it asks for a new
                plate once it is home.
        """
        a = self._arms[arm]
        a.finished = finished
        a.queue = (
            []
            if self._at_home(arm)
            else [
                Waypoint(TRANSIT, self._home_tool[arm][0], [self._home_tool[arm][1]], label="home")
            ]
        )

    def _grasp_orientations(self, yaw: float) -> List[List[float]]:
        """Equivalent top-down tool orientations (wxyz) for a plate at ``yaw``."""
        return [top_down_quaternion(yaw + offset) for offset in self.tool.grasp_yaws]

    # ---------------------------------------------------------------- timing

    def _earliest_start(self, arm: str, motion: _Motion) -> Optional[int]:
        """Smallest delay (rows) after which ``motion`` keeps clear of the other arm.

        The move was planned with the other arm at its resting pose, but that arm may
        still be moving. The move is checked against the other arm's actual motion
        (holding its last pose after its timeline ends), with start delays up to the
        end of that motion. While it waits, the arm holds its pose; the other arm's
        moves were already checked against that.

        Returns:
            Delay in rows, or ``None`` if no delay keeps the arms ``MIN_ARM_GAP`` apart.
        """
        other = _other(arm)
        track, other_track = self._tracks[arm], self._tracks[other]
        steps = motion.q.shape[0]
        held = self._arms[arm].held_part
        mine = self._arm_spheres(arm, motion.q, np.full(steps, -1 if held is None else held))
        latest = max(0, other_track.now - track.now)
        rows = other_track.rows(track.now, latest + steps)
        theirs = self._arm_spheres(
            other, torch.as_tensor(other_track.q[rows]), other_track.held[rows]
        )
        step = max(1, round(DELAY_STEP / self.dt))
        for delay in list(range(0, latest, step)) + [latest]:
            if self._gap(mine, theirs[delay : delay + steps]) >= MIN_ARM_GAP:
                return delay
        return None

    def _arm_spheres(self, arm: str, q: torch.Tensor, held: np.ndarray) -> torch.Tensor:
        """Collision spheres ``(N, S, 4)`` of one arm and its held plate.

        Args:
            arm: Arm name.
            q: Arm joint positions ``(N, dof)`` in the joint order of the arm's planner.
            held: Plate held in each row ``(N,)``, -1 for none.
        """
        state = self._active_state({arm: q})
        kin = self.fk.compute_kinematics(state)
        n = q.shape[0]
        spheres = kin.robot_spheres.view(n, -1, 4)[:, self._arm_spheres_index[arm]]
        # Rows without a plate get spheres of radius -inf, which never limit the gap.
        plates = torch.zeros(n, self._part_spheres.shape[0], 4, device=self._device)
        plates[..., 3] = -math.inf
        if np.any(held >= 0):
            tool = self._tool_poses_batch(state)[arm]
            for part in np.unique(held[held >= 0]):
                mask = torch.as_tensor(held == part, device=self._device)
                plates[mask] = self._plate_spheres(
                    Pose(position=tool.position[mask], quaternion=tool.quaternion[mask]),
                    self._carries[int(part)].tool_to_part,
                )
        return torch.cat([spheres, plates], dim=1)

    def _plate_spheres(self, tool: Pose, tool_to_part: Pose) -> torch.Tensor:
        """Spheres ``(N, 4, 4)`` of a plate held at ``tool_to_part`` by tool poses ``(N,)``."""
        plate = self._plate_poses(tool, tool_to_part)
        n = plate.position.shape[0]
        centers = plate.batch_transform_points(
            self._part_spheres[:, :3].unsqueeze(0).repeat(n, 1, 1).contiguous()
        )
        return torch.cat(
            [centers, self._part_spheres[:, 3].view(1, -1, 1).repeat(n, 1, 1)], dim=-1
        )

    @staticmethod
    def _plate_poses(tool: Pose, tool_to_part: Pose) -> Pose:
        """Poses ``(N,)`` of a plate held at ``tool_to_part`` by tool poses ``(N,)``."""
        n = tool.position.shape[0]
        offset = Pose(
            position=tool_to_part.position.view(1, 3).repeat(n, 1),
            quaternion=tool_to_part.quaternion.view(1, 4).repeat(n, 1),
        )
        return tool.multiply(offset)

    @staticmethod
    def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
        """Smallest surface distance between spheres ``a (N, Sa, 4)`` and ``b (N, Sb, 4)``."""
        gap = torch.cdist(a[..., :3], b[..., :3]) - a[..., 3:4] - b[..., 3].unsqueeze(1)
        return float(gap.min())

    # ---------------------------------------------------------------- motion planning

    def _plan_waypoint(
        self, arm: str, wp: Waypoint, passes: Sequence[Tuple[bool, bool]]
    ) -> Optional[_Motion]:
        """Plan the arm's move to ``wp`` while the other arm rests (see ``_plan_goalset``)."""
        a = self._arms[arm]
        if wp.orientations is not None:
            quats = wp.orientations
        else:
            quats = [a.orientation if a.orientation is not None else self._home_tool[arm][1]]
        goals = [(wp.position, q) for q in quats]
        planner = self.planners[arm]
        self._lock_other_arm(arm)
        state = JointState.from_position(
            torch.as_tensor(self._tracks[arm].q[-1:], device=self._device, dtype=torch.float32),
            joint_names=planner.joint_names,
        )
        linear = wp.kind in (DOWN, UP)
        if linear:
            self._set_finger_collision(arm, enable=False)
        try:
            for thorough, strict in passes:
                planner.update_tool_pose_criteria(
                    self._linear_motion[arm] if linear else self._free_motion[arm]
                )
                motion = self._plan_goalset(
                    arm, goals, state, thorough, strict, free_path=not linear
                )
                if motion is None and linear:
                    # Some vertical moves cannot stay on a straight line; allow a curved
                    # move. The wrist and the held plate remain collision-active.
                    planner.update_tool_pose_criteria(self._free_motion[arm])
                    motion = self._plan_goalset(
                        arm, goals, state, thorough, strict, free_path=True
                    )
                if motion is not None:
                    return motion
        finally:
            planner.update_tool_pose_criteria(self._free_motion[arm])
            if linear:
                self._set_finger_collision(arm, enable=True)
        return None

    def _plan_goalset(
        self,
        arm: str,
        goals: List[GoalOption],
        state: JointState,
        thorough: bool,
        strict: bool,
        free_path: bool,
    ) -> Optional[_Motion]:
        """Plan the arm's move to any of ``goals``.

        The first attempt only lets the arm end near its current configuration (see
        ``_plan_near_branch``). If ``thorough``, graph search and then cuRobo's
        standard planning follow; the latter may pick an elbow or wrist flip.

        Args:
            arm: Arm name.
            goals: Goal options of the arm.
            state: Start configuration of the arm.
            thorough: Add graph search, standard planning, and per-goal retries.
            strict: Reject motions where the tool travels more than ``MAX_TOOL_DETOUR``
                beyond a straight line.
            free_path: The tool path is unconstrained (not a straight line).
        """
        planner = self.planners[arm]
        goal = self._goal_tool_pose(arm, goals)
        motion = self._plan_near_branch(
            arm, goals, goal, state, strict, graph_search=thorough and free_path
        )
        if motion is not None or not thorough:
            return motion
        if len(goals) > 1:
            motion = self._motion(
                arm, planner.plan_pose(goal, state, max_attempts=2), goals, strict
            )
            if motion is not None:
                return motion
        for option in goals:
            result = planner.plan_pose(self._goal_tool_pose(arm, [option]), state, max_attempts=3)
            motion = self._motion(arm, result, [option], strict)
            if motion is not None:
                return motion
        return None

    def _plan_near_branch(
        self,
        arm: str,
        goals: List[GoalOption],
        goal: GoalToolPose,
        state: JointState,
        strict: bool,
        graph_search: bool,
    ) -> Optional[_Motion]:
        """Plan to ``goal`` with trajectories that end near the current configuration.

        cuRobo keeps the fastest of its candidate trajectories, not the shortest, and
        its IK starts from random seeds. A 6-axis arm can often reach a nearby pose
        about as fast by flipping its elbow and wrist (or by turning the shoulder over
        the back), which swings the whole arm far around. Here IK starts from the
        current configuration, and only the solutions closest to it seed the
        trajectory optimizer. When the arm has to pass the other arm, the straight
        joint-space seeds fail; with ``graph_search``, a joint-space plan with graph
        search to the nearest IK solutions is tried next. That search is slow when it
        fails, so quick passes leave it out and wait for the other arm instead.
        """
        planner = self.planners[arm]
        ik_solver = planner.ik_solver
        num_seeds = planner.trajopt_solver.config.num_seeds
        start = state.position.view(1, 1, -1)
        ik_seeds = start + NEAR_BRANCH_SEED_NOISE * torch.randn(
            1, ik_solver.config.num_seeds, start.shape[-1], device=start.device, dtype=start.dtype
        )
        ik_seeds[:, 0] = start[:, 0]
        ik = ik_solver.solve_pose(
            goal, current_state=state, seed_config=ik_seeds, return_seeds=num_seeds
        )
        success = ik.success.view(-1)
        if not success.any():
            return None
        solutions = ik.solution.reshape(-1, ik.solution.shape[-1])
        distance = (solutions - state.position.view(1, -1)).abs().sum(dim=-1)
        distance[~success] = float("inf")
        near = solutions[distance <= distance.min() + NEAR_BRANCH_MARGIN]
        seeds = near[torch.arange(solutions.shape[0], device=near.device) % near.shape[0]]
        result = planner.trajopt_solver.solve_pose(
            goal, state, seed_config=seeds.view_as(ik.solution), use_implicit_goal=True
        )
        motion = self._motion(arm, result, goals, strict)
        if motion is not None or not graph_search:
            return motion
        for j in torch.argsort(distance)[: int(success.sum().clamp(max=2))].tolist():
            target = JointState.from_position(
                solutions[j : j + 1].clone(), joint_names=planner.joint_names
            )
            motion = self._motion(
                arm, planner.plan_cspace(target, state, max_attempts=2), goals, strict
            )
            if motion is not None:
                return motion
        return None

    def _motion(
        self,
        arm: str,
        result: Optional[TrajectoryOptimizerResult],
        goals: List[GoalOption],
        strict: bool,
    ) -> Optional[_Motion]:
        """Turn a successful planner result into a move; if ``strict``, reject detours."""
        if result is None or not result.success.any():
            return None
        planner = self.planners[arm]
        traj = result.get_interpolated_plan()
        q = traj.position.reshape(-1, traj.position.shape[-1])
        q = q[:, [traj.joint_names.index(n) for n in planner.joint_names]].clone()
        tool = self._tool_poses_batch(self._active_state({arm: q}))[arm]
        detour = _path_detour(tool.position)
        if strict and detour > MAX_TOOL_DETOUR:
            return None
        # cuRobo reports the goal set entry it reached; match the end pose directly
        # instead of trusting an index from reused solver buffers.
        end = np.array(tool.quaternion[-1].tolist())
        orientation = max(
            (option[1] for option in goals), key=lambda quat: abs(float(np.dot(end, quat)))
        )
        return _Motion(q, list(orientation), detour)

    def _goal_tool_pose(self, arm: str, goals: List[GoalOption]) -> GoalToolPose:
        """Goal set of the arm's tool frame with one entry per goal option."""
        position = torch.tensor(
            np.array([g[0] for g in goals]), device=self._device, dtype=torch.float32
        )
        quaternion = torch.tensor([g[1] for g in goals], device=self._device, dtype=torch.float32)
        return GoalToolPose.from_poses(
            {self._tool_frame[arm]: Pose(position=position, quaternion=quaternion)},
            ordered_tool_frames=self.planners[arm].tool_frames,
            num_goalset=len(goals),
        )

    # ---------------------------------------------------------------- collision bookkeeping

    def _lock_other_arm(self, arm: str) -> None:
        """Put the other arm in ``arm``'s planner at the end of its timeline.

        The other arm's joints are locked in this planner, so they are fixed transforms
        of its kinematic tree. These transforms are recomputed from the other arm's
        last planned configuration and written in place, which keeps CUDA graphs
        valid. Both arms' held plates are attached as they are at that time.
        """
        other = _other(arm)
        key = (
            self._tracks[other].q[-1].tobytes(),
            self._arms[ARMS[0]].held_part,
            self._arms[ARMS[1]].held_part,
        )
        if self._lock_cache.get(arm) == key:
            return
        parents, children, rows = self._locked[arm]
        state = self._active_state({other: self._tracks[other].q[-1:]})
        poses = self.fk.get_link_poses(state.position, parents + children)
        n = len(parents)
        parent = Pose(
            position=poses.position[0, :n].contiguous(),
            quaternion=poses.quaternion[0, :n].contiguous(),
        )
        child = Pose(
            position=poses.position[0, n:].contiguous(),
            quaternion=poses.quaternion[0, n:].contiguous(),
        )
        planner = self.planners[arm]
        planner.kinematics.config.kinematics_config.fixed_transforms[rows] = (
            parent.inverse().multiply(child).get_affine_matrix()
        )
        for holder in ARMS:
            self._update_attachment(planner, holder)
        if planner.graph_planner is not None:
            planner.graph_planner.reset_buffer()
        self._lock_cache[arm] = key

    def _set_finger_collision(self, arm: str, enable: bool) -> None:
        """Toggle the arm's finger spheres for straight grasp/release moves.

        The open fingers reach down next to the plate and may touch the pile or the
        bin floor. Hand and held-plate spheres stay active, so neither the hand nor the
        carried plate can touch the other arm. Does nothing for tools without fingers.
        """
        fingers = [f"{arm}_{link}" for link in self.tool.finger_links]
        if not fingers:
            return
        if enable:
            self.planners[arm].enable_link_collision(fingers)
        else:
            self.planners[arm].disable_link_collision(fingers)

    def _update_attachment(self, planner: MotionPlanner, holder: str) -> None:
        """Write the plate held by ``holder`` onto its attachment link in ``planner``."""
        a = self._arms[holder]
        manager = planner.attachment_manager
        if a.held_part is None:
            manager.detach(link_name=ATTACH_LINK[holder])
            return
        spheres = self._part_spheres.clone()
        spheres[:, :3] = a.tool_to_part.transform_points(spheres[:, :3].contiguous()).view(-1, 3)
        manager.update(
            spheres,
            planner.default_joint_state.clone().unsqueeze(0),
            link_name=ATTACH_LINK[holder],
        )

    def _reset_attachments(self) -> None:
        for planner in self.planners.values():
            for holder in ARMS:
                planner.attachment_manager.detach(link_name=ATTACH_LINK[holder])

    # ---------------------------------------------------------------- timeline

    def _segments(self, start: int, end: int) -> List[Segment]:
        """Playback segments for rows ``[start, end)``, split where either arm starts a move."""
        cuts = {start, end}
        for track in self._tracks.values():
            for first, last, _ in track.spans:
                cuts.update(r for r in (first, last) if start < r < end)
        cuts = sorted(cuts)
        return [self._segment(a, b) for a, b in zip(cuts[:-1], cuts[1:])]

    def _segment(self, start: int, end: int) -> Segment:
        """Playback segment for rows ``[start, end)`` of both arms' timelines."""
        rows = np.arange(start, end)
        q = {arm: track.q[track.rows(start, len(rows))] for arm, track in self._tracks.items()}
        active = self._active_state(q)
        positions = self.fk.get_full_js(active).position.cpu().numpy().copy()
        for arm, track in self._tracks.items():
            for name in self.tool.finger_joints:
                positions[:, self.full_joint_names.index(f"{arm}_{name}")] = track.finger[
                    track.rows(start, len(rows))
                ]
        label = ", ".join(f"{arm}: {self._span_label(arm, start)}" for arm in ARMS)
        segment = Segment(label, positions, self.dt)
        tool = None
        for part, carry in self._carries.items():
            carry_end = math.inf if carry.end is None else carry.end
            if carry.start >= end or carry_end <= start:
                continue
            if tool is None:
                tool = self._tool_poses_batch(active)
            pile_position, pile_quat = self._pile_pose[part]
            position = np.repeat(np.asarray(pile_position)[None], len(rows), axis=0)
            quat = np.repeat(np.asarray(pile_quat)[None], len(rows), axis=0)
            if carry.end is not None:
                placed_position, placed_quat = self._part_pose[part]
                after = rows >= carry.end
                position[after] = placed_position
                quat[after] = placed_quat
            held = (rows >= carry.start) & (rows < carry_end)
            mask = torch.as_tensor(held, device=self._device)
            plate = self._plate_poses(
                Pose(
                    position=tool[carry.arm].position[mask],
                    quaternion=tool[carry.arm].quaternion[mask],
                ),
                carry.tool_to_part,
            )
            position[held] = plate.position.cpu().numpy()
            quat[held] = plate.quaternion.cpu().numpy()
            segment.part_poses[part] = (position, quat)
            segment.carried_by[part] = carry.arm
        return segment

    def _span_label(self, arm: str, row: int) -> str:
        for first, last, label in self._tracks[arm].spans:
            if first <= row < last:
                return label
        return "idle"

    # ---------------------------------------------------------------- kinematics helpers

    def _active_state(self, rows: Dict[str, object]) -> JointState:
        """Joint state of both arms from per-arm rows ``(N, dof)``; missing arms are at home."""
        n = max([len(r) for r in rows.values()], default=1)
        q = torch.zeros(n, len(self.fk.joint_names), device=self._device, dtype=torch.float32)
        for arm in ARMS:
            values = rows[arm] if arm in rows else self._home_q[arm][None]
            q[:, self._columns[arm]] = torch.as_tensor(
                values, device=self._device, dtype=torch.float32
            )
        return JointState.from_position(q, joint_names=list(self.fk.joint_names))

    def _tool_pose_now(self, arm: str) -> Pose:
        """Tool pose of the arm at the end of its timeline."""
        return self._tool_poses_batch(self._active_state({arm: self._tracks[arm].q[-1:]}))[arm]

    def _tool_poses_batch(self, active: JointState) -> Dict[str, Pose]:
        """Tool poses of both arms along a trajectory of active joint states."""
        kin = self.fk.compute_kinematics(active)
        out = {}
        for arm in ARMS:
            pose = kin.tool_poses[self._tool_frame[arm]]
            out[arm] = Pose(
                position=pose.position.reshape(-1, 3).clone(),
                quaternion=pose.quaternion.reshape(-1, 4).clone(),
            )
        return out

    @staticmethod
    def _part_quaternion(yaw: float) -> List[float]:
        return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    def destroy(self) -> None:
        """Release the planners' GPU resources."""
        for planner in self.planners.values():
            planner.destroy()


# --------------------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------------------

ARM_COLORS = {"left": (66, 133, 244), "right": (244, 140, 46)}
PART_COLOR = (150, 156, 166)


class _PlanningCancelled(Exception):
    """Raised from the segment callback to stop planning when the user starts over."""


def _drain(stream: "queue.Queue[Optional[Segment]]") -> Iterable[Segment]:
    """Yield segments from a queue until a ``None`` sentinel arrives."""
    while True:
        segment = stream.get()
        if segment is None:
            return
        yield segment


class BinPickingViewer:
    """Viser front end: draws the cell, plans on request, and plays back the plan."""

    def __init__(self, demo: DualArmBinPicking, num_parts: int, seed: int, port: int):
        """Start the Viser server and draw the cell; ``seed`` selects the first pile."""
        from curobo.viewer import ViserVisualizer

        self.demo = demo
        self.num_parts = num_parts
        self.seed = seed
        self.viz = ViserVisualizer(
            content_path=ContentPath(robot_config_file=demo.robot_file),
            connect_ip="0.0.0.0",
            connect_port=port,
            add_control_frames=False,
        )
        self.server = self.viz._server
        layout = demo.layout
        self.server.scene.add_grid(
            "/ground_plane", width=4, height=4, position=(0.0, 0.0, -layout.table_height)
        )
        self._draw_cell()
        self._part_handles: Dict[int, object] = {}
        self._owner: Dict[int, str] = {}
        self._plan: Optional[TaskPlan] = None
        self._parts: List[Part] = []
        self._busy = threading.Lock()
        self._stop_playback = threading.Event()

        gui = self.server.gui
        gui.add_markdown(
            f"**Dual-arm bin picking** ({demo.schedule} start)  \n"
            "Blue = left arm, orange = right arm. Labels show each arm's pick order."
        )
        self.plan_status = gui.add_markdown("Starting...")
        self.play_status = gui.add_markdown("")
        self.stats = gui.add_markdown("")
        self.speed = gui.add_slider(
            "Playback speed", min=0.25, max=4.0, step=0.25, initial_value=1.0
        )
        self.new_button = gui.add_button("New random scene", color="green")
        self.replay_button = gui.add_button("Replay", color="blue")
        self.new_button.on_click(lambda _: self._start(self._new_scene_and_plan))
        self.replay_button.on_click(lambda _: self._start(self._replay))

    def run(self) -> None:
        """Plan and play the first scene, then serve until Ctrl+C."""
        self._start(self._new_scene_and_plan, new_seed=False)
        print(f"\nDual-arm bin picking running at http://localhost:{self.server.get_port()}")
        print("Press Ctrl+C to exit.\n")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nShutting down...")

    # ---------------------------------------------------------------- actions

    def _start(self, fn: Callable[..., None], **kwargs) -> None:
        """Stop current playback/planning and run ``fn`` on a worker thread."""
        self._stop_playback.set()

        def worker() -> None:
            with self._busy:
                self._stop_playback.clear()
                fn(**kwargs)

        threading.Thread(target=worker, daemon=True).start()

    def _new_scene_and_plan(self, new_seed: bool = True) -> None:
        """Sample a new pile, plan it, and play segments while they are planned."""
        if new_seed:
            self.seed += 1
        self._parts = sample_parts(
            self.demo.layout, self.num_parts, np.random.default_rng(self.seed)
        )
        self._plan = None
        self.stats.content = ""
        self.plan_status.content = f"**Planning** {len(self._parts)} plates (seed {self.seed})..."
        stream: "queue.Queue[Optional[Segment]]" = queue.Queue()
        player = threading.Thread(target=self._play, args=(_drain(stream),), daemon=True)
        player.start()

        def on_segment(segment: Segment) -> None:
            if self._stop_playback.is_set():
                raise _PlanningCancelled()
            stream.put(segment)

        def on_progress(message: str) -> None:
            self.plan_status.content = message

        try:
            plan = self.demo.plan(self._parts, progress=on_progress, on_segment=on_segment)
        except _PlanningCancelled:
            return
        except RuntimeError as e:
            self.plan_status.content = f"**Planning failed:** {e}"
            return
        finally:
            stream.put(None)
        self._plan = plan
        self.plan_status.content = f"**Planned** in {plan.planning_time:.1f} s"
        self.stats.content = _format_stats(plan, len(self._parts))
        player.join()

    def _replay(self) -> None:
        if self._plan is not None:
            self._play(self._plan.segments)

    def _play(self, segments: Iterable[Segment]) -> None:
        """Play segments in real time (scaled by the speed slider)."""
        self.viz.reset_robot()
        self._draw_parts(self._parts)
        self.play_status.content = "**Executing**"
        picked = {arm: 0 for arm in ARMS}
        names = self.demo.full_joint_names
        for seg in segments:
            for index, arm in seg.carried_by.items():
                if index not in self._owner:
                    picked[arm] += 1
                    self._mark_picked(index, arm, picked[arm])
            for t in range(len(seg.positions)):
                if self._stop_playback.is_set():
                    return
                frame_start = time.time()
                q = torch.as_tensor(seg.positions[t])
                self.viz.set_joint_state(JointState.from_position(q, joint_names=names))
                for index, (pos, quat) in seg.part_poses.items():
                    handle = self._part_handles[index]
                    handle.position = pos[t]
                    handle.wxyz = quat[t]
                time.sleep(max(0.0, seg.dt / self.speed.value - (time.time() - frame_start)))
        self.play_status.content = "**Done.** Click *Replay* or *New random scene*."

    # ---------------------------------------------------------------- drawing

    def _draw_cell(self) -> None:
        scene = self.demo.layout.scene_cfg()["cuboid"]
        for name, cuboid in scene.items():
            if name == "table":
                color = (190, 170, 140)
            elif "pedestal" in name:
                color = (90, 90, 95)
            elif name.startswith("source"):
                color = (60, 120, 90)
            else:
                color = (70, 90, 140)
            self.server.scene.add_box(
                f"/cell/{name}",
                color=color,
                dimensions=tuple(cuboid["dims"]),
                position=tuple(cuboid["pose"][:3]),
                wxyz=tuple(cuboid["pose"][3:]),
            )

    def _draw_parts(self, parts: List[Part]) -> None:
        """Draw all plates in gray at their start poses."""
        for handle in self._part_handles.values():
            handle.remove()
        self._part_handles = {}
        self._owner = {}
        for part in parts:
            self._part_handles[part.index] = self.server.scene.add_box(
                f"/parts/{part.index}",
                color=PART_COLOR,
                dimensions=self.demo.layout.part_size,
                position=tuple(part.position),
                wxyz=tuple(DualArmBinPicking._part_quaternion(part.yaw)),
            )

    def _mark_picked(self, index: int, arm: str, rank: int) -> None:
        """Color a plate by the arm that picked it and label it with the pick order."""
        self._owner[index] = arm
        self._part_handles[index].color = ARM_COLORS[arm]
        self.server.scene.add_label(
            f"/parts/{index}/label",
            text=f"{arm[0].upper()}{rank}",
            position=(0.0, 0.0, 0.006),
            font_screen_scale=0.6,
        )


def _format_stats(plan: TaskPlan, num_parts: int) -> str:
    left, right = len(plan.pick_order["left"]), len(plan.pick_order["right"])
    return (
        f"Plates moved: **{left + right}/{num_parts}** "
        f"(left {left}, right {right})  \n"
        f"Arm moves planned: {plan.num_motions}  \n"
        f"Planning time: {plan.planning_time:.1f} s  \n"
        f"Motion time: {plan.execution_time:.1f} s  \n"
        f"Min. arm-to-arm clearance: {plan.min_arm_clearance * 1000:.0f} mm  \n"
        f"Labels: L/R = arm, number = pick order"
    )


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def plan_headless(demo: DualArmBinPicking, num_parts: int, seed: int) -> TaskPlan:
    """Sample a pile, plan the full task, and print a summary."""
    parts = sample_parts(demo.layout, num_parts, np.random.default_rng(seed))
    plan = demo.plan(parts)
    layers = max(
        round((p.position[2] - demo.layout.bin_floor_z) / demo.layout.part_size[2] + 0.5)
        for p in parts
    )
    print(f"seed {seed}: {len(parts)} plates in {layers} layers")
    for arm in ARMS:
        print(f"  {arm:5s} pick order: {plan.pick_order[arm]}")
    print(
        f"  moved {len(parts) - len(plan.leftover)}/{len(parts)}, {plan.num_motions} moves, "
        f"planning {plan.planning_time:.1f} s, motion {plan.execution_time:.1f} s, "
        f"min arm clearance {plan.min_arm_clearance * 1000:.1f} mm, "
        f"max tool detour {plan.max_tool_detour * 1000:.0f} mm"
    )
    return plan


def lifted_in_pile_order(plan: TaskPlan, parts: Sequence[Part]) -> bool:
    """True if every plate was picked strictly after all plates lying on top of it."""
    times = plan.pick_times
    return all(
        b in times and (p.index not in times or times[b] < times[p.index])
        for p in parts
        for b in p.blockers
    )


def test(robot_file: str = ROBOT_FILE) -> None:
    """Check that random piles are emptied in pile order without arm-to-arm contact."""
    for schedule in ("staggered", "synchronous"):
        demo = DualArmBinPicking(schedule=schedule, robot_file=robot_file)
        for seed in (0, 1):
            plan = plan_headless(demo, num_parts=12, seed=seed)
            parts = sample_parts(demo.layout, 12, np.random.default_rng(seed))
            assert not plan.leftover, f"{schedule}, seed {seed}: plates left {plan.leftover}"
            assert plan.pick_order["left"] and plan.pick_order["right"], "both arms should work"
            assert plan.min_arm_clearance > 0.0, f"arms touch: {plan.min_arm_clearance:.4f} m"
            assert lifted_in_pile_order(plan, parts), "a plate was lifted before the plate on top"
            assert plan.max_tool_detour < 0.3, (
                f"an arm swung far around: {plan.max_tool_detour:.2f} m"
            )
        demo.destroy()


def main() -> None:
    """Parse arguments and run the viewer, a headless plan, or the self-test."""
    parser = argparse.ArgumentParser(description="Dual-arm bin picking with cuRobo")
    parser.add_argument("--num-parts", type=int, default=14, help="Number of plates (default: 14)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the first pile")
    parser.add_argument(
        "--schedule",
        choices=["staggered", "synchronous"],
        default="staggered",
        help="staggered: the right arm starts after the left arm's first pick; "
        "synchronous: both arms start together",
    )
    parser.add_argument(
        "--robot",
        default=ROBOT_FILE,
        help=f"Dual-arm robot config (default: {ROBOT_FILE}; dual_franka.yml also works)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    parser.add_argument("--headless", action="store_true", help="Plan once without the viewer")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    args = parser.parse_args()

    if args.test:
        test(args.robot)
        sys.exit(0)

    demo = DualArmBinPicking(schedule=args.schedule, robot_file=args.robot)
    if args.headless:
        plan_headless(demo, args.num_parts, args.seed)
        return
    BinPickingViewer(demo, args.num_parts, args.seed, args.port).run()


if __name__ == "__main__":
    main()
