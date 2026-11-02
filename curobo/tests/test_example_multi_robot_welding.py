# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for seam geometry, timing, occupancy, and interlocks of the welding example."""

import math
from typing import Dict, List

import numpy as np
import pytest
import torch

from curobo.examples.reference.multi_robot_welding import (
    APPROACH,
    DT,
    RETRACT,
    SEAM,
    OccupancyGrid,
    Seam,
    ToolPath,
    WeldCell,
    WeldMotion,
    WeldProcess,
    WeldScheduler,
    seam_tool_path,
    time_tool_path,
    torch_rotation,
    trapezoid_times,
)


@pytest.mark.parametrize("push_deg", [0.0, 10.0, 25.0])
def test_torch_rotation_holds_work_and_push_angle(push_deg):
    """The torch axis keeps the work angle and tilts by the push angle."""
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    rotation = torch_rotation(tangent, normal, math.radians(push_deg))
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    z = rotation[:, 2]
    # The torch points into the joint along the bisector, tilted towards the travel direction.
    assert np.dot(z, tangent) == pytest.approx(math.sin(math.radians(push_deg)))
    assert np.dot(-z, normal) == pytest.approx(math.cos(math.radians(push_deg)))


def test_tool_path_approaches_along_torch_axis():
    """Approach and retract are straight moves along the torch axis."""
    process = WeldProcess()
    seam = Seam("s", np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]), np.array([0.0, 0.0, 1.0]))
    for reverse in (False, True):
        path = seam_tool_path(seam, process, reverse)
        first, last = path.seam_range
        start, end = (
            (seam.points[1], seam.points[0]) if reverse else (seam.points[0], seam.points[1])
        )
        assert path.positions[first] == pytest.approx(start)
        assert path.positions[last] == pytest.approx(end)
        assert np.all(path.phase[:first] == APPROACH) and np.all(path.phase[last + 1 :] == RETRACT)
        # Approach and retract run along the torch axis, out to the approach distance.
        for index, edge in ((0, first), (-1, last)):
            offset = path.positions[index] - path.positions[edge]
            axis = path.rotations[edge][:, 2]
            assert offset == pytest.approx(-process.approach_distance * axis, abs=1e-9)


def test_trapezoid_times_profiles():
    """Trapezoidal and triangular speed profiles take the expected time."""
    s = np.linspace(0.0, 1.0, 101)
    t = trapezoid_times(s, speed=0.5, acceleration=1.0)
    assert np.all(np.diff(t) > 0)
    assert t[-1] == pytest.approx(0.5 / 1.0 * 2 + (1.0 - 0.25) / 0.5)
    short = trapezoid_times(np.linspace(0.0, 0.01, 11), speed=0.5, acceleration=1.0)
    assert short[-1] == pytest.approx(2 * math.sqrt(0.01 / 1.0))


def _straight_path(length: float, process: WeldProcess) -> ToolPath:
    seam = Seam("s", np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]), np.array([0.0, 0.0, 1.0]))
    return seam_tool_path(seam, process)


def test_time_tool_path_welds_at_travel_speed():
    """The tip moves at the travel speed and pauses at the seam start."""
    process = WeldProcess(travel_speed=0.05, arc_dwell=0.5)
    path = _straight_path(0.3, process)
    # One joint moves 1 rad per meter of tip travel: the joint trace is the path length.
    q = path.arc_length[:, None].repeat(2, axis=1)
    positions, bead = time_tool_path(q, path, process, joint_speed=np.array([10.0, 10.0]))
    first, last = path.seam_range
    s_seam = (path.arc_length[first], path.arc_length[last])
    welding = (bead > 0.1) & (bead < 0.9)
    speed = np.diff(positions[:, 0])[welding[1:]] / DT
    assert speed == pytest.approx(process.travel_speed, rel=0.02)
    assert bead[0] == -1.0 and bead[-1] == pytest.approx(1.0)
    assert np.all(np.diff(bead[bead >= 0]) >= -1e-6)
    # The torch stands still at the seam start while the arc ignites.
    at_start = np.isclose(positions[:, 0], s_seam[0], atol=1e-6) & (bead >= 0)
    assert at_start.sum() * DT >= process.arc_dwell - 2 * DT


def test_time_tool_path_slows_down_for_joint_limits():
    """Joint velocity limits stretch the motion instead of being exceeded."""
    process = WeldProcess()
    path = _straight_path(0.2, process)
    q = path.arc_length[:, None] * 20.0  # a joint that would need 0.8 rad/s while welding
    fast, _ = time_tool_path(q, path, process, joint_speed=np.array([10.0]))
    slow, _ = time_tool_path(q, path, process, joint_speed=np.array([0.4]))
    assert len(slow) > 1.5 * len(fast)
    assert np.max(np.abs(np.diff(slow[:, 0]))) / DT <= 0.4 * 1.05


def test_occupancy_grid_separates_disjoint_spheres():
    """Spheres without a shared voxel are at least two margins apart."""
    grid = OccupancyGrid(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        voxel_size=0.03,
        margin=0.02,
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = torch.tensor([*rng.uniform(-0.5, 0.5, 3), rng.uniform(0.01, 0.1)], dtype=torch.float32)
        b = torch.tensor([*rng.uniform(-0.5, 0.5, 3), rng.uniform(0.01, 0.1)], dtype=torch.float32)
        gap = float(torch.linalg.norm(a[:3] - b[:3]) - a[3] - b[3])
        shared = bool((grid.mark(a[None]) & grid.mark(b[None])).any())
        # No shared voxel guarantees at least 2 * margin between the spheres.
        if not shared:
            assert gap >= 2 * grid.margin - 1e-6
        if gap < 2 * grid.margin:
            assert shared


def test_default_programs_weld_every_seam_once():
    """The default programs cover every seam exactly once."""
    cell = WeldCell()
    names = [s.name for s in cell.seams()]
    assert len(names) == len(set(names))
    programmed = [n for program in cell.programs().values() for n in program]
    assert sorted(programmed) == sorted(names)


def test_scene_is_expressed_in_robot_base_frame():
    """Obstacles in a robot's scene map back to their world poses."""
    cell = WeldCell()
    for station in cell.stations:
        scene = cell.scene_cfg(station.name)["cuboid"]
        for name, cuboid in cell.obstacles().items():
            local = np.array(scene[name]["pose"][:3])
            assert station.to_world(local) == pytest.approx(cuboid["pose"][:3], abs=1e-9)


# --------------------------------------------------------------------------------------
# Scheduler with robots that move on a line
# --------------------------------------------------------------------------------------


class _LineGrid:
    def centers(self, mask: torch.Tensor) -> np.ndarray:
        x = torch.nonzero(mask).view(-1).float().numpy()
        return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


class _LinePlanner:
    """Robots on a line from 0 to 100. A robot at ``x`` takes the cells ``x - 4 .. x + 4``.

    Robot ``a`` stands at 0 and looks along +x, robot ``b`` stands at 100 and looks back;
    a robot's joint value is its distance from its base. A seam is a point on the line,
    welded in place in 40 ticks; transfers move one cell per tick. A robot reaches seams
    up to ``reach`` cells from its base.
    """

    def __init__(self, reach: float = 100.0):
        self.reach = reach
        self.home = np.array([10.0], dtype=np.float32)
        self.max_acceleration = np.array([1.0])
        self.grid = _LineGrid()

    @staticmethod
    def world(robot: str, q: np.ndarray) -> np.ndarray:
        return q if robot == "a" else 100.0 - q

    def plan_weld(self, robot: str, seam: Seam, reverse: bool) -> WeldMotion:
        x = float(seam.points[0, 0])
        q = x if robot == "a" else 100.0 - x
        if q > self.reach:
            return None
        positions = np.full((40, 1), q, dtype=np.float32)
        bead = np.linspace(0.0, 1.0, 40).astype(np.float32)
        path = ToolPath(
            np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0), np.array([SEAM, SEAM])
        )
        return WeldMotion(robot, seam.name, reverse, positions, bead, path)

    def plan_transfer(self, robot: str, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        steps = max(2, int(abs(float(goal[0] - start[0]))) + 1)
        return np.linspace(start, goal, steps).astype(np.float32)

    def occupancy(self, robot: str, positions: np.ndarray) -> torch.Tensor:
        x = self.world(robot, np.asarray(positions).reshape(-1))
        cells = torch.zeros(101, dtype=torch.bool)
        cells[max(0, int(math.floor(x.min() - 4))) : int(math.ceil(x.max() + 4)) + 1] = True
        return cells


def _seams(points: Dict[str, float]) -> List[Seam]:
    return [
        Seam(n, np.array([[x, 0.0, 0.0], [x, 0.0, 0.0]]), np.array([0.0, 0.0, 1.0]))
        for n, x in points.items()
    ]


def _min_gap(planner: _LinePlanner, plan) -> float:
    xa = planner.world("a", plan.joint_trajectory("a")[:, 0])
    xb = planner.world("b", plan.joint_trajectory("b")[:, 0])
    return float(np.min(xb - xa))


def test_interlock_makes_robot_wait_for_shared_space():
    """A robot waits while another robot holds the space it needs."""
    planner = _LinePlanner()
    seams = _seams({"sa": 45.0, "sb": 52.0})
    programs = {"a": ["sa"], "b": ["sb"]}
    plan = WeldScheduler(planner, seams, ["a", "b"], programs=programs).plan()
    assert not plan.leftover
    assert len(plan.waits) == 1
    # Both welds need cells 48..49, so they must not run at the same time.
    assert _min_gap(planner, plan) > 8.0
    welds = {r: next(m for m in plan.motions[r] if m.kind == "weld") for r in ("a", "b")}
    assert welds["a"].end <= welds["b"].start or welds["b"].end <= welds["a"].start

    unsafe = WeldScheduler(planner, seams, ["a", "b"], programs=programs, interlocks=False).plan()
    assert not unsafe.waits
    assert _min_gap(planner, unsafe) < 8.0


def test_robots_blocking_each_other_retreat_home():
    """Robots that block each other resolve it by one retreating home."""
    # After their first welds each robot stands in the other's way to its second seam.
    planner = _LinePlanner()
    seams = _seams({"a1": 45.0, "a2": 58.0, "b1": 55.0, "b2": 42.0})
    programs = {"a": ["a1", "a2"], "b": ["b1", "b2"]}
    plan = WeldScheduler(planner, seams, ["a", "b"], programs=programs).plan()
    assert not plan.leftover
    assert plan.yields >= 1
    assert plan.weld_order("a") == ["a1", "a2"] and plan.weld_order("b") == ["b1", "b2"]
    # Occupied cells (x +- 4) never overlap.
    assert _min_gap(planner, plan) > 8.0


def test_dynamic_assignment_shares_seams():
    """Run-time seam assignment gives both robots work and keeps them apart."""
    planner = _LinePlanner()
    seams = _seams({f"s{k}": x for k, x in enumerate((20.0, 30.0, 50.0, 70.0, 80.0))})
    plan = WeldScheduler(planner, seams, ["a", "b"]).plan()
    assert not plan.leftover
    assert plan.weld_order("a") and plan.weld_order("b")
    assert _min_gap(planner, plan) > 8.0


def test_unreachable_program_step_is_handed_over():
    """A program step its robot cannot reach goes to another robot that can."""
    planner = _LinePlanner(reach=70.0)
    seams = _seams({"near": 30.0, "far": 85.0})
    plan = WeldScheduler(planner, seams, ["a", "b"], programs={"a": ["near", "far"]}).plan()
    assert not plan.leftover
    assert plan.handovers == [("far", "a", "b")]
    assert plan.weld_order("a") == ["near"] and plan.weld_order("b") == ["far"]
