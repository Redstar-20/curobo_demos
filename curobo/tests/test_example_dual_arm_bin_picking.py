# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the pile and pick-sequencing logic of the dual-arm bin picking example."""

import math

import numpy as np
import pytest

from curobo.examples.reference.dual_arm_bin_picking import (
    PANDA_HAND,
    CellLayout,
    Part,
    PickSequencer,
    TaskPlan,
    _Track,
    lifted_in_pile_order,
    sample_parts,
    tool_footprint,
    top_down_quaternion,
)


@pytest.fixture
def layout() -> CellLayout:
    return CellLayout()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_parts_stay_in_bin_and_stack_flat(layout, seed):
    parts = sample_parts(layout, 14, np.random.default_rng(seed))
    thick = layout.part_size[2]
    half_inner = np.array(layout.bin_inner_size) / 2
    for part in parts:
        offset = np.abs(part.position[:2] - np.array(layout.source_bin_xy))
        assert np.all(offset < half_inner)
        layer = (part.position[2] - layout.bin_floor_z) / thick - 0.5
        assert layer == pytest.approx(round(layer), abs=1e-6)
        for blocker in part.blockers:
            assert parts[blocker].position[2] > part.position[2]
    assert any(p.blockers for p in parts), "a 14-plate pile should have overlapping plates"


def test_sequencer_offers_only_uncovered_plates():
    bottom = Part(0, np.array([-0.16, 0.0, 0.014]), 0.0, blockers={1})
    top = Part(1, np.array([-0.16, 0.01, 0.022]), 0.0)
    sequencer = PickSequencer([bottom, top])

    assert sequencer.candidates("left", time=0.0) == [1]
    sequencer.commit(1)
    # The plate below is offered only from the time the plate on top is lifted.
    assert sequencer.candidates("right", time=5.0) == []
    sequencer.mark_lifted(1, time=2.0)
    assert sequencer.candidates("right", time=1.0) == []
    assert sequencer.candidates("right", time=2.0) == [0]


def test_sequencer_prefers_plates_near_the_arm():
    near_left = Part(0, np.array([-0.16, 0.1, 0.014]), 0.0)
    near_right = Part(1, np.array([-0.16, -0.1, 0.014]), 0.0)
    sequencer = PickSequencer([near_left, near_right])
    assert sequencer.candidates("left", time=0.0)[0] == 0
    assert sequencer.candidates("right", time=0.0)[0] == 1


@pytest.mark.parametrize("yaw, compatible", [(0.0, False), (math.pi / 2, True)])
def test_sequencer_skips_overlapping_tool_footprints(yaw, compatible):
    # Plates 12 cm apart along y. At yaw 0 each Franka hand spans about 25 cm along y
    # and the footprints overlap; at yaw pi/2 the hands are narrow along y and fit.
    half = PANDA_HAND.footprint_half
    a = Part(0, np.array([-0.16, 0.06, 0.014]), yaw)
    b = Part(1, np.array([-0.16, -0.06, 0.014]), yaw)
    sequencer = PickSequencer([a, b], footprint_half=half)
    offered = sequencer.candidates(
        "right", 0.0, keep_away_from=tool_footprint(a.position, a.yaw, half)
    )
    assert (1 in offered) == compatible


def test_sequencer_rejects_after_repeated_failures():
    part = Part(0, np.array([-0.16, 0.0, 0.014]), 0.0)
    sequencer = PickSequencer([part], max_failures=2)
    sequencer.record_failure("left", 0)
    assert sequencer.candidates("left", 0.0) == [0]
    sequencer.record_failure("left", 0)
    assert sequencer.candidates("left", 0.0) == []
    assert not sequencer.has_work("left")
    assert sequencer.has_work("right")


def test_released_plate_is_offered_again_and_still_blocks():
    bottom = Part(0, np.array([-0.16, 0.0, 0.014]), 0.0, blockers={1})
    top = Part(1, np.array([-0.16, 0.01, 0.022]), 0.0)
    sequencer = PickSequencer([bottom, top])
    sequencer.commit(1)
    sequencer.mark_lifted(1, time=2.0)
    sequencer.release(1)
    assert 1 in sequencer.remaining
    # The released plate is back on top, so the plate below is covered again.
    assert sequencer.candidates("right", time=3.0) == [1]


def test_top_down_quaternion_points_hand_down():
    w, x, y, z = top_down_quaternion(0.3)
    # Hand z-axis in world frame: third column of the rotation matrix.
    hand_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    assert hand_z == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_track_holds_its_last_state_after_its_end():
    track = _Track(
        q=np.zeros((1, 2), dtype=np.float32),
        finger=np.zeros(1, dtype=np.float32),
        held=np.array([-1]),
    )
    track.extend(
        np.ones((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32), held=3, label="move"
    )
    assert track.now == 2
    assert track.held.tolist() == [-1, 3, 3]
    assert track.spans == [(1, 3, "move")]
    # Rows past the end repeat the last row: the arm holds still until its next move.
    assert track.rows(1, 4).tolist() == [1, 2, 2, 2]


@pytest.mark.parametrize("times, in_order", [({1: 2.0, 0: 4.0}, True), ({1: 4.0, 0: 2.0}, False)])
def test_pile_order_is_checked_on_pick_times(times, in_order):
    bottom = Part(0, np.array([-0.16, 0.0, 0.014]), 0.0, blockers={1})
    top = Part(1, np.array([-0.16, 0.01, 0.022]), 0.0)
    plan = TaskPlan(
        segments=[],
        joint_names=[],
        pick_order={},
        leftover=[],
        pick_times=times,
        num_motions=0,
        planning_time=0.0,
        execution_time=0.0,
        min_arm_clearance=0.0,
        max_tool_detour=0.0,
    )
    assert lifted_in_pile_order(plan, [bottom, top]) == in_order
