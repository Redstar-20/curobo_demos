# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Regression tests for the KUKA KR10 R1100-2 welding torch model."""

import math

import numpy as np
import pytest
import torch

from curobo._src.util_file import get_robot_configs_path, join_path
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.robot_builder import RobotDebugger
from curobo.types import JointState, Pose

ARM = "kr10_r1100_2.yml"
WELDING = "kr10_r1100_2_welding.yml"


def _pose(config: str, frame: str, q: torch.Tensor) -> Pose:
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(config))
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names))
    pose = state.tool_poses[frame]
    return Pose(position=pose.position.view(-1, 3), quaternion=pose.quaternion.view(-1, 4))


def test_torch_tcp_sits_at_the_wire_tip():
    """The wire tip is 0.16 m along a neck bent 45 deg from the flange axis."""
    q = torch.tensor([[0.3, -1.2, 1.4, 0.5, 0.8, -0.4]], device="cuda", dtype=torch.float32)
    flange = _pose(ARM, "tool0", q)
    tcp = _pose(WELDING, "torch_tcp", q)
    local = flange.inverse().multiply(tcp)
    neck = math.radians(45.0)
    expected = [0.16 * math.sin(neck), 0.0, 0.20 + 0.16 * math.cos(neck)]
    assert local.position.view(-1).tolist() == pytest.approx(expected, abs=1e-4)
    torch_axis = local.get_rotation().view(3, 3)[:, 2].cpu().numpy()
    assert torch_axis == pytest.approx([math.sin(neck), 0.0, math.cos(neck)], abs=1e-4)


def test_torch_spheres_leave_the_tip_free():
    """No torch sphere comes closer than 1.5 cm to the wire tip, so the tip can reach a seam."""
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(WELDING))
    state = kin.compute_kinematics(kin.default_joint_state.unsqueeze(0))
    index = kin.config.kinematics_config.get_sphere_index_from_link_name("torch")
    spheres = state.robot_spheres.view(-1, 4)[index]
    tip = state.tool_poses["torch_tcp"].position.view(1, 3)
    gap = torch.linalg.norm(spheres[:, :3] - tip, dim=-1) - spheres[:, 3]
    assert float(gap.min()) >= 0.015
    assert np.isfinite(float(gap.min()))


def test_home_pose_is_collision_free():
    """The default (home) pose has no self-collision."""
    result = RobotDebugger(
        join_path(get_robot_configs_path(), WELDING)
    ).check_default_joint_configuration_collision()
    assert not result["has_collision"], result.get("colliding_pairs")
