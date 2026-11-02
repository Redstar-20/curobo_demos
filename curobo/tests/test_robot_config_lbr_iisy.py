# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Regression tests for the KUKA LBR iisy3 R760 collision spheres.

The default RobotBuilder fit put a base sphere 2.5 cm below the mounting face (the
robot then starts in collision with its pedestal). These tests keep a refit honest.
"""

import pytest
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.robot_builder import RobotDebugger
from curobo._src.util_file import get_robot_configs_path, join_path

SINGLE = "lbr_iisy3_r760.yml"
DUAL = "dual_lbr_iisy3_r760.yml"


def _link_spheres(kin: Kinematics, link: str) -> torch.Tensor:
    """World-frame spheres ``(n, 4)`` of one link at the default joint state."""
    kin_cfg = kin.config.kinematics_config
    state = kin.compute_kinematics(kin.default_joint_state.unsqueeze(0))
    spheres = state.robot_spheres.view(-1, 4)
    index = kin_cfg.get_sphere_index_from_link_name(link)
    spheres = spheres[index]
    return spheres[spheres[:, 3] > 0]


def test_base_spheres_stay_above_mounting_face():
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(SINGLE))
    base = _link_spheres(kin, "base_link")
    assert base.shape[0] > 0
    assert float((base[:, 2] - base[:, 3]).min()) >= 0.0


def test_default_tool_points_down():
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(SINGLE))
    state = kin.compute_kinematics(kin.default_joint_state.unsqueeze(0))
    w, x, y, z = state.tool_poses["tool0"].quaternion.view(-1).tolist()
    tool_z_world_z = 1 - 2 * (x * x + y * y)
    assert tool_z_world_z == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize("config", [SINGLE, DUAL])
def test_default_pose_is_collision_free(config: str):
    result = RobotDebugger(join_path(get_robot_configs_path(), config)).check_default_joint_configuration_collision()
    assert not result["has_collision"], result.get("colliding_pairs")
