from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .pose_utils import get_pose_command_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def goal_reached_success(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    pos_threshold_m: float = 0.35,
    yaw_threshold_rad: float = 0.35,
    min_hold_s: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Success condition for the Go2 task (boolean per env).

    Success when:
    - planar distance to pose target <= pos_threshold_m
    - yaw error to commanded heading <= yaw_threshold_rad
    and the condition holds for at least min_hold_s to avoid single-step flicker.
    """
    target_pos_2d, target_heading = get_pose_command_world(env, command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    asset: Articulation = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    dist = torch.linalg.norm(pos_xy - target_pos_2d, dim=-1)

    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_err = torch.abs(_wrap_to_pi(yaw - target_heading))

    cond = (dist <= pos_threshold_m) & (yaw_err <= yaw_threshold_rad)

    # Hold-time filter (buffered on env instance to keep it local to this task)
    hold_steps = int(max(1.0, float(min_hold_s) / float(env.step_dt)))
    buf_name = "_go2_success_hold_count"
    if not hasattr(env, buf_name) or getattr(env, buf_name) is None:
        setattr(env, buf_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.int32))
    hold = getattr(env, buf_name)

    hold = torch.where(cond, hold + 1, torch.zeros_like(hold))
    setattr(env, buf_name, hold)

    return hold >= hold_steps

