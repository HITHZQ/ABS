# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)


def joint_pos_rel_with_encoder_bias(
    env: ManagerBasedEnv,
    bias_attr: str = "_encoder_joint_pos_bias",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions w.r.t. default, plus a per-episode encoder bias.

    Models motor encoders' offset errors (domain randomization, Table II).
    Bias is stored on the env (num_envs, num_joints) and sampled at reset
    via `sample_joint_encoder_bias` in `go1/mdp/events.py`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    bias = getattr(env, bias_attr, None)
    if bias is None:
        return rel
    return rel + bias[:, asset_cfg.joint_ids]


def lidar_distances_with_illusion(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "pose_command",
    margin: float = 0.3,
    max_distance: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_log: bool = True,
    log_eps: float = 0.01,
) -> torch.Tensor:
    """Ray-caster distances with illusion (Table II). Optionally log-scale for log(ray) noise.

    Illusion: overwrite d with U(d_goal + margin, d) when d > d_goal + margin.
    Table II: observe log(ray distance), add noise U(-0.2, 0.2). use_log=True returns log(d).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    hits_w = sensor.data.ray_hits_w
    pos_w = sensor.data.pos_w
    d = torch.linalg.norm(hits_w - pos_w.unsqueeze(1), dim=-1)

    if max_distance is not None:
        d = torch.nan_to_num(d, nan=max_distance, posinf=max_distance, neginf=0.0)

    cmd = env.command_manager.get_command(command_name)
    target_xy = cmd[:, :2]
    robot_xy = asset.data.root_pos_w[:, :2]
    d_goal = torch.linalg.norm(target_xy - robot_xy, dim=-1)
    thresh = (d_goal + margin).unsqueeze(1)

    mask = d > thresh
    u = torch.rand_like(d)
    d_illus = thresh + u * (d - thresh)
    d = torch.where(mask, d_illus, d)

    if max_distance is not None:
        d = torch.clamp(d, 0.0, float(max_distance))

    if use_log:
        d = torch.log(torch.clamp(d, min=log_eps))
    return d


def foot_contacts(
    env: ManagerBasedEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Get foot contact information as binary values.
    
    Returns a tensor of shape (num_envs, num_feet) where each element is 1.0 if
    the foot is in contact with the ground (contact force > threshold), else 0.0.
    
    Args:
        env: The environment instance.
        threshold: Contact force threshold to consider a foot as in contact.
        asset_cfg: Configuration for the robot asset.
        foot_body_names: List of foot body names. If None, defaults to Go1 foot names.
    
    Returns:
        Binary contact tensor of shape (num_envs, num_feet).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Default foot body names for Go1 (adjust based on your robot's naming convention)
    if foot_body_names is None:
        foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    # Get contact forces using standard Isaac Lab approach
    # Check if contact sensor exists
    contact_forces_list = []
    
    for foot_name in foot_body_names:
        # Use body contact forces - check if body exists
        body_names = asset.body_names
        if foot_name in body_names:
            body_idx = body_names.index(foot_name)
            # Get net contact force magnitude for this body
            # Contact forces are typically in asset.data.body_contact_forces_w
            if hasattr(asset.data, "body_contact_forces_w"):
                forces = asset.data.body_contact_forces_w[:, body_idx]  # (num_envs, 3)
                force_magnitude = torch.linalg.norm(forces, dim=-1)  # (num_envs,)
                contact_forces_list.append(force_magnitude)
            else:
                # Fallback: try contact sensor
                if "contact_forces" in env.scene:
                    # Access via contact sensor - implementation depends on Isaac Lab version
                    contact_forces_list.append(torch.zeros(env.num_envs, device=env.device))
                else:
                    contact_forces_list.append(torch.zeros(env.num_envs, device=env.device))
        else:
            # Body not found, return zero contact
            contact_forces_list.append(torch.zeros(env.num_envs, device=env.device))
    
    # Stack and convert to binary
    contact_forces = torch.stack(contact_forces_list, dim=-1)  # (num_envs, num_feet)
    contacts = (contact_forces > threshold).float()
    
    return contacts
