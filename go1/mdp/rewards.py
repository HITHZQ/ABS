# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)


def linear_velocity_command_error(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    command_index: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking linear velocity command (negative L2 error).
    
    Args:
        command_name: Name of the command in the command manager.
        command_index: Index of the velocity component (0=x, 1=y).
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Negative L2 error between commanded and actual linear velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get twist command: [lin_vel_x, lin_vel_y, ang_vel_yaw]
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3)
    cmd_vel = cmd[:, command_index]  # Extract x or y component
    
    # Get actual linear velocity in base frame
    # Transform world velocity to base frame
    base_quat = asset.data.root_quat_w
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]  # Only x, y components
    
    # Rotate to base frame (2D rotation)
    cos_yaw = base_quat[:, 0]**2 + base_quat[:, 3]**2 - base_quat[:, 1]**2 - base_quat[:, 2]**2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    
    # Rotate velocity to base frame
    base_lin_vel_x = base_lin_vel_w[:, 0] * cos_yaw + base_lin_vel_w[:, 1] * sin_yaw
    base_lin_vel_y = -base_lin_vel_w[:, 0] * sin_yaw + base_lin_vel_w[:, 1] * cos_yaw
    
    # Select the appropriate component
    if command_index == 0:
        actual_vel = base_lin_vel_x
    else:
        actual_vel = base_lin_vel_y
    
    # Return negative L2 error (to be maximized as reward)
    error = cmd_vel - actual_vel
    return -torch.square(error)


def angular_velocity_command_error(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    command_index: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking angular velocity command (negative L2 error).
    
    Args:
        command_name: Name of the command in the command manager.
        command_index: Index of the angular velocity component (typically 2 for yaw).
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Negative L2 error between commanded and actual angular velocity (yaw).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get twist command: [lin_vel_x, lin_vel_y, ang_vel_yaw]
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3)
    cmd_ang_vel = cmd[:, command_index]  # Extract yaw component
    
    # Get actual angular velocity (yaw component)
    actual_ang_vel = asset.data.root_ang_vel_w[:, 2]  # yaw is z-axis rotation
    
    # Return negative L2 error (to be maximized as reward)
    error = cmd_ang_vel - actual_ang_vel
    return -torch.square(error)


def linear_velocity_tracking_reward_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_linvel: float = 0.5,
) -> torch.Tensor:
    """Reward for tracking linear velocity (vx, vy).
    r_linvel = exp(-((vx - v^c_x)^2 + (vy - v^c_y)^2) / σ²_linvel), σ_linvel = 0.5 m/s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    base_quat = asset.data.root_quat_w
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]
    cos_yaw = base_quat[:, 0] ** 2 + base_quat[:, 3] ** 2 - base_quat[:, 1] ** 2 - base_quat[:, 2] ** 2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    vx = base_lin_vel_w[:, 0] * cos_yaw + base_lin_vel_w[:, 1] * sin_yaw
    vy = -base_lin_vel_w[:, 0] * sin_yaw + base_lin_vel_w[:, 1] * cos_yaw
    ex = cmd[:, 0] - vx
    ey = cmd[:, 1] - vy
    s2 = sigma_linvel ** 2
    return torch.exp(-((ex ** 2 + ey ** 2) / s2))


def angular_velocity_tracking_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Angular velocity tracking error. r_angvel = ‖ωz − ω^c_z‖²_2 (use weight −0.5 in r_task)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    actual = asset.data.root_ang_vel_w[:, 2]
    err = actual - cmd[:, 2]
    return torch.square(err)


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 when not terminated, 0.0 when terminated. r_alive (use weight 5)."""
    m = getattr(env, "termination_manager", None)
    if m is None:
        return torch.ones(env.num_envs, device=env.device)
    t = getattr(m, "terminated", None) or getattr(m, "_terminated_buf", None)
    if t is None:
        return torch.ones(env.num_envs, device=env.device)
    return 1.0 - t.float()


def posture_penalty_recovery(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    """r_posture = ‖q − q̄_rec‖₁. q̄_rec is nominal low-height standing for seamless switch to agile (use weight −0.1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    joint_names = asset.joint_names
    if nominal_joint_pos is None:
        nominal_joint_pos = {
            "FL_hip_joint": 0.0, "RL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 1.0, "RL_thigh_joint": 1.0, "FR_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.7, "RL_calf_joint": -1.7, "FR_calf_joint": -1.7, "RR_calf_joint": -1.7,
        }
    q_bar = torch.zeros_like(q)
    default = asset.data.default_joint_pos
    for i, name in enumerate(joint_names):
        val = nominal_joint_pos.get(name)
        q_bar[:, i] = float(val) if val is not None else default[0, i]
    return torch.sum(torch.abs(q - q_bar), dim=-1)


def undesired_contacts_comprehensive(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    horizontal_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_name: str = "trunk",
    thigh_body_pattern: str = ".*thigh",
    calf_body_pattern: str = ".*calf",
    foot_body_names: list[str] | None = None,
    penalize_calf: bool = True,
) -> torch.Tensor:
    """Penalty for undesired contacts.
    
    Undesired: base, thighs, calves (optional), horizontal foot contacts.
    Set penalize_calf=False to allow knee/calf contacts (e.g. recovery policy, max deceleration).
    
    Args:
        env: The environment instance.
        threshold: Contact force threshold for base/thighs/calves.
        horizontal_threshold: Horizontal contact force threshold for feet.
        asset_cfg: Configuration for the robot asset.
        base_body_name: Name of the base/trunk body.
        thigh_body_pattern: Regex pattern for thigh body names.
        calf_body_pattern: Regex pattern for calf body names.
        foot_body_names: List of foot body names. If None, defaults to Go1 foot names.
        penalize_calf: If False, do not penalize calf/knee contacts (recovery: allow knee for deceleration).
    
    Returns:
        Sum of undesired contact forces (to be minimized as penalty).
    """
    import re
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Default foot body names for Go1
    if foot_body_names is None:
        foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    
    # Get contact forces - try different methods depending on Isaac Lab version
    try:
        # Method 1: Use contact sensor if available
        if "contact_forces" in env.scene:
            contact_sensor = env.scene["contact_forces"]
            body_names = asset.body_names
            
            # 1. Check base collisions
            if base_body_name in body_names:
                base_idx = body_names.index(base_body_name)
                base_forces = contact_sensor.data.net_forces_w_history[:, :, base_idx]  # (num_envs, history, 3)
                base_force_mag = torch.linalg.norm(base_forces, dim=-1)  # (num_envs, history)
                base_penalty = torch.max(base_force_mag, dim=-1)[0] if base_force_mag.shape[1] > 0 else base_force_mag[:, 0]
                total_penalty += torch.where(base_penalty > threshold, base_penalty, torch.zeros_like(base_penalty))
            
            # 2. Check thigh collisions
            thigh_indices = [i for i, name in enumerate(body_names) if re.match(thigh_body_pattern, name)]
            for thigh_idx in thigh_indices:
                thigh_forces = contact_sensor.data.net_forces_w_history[:, :, thigh_idx]
                thigh_force_mag = torch.linalg.norm(thigh_forces, dim=-1)
                thigh_penalty = torch.max(thigh_force_mag, dim=-1)[0] if thigh_force_mag.shape[1] > 0 else thigh_force_mag[:, 0]
                total_penalty += torch.where(thigh_penalty > threshold, thigh_penalty, torch.zeros_like(thigh_penalty))
            
            # 3. Check calf collisions (skip if penalize_calf=False, e.g. recovery allows knee contact)
            if penalize_calf:
                calf_indices = [i for i, name in enumerate(body_names) if re.match(calf_body_pattern, name)]
                for calf_idx in calf_indices:
                    calf_forces = contact_sensor.data.net_forces_w_history[:, :, calf_idx]
                    calf_force_mag = torch.linalg.norm(calf_forces, dim=-1)
                    calf_penalty = torch.max(calf_force_mag, dim=-1)[0] if calf_force_mag.shape[1] > 0 else calf_force_mag[:, 0]
                    total_penalty += torch.where(calf_penalty > threshold, calf_penalty, torch.zeros_like(calf_penalty))
            
            # 4. Check horizontal foot collisions
            for foot_name in foot_body_names:
                if foot_name in body_names:
                    foot_idx = body_names.index(foot_name)
                    foot_forces = contact_sensor.data.net_forces_w_history[:, :, foot_idx]  # (num_envs, history, 3)
                    # Get horizontal component (x, y) - ignore vertical (z)
                    foot_forces_horizontal = foot_forces[:, :, :2]  # (num_envs, history, 2)
                    foot_horizontal_mag = torch.linalg.norm(foot_forces_horizontal, dim=-1)  # (num_envs, history)
                    foot_horizontal_penalty = torch.max(foot_horizontal_mag, dim=-1)[0] if foot_horizontal_mag.shape[1] > 0 else foot_horizontal_mag[:, 0]
                    total_penalty += torch.where(foot_horizontal_penalty > horizontal_threshold, foot_horizontal_penalty, torch.zeros_like(foot_horizontal_penalty))
        
        else:
            # Method 2: Use body contact forces directly from asset data
            if hasattr(asset.data, "body_contact_forces_w"):
                body_names = asset.body_names
                body_forces = asset.data.body_contact_forces_w  # (num_envs, num_bodies, 3)
                
                # 1. Base collisions
                if base_body_name in body_names:
                    base_idx = body_names.index(base_body_name)
                    base_force_mag = torch.linalg.norm(body_forces[:, base_idx], dim=-1)
                    total_penalty += torch.where(base_force_mag > threshold, base_force_mag, torch.zeros_like(base_force_mag))
                
                # 2. Thigh collisions
                for i, name in enumerate(body_names):
                    if re.match(thigh_body_pattern, name):
                        thigh_force_mag = torch.linalg.norm(body_forces[:, i], dim=-1)
                        total_penalty += torch.where(thigh_force_mag > threshold, thigh_force_mag, torch.zeros_like(thigh_force_mag))
                
                # 3. Calf collisions (skip if penalize_calf=False)
                if penalize_calf:
                    for i, name in enumerate(body_names):
                        if re.match(calf_body_pattern, name):
                            calf_force_mag = torch.linalg.norm(body_forces[:, i], dim=-1)
                            total_penalty += torch.where(calf_force_mag > threshold, calf_force_mag, torch.zeros_like(calf_force_mag))
                
                # 4. Horizontal foot collisions
                for foot_name in foot_body_names:
                    if foot_name in body_names:
                        foot_idx = body_names.index(foot_name)
                        foot_forces_horizontal = body_forces[:, foot_idx, :2]  # (num_envs, 2)
                        foot_horizontal_mag = torch.linalg.norm(foot_forces_horizontal, dim=-1)  # (num_envs,)
                        total_penalty += torch.where(foot_horizontal_mag > horizontal_threshold, foot_horizontal_mag, torch.zeros_like(foot_horizontal_mag))
    
    except (AttributeError, KeyError, IndexError):
        try:
            from isaaclab.envs.mdp import undesired_contacts
            bodies = f"{base_body_name}|{thigh_body_pattern}"
            if penalize_calf:
                bodies += f"|{calf_body_pattern}"
            base_thigh_calf_penalty = undesired_contacts(
                env,
                threshold=threshold,
                sensor_cfg=SceneEntityCfg(asset_cfg.name, body_names=bodies),
            )
            total_penalty += base_thigh_calf_penalty
        except (ImportError, AttributeError):
            pass
    
    return total_penalty


def possoft(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sigma: float = 2.0,
    tr_steps: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft position tracking reward to encourage exploration for goal reaching.
    
    Formula: r(possoft) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr)
    
    This reward encourages the robot to reach the goal before T - Tr to maximize
    tracking rewards, freeing it from explicit motion constraints such as target
    velocities that may limit agility.
    
    Where:
    - error: position error (distance to goal in meters)
    - sigma: normalization parameter for tracking errors (in meters, default: 2.0 m)
    - Tr: time threshold in seconds (default: 2.0 s)
    - T: episode length in seconds
    - t: current time in episode
    - Only active when t > T - Tr (last Tr seconds of episode)
    
    For soft position tracking:
    - σsoft = 2 m (normalizes the tracking errors)
    - Tr = 2 s (time threshold)
    - Error is the distance to the goal
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        sigma: Normalization parameter for error (in meters). Default: 2.0 m.
        tr_steps: Time threshold Tr in seconds. Default: 2.0 s.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Soft position tracking reward (scaled by 1/Tr, only active in last Tr seconds).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get target position from command
    # For pose commands, format is typically [x, y, heading] or [x, y, z, heading]
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    
    # Extract target x, y (first two components)
    if cmd.shape[1] >= 2:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
    else:
        target_pos_2d = cmd[:, :2]  # Fallback
    
    # Get current robot position (x, y only for 2D navigation)
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute position error
    error_2d = current_pos - target_pos_2d  # (num_envs, 2)
    error_norm = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Soft function: 1/(1 + (error/sigma)^2)
    soft_reward = 1.0 / (1.0 + torch.square(error_norm / sigma))
    
    # Get episode time information
    # Check if we're in the last Tr seconds of the episode
    try:
        # Get episode length from termination config
        episode_length_s = 9.0  # Default
        if hasattr(env, "cfg") and hasattr(env.cfg, "terminations"):
            if hasattr(env.cfg.terminations, "time_out"):
                if hasattr(env.cfg.terminations.time_out, "params"):
                    episode_length_s = env.cfg.terminations.time_out.params.get("time_out", 9.0)
        
        # Get current time in episode
        # In Isaac Lab, episode_length_buf typically counts remaining steps
        current_time_s = None
        if hasattr(env, "episode_length_buf"):
            # episode_length_buf is remaining steps, so current time = (max - remaining) * dt
            remaining_steps = env.episode_length_buf.float()  # (num_envs,)
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        elif hasattr(env, "_episode_length_buf"):
            remaining_steps = env._episode_length_buf.float()
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        
        if current_time_s is not None:
            # Check if t > T - Tr (in last Tr seconds)
            time_threshold = episode_length_s - tr_steps
            time_mask = current_time_s > time_threshold
            
            # Scale by 1/Tr and apply time mask
            scaled_reward = soft_reward * (1.0 / tr_steps) * time_mask.float()
        else:
            # Fallback: apply reward without time restriction (scaled by 1/Tr)
            scaled_reward = soft_reward * (1.0 / tr_steps)
    except (AttributeError, KeyError, TypeError):
        # Fallback: apply reward without time restriction (scaled by 1/Tr)
        scaled_reward = soft_reward * (1.0 / tr_steps)
    
    return scaled_reward


def postight(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sigma: float = 0.5,
    tr_steps: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tight position tracking reward for precise goal reaching.
    
    Formula: r(postight) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr)
    
    This reward provides tighter position tracking with smaller sigma and shorter
    time window compared to possoft, encouraging more precise goal reaching.
    
    Where:
    - error: position error (distance to goal in meters)
    - sigma: normalization parameter for tracking errors (in meters, default: 0.5 m)
    - Tr: time threshold in seconds (default: 1.0 s)
    - T: episode length in seconds
    - t: current time in episode
    - Only active when t > T - Tr (last Tr seconds of episode)
    
    For tight position tracking:
    - σtight = 0.5 m (normalizes the tracking errors, tighter than possoft)
    - Tr = 1 s (time threshold, shorter than possoft)
    - Error is the distance to the goal
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        sigma: Normalization parameter for error (in meters). Default: 0.5 m.
        tr_steps: Time threshold Tr in seconds. Default: 1.0 s.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Tight position tracking reward (scaled by 1/Tr, only active in last Tr seconds).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get target position from command
    # For pose commands, format is typically [x, y, heading] or [x, y, z, heading]
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    
    # Extract target x, y (first two components)
    if cmd.shape[1] >= 2:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
    else:
        target_pos_2d = cmd[:, :2]  # Fallback
    
    # Get current robot position (x, y only for 2D navigation)
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute position error
    error_2d = current_pos - target_pos_2d  # (num_envs, 2)
    error_norm = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Soft function: 1/(1 + (error/sigma)^2)
    # With smaller sigma (0.5 m), this provides tighter tracking
    soft_reward = 1.0 / (1.0 + torch.square(error_norm / sigma))
    
    # Get episode time information
    # Check if we're in the last Tr seconds of the episode
    try:
        # Get episode length from termination config
        episode_length_s = 9.0  # Default
        if hasattr(env, "cfg") and hasattr(env.cfg, "terminations"):
            if hasattr(env.cfg.terminations, "time_out"):
                if hasattr(env.cfg.terminations.time_out, "params"):
                    episode_length_s = env.cfg.terminations.time_out.params.get("time_out", 9.0)
        
        # Get current time in episode
        # In Isaac Lab, episode_length_buf typically counts remaining steps
        current_time_s = None
        if hasattr(env, "episode_length_buf"):
            # episode_length_buf is remaining steps, so current time = (max - remaining) * dt
            remaining_steps = env.episode_length_buf.float()  # (num_envs,)
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        elif hasattr(env, "_episode_length_buf"):
            remaining_steps = env._episode_length_buf.float()
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        
        if current_time_s is not None:
            # Check if t > T - Tr (in last Tr seconds)
            time_threshold = episode_length_s - tr_steps
            time_mask = current_time_s > time_threshold
            
            # Scale by 1/Tr and apply time mask
            scaled_reward = soft_reward * (1.0 / tr_steps) * time_mask.float()
        else:
            # Fallback: apply reward without time restriction (scaled by 1/Tr)
            scaled_reward = soft_reward * (1.0 / tr_steps)
    except (AttributeError, KeyError, TypeError):
        # Fallback: apply reward without time restriction (scaled by 1/Tr)
        scaled_reward = soft_reward * (1.0 / tr_steps)
    
    return scaled_reward


def heading(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sigma: float = 1.0,
    tr_steps: float = 2.0,
    sigma_soft: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Heading tracking reward for goal orientation.
    
    Formula: r(heading) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr) * (if dist <= σsoft)
    
    This reward tracks the relative yaw angle to the goal heading. It is disabled
    when the distance to the goal is larger than σsoft so that collision avoidance
    is not affected.
    
    Where:
    - error: relative yaw angle to goal heading (in radians)
    - sigma: normalization parameter for tracking errors (in radians, default: 1.0 rad)
    - Tr: time threshold in seconds (default: 2.0 s)
    - T: episode length in seconds
    - t: current time in episode
    - σsoft: distance threshold to disable heading reward (default: 2.0 m)
    - Only active when t > T - Tr (last Tr seconds) AND distance <= σsoft
    
    For heading tracking:
    - σheading = 1 rad (normalizes the tracking errors)
    - Tr = 2 s (time threshold)
    - Error: relative yaw angle to the goal heading
    - Disabled when distance to goal > σsoft (2 m)
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        sigma: Normalization parameter for heading error (in radians). Default: 1.0 rad.
        tr_steps: Time threshold Tr in seconds. Default: 2.0 s.
        sigma_soft: Distance threshold to disable heading reward (in meters). Default: 2.0 m.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Heading tracking reward (scaled by 1/Tr, only active in last Tr seconds and when close to goal).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get command (pose command: [x, y, heading] or [x, y, z, heading])
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    
    # Extract target position and heading
    if cmd.shape[1] >= 3:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
        target_heading = cmd[:, 2] if cmd.shape[1] == 3 else cmd[:, 3]  # heading is 3rd or 4th component
    else:
        target_pos_2d = cmd[:, :2]
        target_heading = torch.zeros(env.num_envs, device=env.device)
    
    # Get current robot position and heading
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute distance to goal
    error_2d = current_pos - target_pos_2d
    distance_to_goal = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Get current robot yaw angle
    _, _, current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)  # (num_envs,)
    
    # Compute relative yaw angle to goal heading
    # The error is the difference between current yaw and target heading
    heading_error = current_yaw - target_heading
    
    # Normalize to [-pi, pi]
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    
    # Take absolute value for error magnitude
    heading_error_abs = torch.abs(heading_error)  # (num_envs,)
    
    # Soft function: 1/(1 + (error/sigma)^2)
    soft_reward = 1.0 / (1.0 + torch.square(heading_error_abs / sigma))
    
    # Get episode time information
    # Check if we're in the last Tr seconds of the episode
    try:
        # Get episode length from termination config
        episode_length_s = 9.0  # Default
        if hasattr(env, "cfg") and hasattr(env.cfg, "terminations"):
            if hasattr(env.cfg.terminations, "time_out"):
                if hasattr(env.cfg.terminations.time_out, "params"):
                    episode_length_s = env.cfg.terminations.time_out.params.get("time_out", 9.0)
        
        # Get current time in episode
        current_time_s = None
        if hasattr(env, "episode_length_buf"):
            remaining_steps = env.episode_length_buf.float()  # (num_envs,)
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        elif hasattr(env, "_episode_length_buf"):
            remaining_steps = env._episode_length_buf.float()
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        
        if current_time_s is not None:
            # Check if t > T - Tr (in last Tr seconds)
            time_threshold = episode_length_s - tr_steps
            time_mask = current_time_s > time_threshold
        else:
            time_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    except (AttributeError, KeyError, TypeError):
        # Fallback: assume we're in the time window
        time_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    
    # Disable heading reward when distance to goal > σsoft
    # This ensures collision avoidance is not affected
    distance_mask = distance_to_goal <= sigma_soft
    
    # Combine masks: must be in time window AND close to goal
    combined_mask = time_mask & distance_mask
    
    # Scale by 1/Tr and apply combined mask
    scaled_reward = soft_reward * (1.0 / tr_steps) * combined_mask.float()
    
    return scaled_reward


def stand(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    tr_stand: float = 1.0,
    sigma_tight: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    """Stand reward to encourage maintaining standing pose when close to goal.
    
    Formula: r(stand) = ||q - q̄||₁ * (1(if t > T - Tr,stand) / Tr,stand) * 1(d_goal < σtight)
    
    This reward penalizes deviation from nominal standing joint positions when the robot
    is very close to the goal in the final seconds, encouraging a stable standing pose.
    
    Where:
    - q: current joint positions
    - q̄: nominal joint positions for standing
    - ||q - q̄||₁: L1 norm (sum of absolute differences)
    - Tr,stand: time threshold in seconds (default: 1.0 s)
    - T: episode length in seconds
    - t: current time in episode
    - d_goal: distance to goal
    - σtight: distance threshold (default: 0.5 m, same as postight)
    - Only active when t > T - Tr,stand (last Tr,stand seconds) AND d_goal < σtight
    
    For stand reward:
    - Tr,stand = 1 s (time threshold)
    - σtight = 0.5 m (distance threshold, same as postight)
    - q̄: nominal joint positions for standing
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        tr_stand: Time threshold Tr,stand in seconds. Default: 1.0 s.
        sigma_tight: Distance threshold σtight in meters. Default: 0.5 m.
        asset_cfg: Configuration for the robot asset.
        nominal_joint_pos: Dictionary of nominal joint positions for standing.
                          If None, uses default Go1 standing pose.
    
    Returns:
        Stand penalty (L1 norm of joint position error, scaled by 1/Tr,stand,
        only active in last Tr,stand seconds and when very close to goal).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Default nominal joint positions for Go1 standing pose
    if nominal_joint_pos is None:
        nominal_joint_pos = {
            "FL_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
    
    # Get current joint positions
    current_joint_pos = asset.data.joint_pos  # (num_envs, num_joints)
    
    # Get nominal joint positions as tensor
    joint_names = asset.joint_names
    nominal_pos_tensor = torch.zeros_like(current_joint_pos)
    
    for i, joint_name in enumerate(joint_names):
        if joint_name in nominal_joint_pos:
            nominal_pos_tensor[:, i] = nominal_joint_pos[joint_name]
        else:
            # Use default joint position if not specified
            nominal_pos_tensor[:, i] = asset.data.default_joint_pos[:, i]
    
    # Compute L1 norm: ||q - q̄||₁
    joint_error = current_joint_pos - nominal_pos_tensor  # (num_envs, num_joints)
    l1_norm = torch.sum(torch.abs(joint_error), dim=-1)  # (num_envs,)
    
    # Get distance to goal
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    if cmd.shape[1] >= 2:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
    else:
        target_pos_2d = cmd[:, :2]
    
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    error_2d = current_pos - target_pos_2d
    distance_to_goal = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Get episode time information
    # Check if we're in the last Tr,stand seconds of the episode
    try:
        # Get episode length from termination config
        episode_length_s = 9.0  # Default
        if hasattr(env, "cfg") and hasattr(env.cfg, "terminations"):
            if hasattr(env.cfg.terminations, "time_out"):
                if hasattr(env.cfg.terminations.time_out, "params"):
                    episode_length_s = env.cfg.terminations.time_out.params.get("time_out", 9.0)
        
        # Get current time in episode
        current_time_s = None
        if hasattr(env, "episode_length_buf"):
            remaining_steps = env.episode_length_buf.float()  # (num_envs,)
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        elif hasattr(env, "_episode_length_buf"):
            remaining_steps = env._episode_length_buf.float()
            max_steps = env.max_episode_length
            current_time_s = (max_steps - remaining_steps) * env.step_dt
        
        if current_time_s is not None:
            # Check if t > T - Tr,stand (in last Tr,stand seconds)
            time_threshold = episode_length_s - tr_stand
            time_mask = current_time_s > time_threshold
        else:
            time_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    except (AttributeError, KeyError, TypeError):
        # Fallback: assume we're in the time window
        time_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    
    # Only active when distance to goal < σtight
    distance_mask = distance_to_goal < sigma_tight
    
    # Combine masks: must be in time window AND very close to goal
    combined_mask = time_mask & distance_mask
    
    # Scale by 1/Tr,stand and apply combined mask
    scaled_penalty = l1_norm * (1.0 / tr_stand) * combined_mask.float()
    
    return scaled_penalty


def agile(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    vmax: float = 4.5,
    sigma_tight: float = 0.5,
    correct_direction_threshold: float = 105.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Agile reward to encourage fast forward motion or staying at goal.
    
    Formula: r(agile) = max{relu(vx/vmax) * 1(correct direction), 1(d_goal < σtight)}
    
    This reward encourages the robot to either:
    1. Run fast forward in the correct direction (toward goal)
    2. Stay at the goal (when very close)
    
    Where:
    - vx: forward velocity in robot base frame (m/s)
    - vmax: upper bound of forward velocity (default: 4.5 m/s)
    - relu(vx/vmax): normalized forward velocity (clamped to [0, 1])
    - correct direction: angle between robot heading and robot-goal line < threshold (default: 105°)
    - σtight: distance threshold (default: 0.5 m, same as postight)
    - d_goal: distance to goal
    
    To maximize this term, the robot has to either run fast or stay at the goal.
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        vmax: Maximum forward velocity in m/s. Default: 4.5 m/s.
        sigma_tight: Distance threshold σtight in meters. Default: 0.5 m.
        correct_direction_threshold: Angle threshold in degrees. Default: 105°.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Agile reward (maximum of fast forward motion reward or goal proximity reward).
    """
    import math
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get command to extract goal position
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    if cmd.shape[1] >= 2:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
    else:
        target_pos_2d = cmd[:, :2]
    
    # Get current robot position
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute distance to goal
    error_2d = current_pos - target_pos_2d
    distance_to_goal = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Get forward velocity vx in robot base frame
    base_quat = asset.data.root_quat_w
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]  # Only x, y components in world frame
    
    # Rotate world velocity to base frame (2D rotation)
    cos_yaw = base_quat[:, 0]**2 + base_quat[:, 3]**2 - base_quat[:, 1]**2 - base_quat[:, 2]**2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    
    # Rotate velocity to base frame
    base_lin_vel_x = base_lin_vel_w[:, 0] * cos_yaw + base_lin_vel_w[:, 1] * sin_yaw
    vx = base_lin_vel_x  # Forward velocity in base frame (num_envs,)
    
    # Compute angle between robot heading and robot-goal line
    # Robot heading direction (forward in base frame, transformed to world)
    robot_heading_vec = torch.stack([cos_yaw, sin_yaw], dim=-1)  # (num_envs, 2) - forward direction in world
    
    # Robot-goal direction vector
    to_goal_vec = target_pos_2d - current_pos  # (num_envs, 2)
    to_goal_norm = torch.linalg.norm(to_goal_vec, dim=-1, keepdim=True)  # (num_envs, 1)
    to_goal_dir = torch.where(
        to_goal_norm > 1e-6,
        to_goal_vec / to_goal_norm,
        torch.zeros_like(to_goal_vec)
    )  # (num_envs, 2)
    
    # Compute angle between robot heading and goal direction
    # Dot product gives cos(angle)
    dot_product = torch.sum(robot_heading_vec * to_goal_dir, dim=-1)  # (num_envs,)
    # Clamp to [-1, 1] for numerical stability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle_rad = torch.acos(dot_product)  # (num_envs,) - angle in radians
    angle_deg = angle_rad * 180.0 / math.pi  # (num_envs,) - angle in degrees
    
    # Check if correct direction: angle < threshold (105°)
    correct_direction_mask = angle_deg < correct_direction_threshold  # (num_envs,)
    
    # Term 1: relu(vx/vmax) * 1(correct direction)
    # relu(vx/vmax) = max(0, vx/vmax)
    normalized_vx = vx / vmax  # (num_envs,)
    relu_vx = torch.clamp(normalized_vx, min=0.0)  # relu: max(0, vx/vmax)
    term1 = relu_vx * correct_direction_mask.float()  # (num_envs,)
    
    # Term 2: 1(d_goal < σtight)
    term2 = (distance_to_goal < sigma_tight).float()  # (num_envs,)
    
    # Take maximum of the two terms
    agile_reward = torch.maximum(term1, term2)  # (num_envs,)
    
    return agile_reward


def stall(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sigma_soft: float = 2.0,
    velocity_threshold: float = 0.1,
    correct_direction_threshold: float = 105.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stall penalty to penalize robot for time waste.
    
    Formula: r(stall) = 1 if (robot is static) AND (d_goal > σsoft) AND (not correct direction)
    
    This term penalizes the robot for staying static when:
    - Far from goal (d_goal > σsoft = 2 m)
    - Not heading toward goal (angle >= 105°)
    
    This prevents the robot from wasting time by stalling when it should be moving.
    
    Where:
    - static: robot velocity below threshold (default: 0.1 m/s)
    - d_goal: distance to goal
    - σsoft: distance threshold (default: 2.0 m)
    - correct direction: angle between robot heading and robot-goal line < threshold (default: 105°)
    - NOT correct direction: angle >= threshold
    
    Args:
        env: The environment instance.
        command_name: Name of the command (pose_command).
        sigma_soft: Distance threshold σsoft in meters. Default: 2.0 m.
        velocity_threshold: Velocity threshold to consider robot as static (m/s). Default: 0.1 m/s.
        correct_direction_threshold: Angle threshold in degrees. Default: 105°.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Stall penalty (1.0 if stalling conditions met, 0.0 otherwise).
    """
    import math
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get command to extract goal position
    cmd = env.command_manager.get_command(command_name)  # shape (N, 3) or (N, 4)
    if cmd.shape[1] >= 2:
        target_pos_2d = cmd[:, :2]  # (num_envs, 2)
    else:
        target_pos_2d = cmd[:, :2]
    
    # Get current robot position
    current_pos = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute distance to goal
    error_2d = current_pos - target_pos_2d
    distance_to_goal = torch.linalg.norm(error_2d, dim=-1)  # (num_envs,)
    
    # Check if robot is static (low velocity)
    # Get linear velocity magnitude in world frame
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]  # (num_envs, 2) - x, y components
    velocity_magnitude = torch.linalg.norm(base_lin_vel_w, dim=-1)  # (num_envs,)
    is_static = velocity_magnitude < velocity_threshold  # (num_envs,)
    
    # Check if distance to goal > σsoft
    is_far_from_goal = distance_to_goal > sigma_soft  # (num_envs,)
    
    # Check if NOT in correct direction (angle >= threshold)
    base_quat = asset.data.root_quat_w
    
    # Robot heading direction (forward in base frame, transformed to world)
    cos_yaw = base_quat[:, 0]**2 + base_quat[:, 3]**2 - base_quat[:, 1]**2 - base_quat[:, 2]**2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    robot_heading_vec = torch.stack([cos_yaw, sin_yaw], dim=-1)  # (num_envs, 2)
    
    # Robot-goal direction vector
    to_goal_vec = target_pos_2d - current_pos  # (num_envs, 2)
    to_goal_norm = torch.linalg.norm(to_goal_vec, dim=-1, keepdim=True)  # (num_envs, 1)
    to_goal_dir = torch.where(
        to_goal_norm > 1e-6,
        to_goal_vec / to_goal_norm,
        torch.zeros_like(to_goal_vec)
    )  # (num_envs, 2)
    
    # Compute angle between robot heading and goal direction
    dot_product = torch.sum(robot_heading_vec * to_goal_dir, dim=-1)  # (num_envs,)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle_rad = torch.acos(dot_product)  # (num_envs,) - angle in radians
    angle_deg = angle_rad * 180.0 / math.pi  # (num_envs,) - angle in degrees
    
    # NOT in correct direction: angle >= threshold (105°)
    not_correct_direction = angle_deg >= correct_direction_threshold  # (num_envs,)
    
    # Stall penalty: 1 if all conditions are met
    # (static) AND (far from goal) AND (not correct direction)
    stall_penalty = (is_static & is_far_from_goal & not_correct_direction).float()  # (num_envs,)

    return stall_penalty


class regularization_reward(ManagerTermBase):
    """Combined regularization reward per the paper formula.

    r(regularization) =
      - 2 * v_z^2
      - 0.05 * (omega_x^2 + omega_y^2)
      - 20 * (g_x^2 + g_y^2)
      - 0.0005 * ||tau||_2^2
      - 20 * sum_i ReLU(|tau_i| - 0.85 * tau_i,lim)
      - 0.0005 * ||q_dot||_2^2
      - 20 * sum_i ReLU(|q_dot_i| - 0.9 * q_dot_i,lim)
      - 20 * sum_i ReLU(|q_i - q_center_i| - 0.95 * q_i,lim)
      - 2e-7 * ||q_ddot||_2^2
      - 4e-6 * ||a_dot||_2^2
      - 20 * 1(fly)

    where tau = joint torques, tau_lim = torque limits, q_dot_lim = velocity limits,
    q_lim = joint position half-range, and 1(fly) = 1 when no foot contact.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg
        n = env.num_envs
        j = asset.num_joints
        self.nj = j
        self.device = env.device

        # Torque limits (effort limits): (num_joints,) or (1, num_joints)
        tau_lim = asset.data.joint_effort_limits
        if tau_lim.dim() == 1:
            tau_lim = tau_lim.unsqueeze(0).expand(n, -1)
        self.tau_lim = tau_lim.clamp(min=1e-6)

        # Velocity limits: use same for all joints from params or actuator
        vlim = cfg.params.get("velocity_limit", 31.4159)
        self.qd_lim = torch.full((n, j), vlim, device=self.device)

        # Position limits: half-range from soft limits, center
        low = asset.data.soft_joint_pos_limits[..., 0]
        high = asset.data.soft_joint_pos_limits[..., 1]
        if low.dim() == 1:
            low = low.unsqueeze(0).expand(n, -1)
            high = high.unsqueeze(0).expand(n, -1)
        self.q_center = (low + high) * 0.5
        self.q_lim = ((high - low) * 0.5).clamp(min=1e-6)

        # Buffers for action rate and joint acceleration
        self.prev_action = torch.zeros(n, j, device=self.device)
        self.prev_joint_vel = torch.zeros(n, j, device=self.device)

        self.foot_body_names = cfg.params.get(
            "foot_body_names", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        )
        self.contact_threshold = cfg.params.get("contact_threshold", 1.0)

    def reset(self, env_ids: torch.Tensor):
        self.prev_action[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        velocity_limit: float = 31.4159,
        foot_body_names: list[str] | None = None,
        contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        j = self.nj
        dt = env.step_dt

        # Base linear velocity z
        v_z = asset.data.root_lin_vel_w[:, 2]
        term_vz = -2.0 * (v_z ** 2)

        # Base angular velocity x, y
        omega_x = asset.data.root_ang_vel_w[:, 0]
        omega_y = asset.data.root_ang_vel_w[:, 1]
        term_omega = -0.05 * (omega_x ** 2 + omega_y ** 2)

        # Projected gravity in base frame (g_x, g_y)
        grav_b = asset.data.projected_gravity_b
        g_xy_sq = (grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2)
        term_grav = -20.0 * g_xy_sq

        # Joint torques: use applied_torque if available, else fallback to zero
        tau = getattr(asset.data, "applied_torque", None)
        if tau is None:
            tau = getattr(asset.data, "joint_effort", None)
        if tau is None:
            tau = torch.zeros(env.num_envs, j, device=self.device)
        else:
            if tau.dim() == 1:
                tau = tau.unsqueeze(0).expand(env.num_envs, -1)
            ntau = tau.shape[-1]
            if ntau > j:
                tau = tau[..., :j].contiguous()
            elif ntau < j:
                pad = torch.zeros(env.num_envs, j - ntau, device=self.device, dtype=tau.dtype)
                tau = torch.cat([tau, pad], dim=-1)

        tau_lim = self.tau_lim
        if tau_lim.shape[0] != env.num_envs:
            tau_lim = tau_lim.expand(env.num_envs, -1)
        term_tau_l2 = -0.0005 * (tau ** 2).sum(dim=-1)
        tau_viol = torch.relu(torch.abs(tau) - 0.85 * tau_lim).sum(dim=-1)
        term_tau_viol = -20.0 * tau_viol

        # Joint velocities
        qd = asset.data.joint_vel[..., :j]
        qd_lim = self.qd_lim
        if qd_lim.shape[0] != env.num_envs:
            qd_lim = qd_lim.expand(env.num_envs, -1)
        term_qd_l2 = -0.0005 * (qd ** 2).sum(dim=-1)
        qd_viol = torch.relu(torch.abs(qd) - 0.9 * qd_lim).sum(dim=-1)
        term_qd_viol = -20.0 * qd_viol

        # Joint positions (deviation from center, relative to half-range)
        q = asset.data.joint_pos[..., :j]
        qc = self.q_center
        ql = self.q_lim
        if qc.shape[0] != env.num_envs:
            qc = qc.expand(env.num_envs, -1)
            ql = ql.expand(env.num_envs, -1)
        q_dev = torch.abs(q - qc)
        q_pos_viol = torch.relu(q_dev - 0.95 * ql).sum(dim=-1)
        term_q_viol = -20.0 * q_pos_viol

        # Joint acceleration from finite difference
        qdd = (qd - self.prev_joint_vel) / dt
        self.prev_joint_vel.copy_(qd)
        term_qdd_l2 = -2.0e-7 * (qdd ** 2).sum(dim=-1)

        # Action rate
        action = env.action_manager.action
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(env.num_envs, -1)
        na = action.shape[-1]
        if na > j:
            action = action[..., :j].contiguous()
        elif na < j:
            pad = torch.zeros(env.num_envs, j - na, device=self.device, dtype=action.dtype)
            action = torch.cat([action, pad], dim=-1)
        a_dot = (action - self.prev_action) / dt
        self.prev_action.copy_(action)
        term_adot_l2 = -4.0e-6 * (a_dot ** 2).sum(dim=-1)

        # Fly: no foot contact
        contacts = obs.foot_contacts(
            env,
            threshold=self.contact_threshold,
            asset_cfg=asset_cfg,
            foot_body_names=self.foot_body_names,
        )
        any_contact = (contacts.sum(dim=-1) > 0.0).float()
        fly = 1.0 - any_contact
        term_fly = -20.0 * fly

        r = (
            term_vz + term_omega + term_grav
            + term_tau_l2 + term_tau_viol
            + term_qd_l2 + term_qd_viol
            + term_q_viol
            + term_qdd_l2 + term_adot_l2
            + term_fly
        )
        return r
