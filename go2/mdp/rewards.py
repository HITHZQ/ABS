from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .pose_utils import get_pose_command_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining upright posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    up_proj = (-asset.data.projected_gravity_b)[:, 2]
    return (up_proj > threshold).float()


def distance_to_goal_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    sigma: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense shaping reward for approaching pose goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_2d, _ = get_pose_command_world(env, command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)
    current_pos = asset.data.root_pos_w[:, :2]
    distance = torch.linalg.norm(current_pos - target_pos_2d, dim=-1)
    return 1.0 / (1.0 + torch.square(distance / sigma))


def agile(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    vmax: float = 4.5,
    sigma_tight: float = 0.5,
    correct_direction_threshold: float = 105.0,
    vel_toward_goal_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Agile reward: fast forward progress or stay near goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_2d, _ = get_pose_command_world(env, command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)

    current_pos = asset.data.root_pos_w[:, :2]
    to_goal = target_pos_2d - current_pos
    distance_to_goal = torch.linalg.norm(to_goal, dim=-1)

    base_quat = asset.data.root_quat_w
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]
    cos_yaw = base_quat[:, 0] ** 2 + base_quat[:, 3] ** 2 - base_quat[:, 1] ** 2 - base_quat[:, 2] ** 2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    vx = base_lin_vel_w[:, 0] * cos_yaw + base_lin_vel_w[:, 1] * sin_yaw

    robot_heading_vec = torch.stack([cos_yaw, sin_yaw], dim=-1)
    to_goal_norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True)
    to_goal_dir = torch.where(to_goal_norm > 1e-6, to_goal / to_goal_norm, torch.zeros_like(to_goal))
    dot_product = torch.clamp(torch.sum(robot_heading_vec * to_goal_dir, dim=-1), -1.0, 1.0)
    angle_deg = torch.acos(dot_product) * 180.0 / torch.pi
    correct_direction_mask = angle_deg < correct_direction_threshold

    term1a = torch.clamp(vx / vmax, min=0.0) * correct_direction_mask.float()
    vel_toward_goal = torch.sum(base_lin_vel_w * to_goal_dir, dim=-1)
    term1b = vel_toward_goal_scale * torch.clamp(vel_toward_goal / vmax, min=0.0)
    term1 = torch.maximum(term1a, term1b)
    term2 = (distance_to_goal < sigma_tight).float()
    return torch.maximum(term1, term2)


def backward_toward_goal_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vx_deadband: float = 0.05,
    dist_min_m: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize moving toward the goal while having negative forward velocity (i.e. backing up).

    The penalty is proportional to the positive world-frame speed component toward the goal when
    the body-frame forward velocity v_x is negative beyond a deadband. This targets the common
    shortcut where the policy approaches the target quickly by running backward.
    """
    target_pos_2d, _ = get_pose_command_world(env, command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    to_goal = target_pos_2d - pos_xy
    dist = torch.linalg.norm(to_goal, dim=-1)
    to_goal_dir = torch.nn.functional.normalize(to_goal, dim=-1, eps=eps)

    # world-frame planar velocity
    v_w = asset.data.root_lin_vel_w[:, :2]
    v_toward = torch.sum(v_w * to_goal_dir, dim=-1).clamp(min=0.0)

    # body-frame forward velocity (x)
    base_quat = asset.data.root_quat_w
    cos_yaw = base_quat[:, 0] ** 2 + base_quat[:, 3] ** 2 - base_quat[:, 1] ** 2 - base_quat[:, 2] ** 2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    vx = v_w[:, 0] * cos_yaw + v_w[:, 1] * sin_yaw

    backing = vx < -abs(float(vx_deadband))
    far_enough = dist > float(dist_min_m)
    return torch.where(backing & far_enough, v_toward, torch.zeros_like(v_toward))


def align_heading_before_motion_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dist_heading_blend_m: float = 0.75,
    min_bearing_dist_m: float = 0.12,
    planar_speed_deadband: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize planar speed when body +X is not aligned with a *smoothly blended* desired direction.

    Blends (in horizontal plane): toward-goal bearing vs pose-command heading with weight growing
    as distance shrinks, so near the goal we do not require the robot to face a direction almost
    orthogonal to the last approach (a hard switch caused local ``stay still'' optima).

    Uses ``max(0, ‖v_xy‖² - v_dead²)`` so small exploratory motion and in-place reorientation are
    not over-penalized.
    """
    target_pos_2d, target_heading = get_pose_command_world(env, command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    to_goal = target_pos_2d - pos_xy
    dist = torch.linalg.norm(to_goal, dim=-1)

    dir_from_heading = torch.stack((torch.cos(target_heading), torch.sin(target_heading)), dim=-1)
    dir_from_bearing = torch.nn.functional.normalize(to_goal, dim=-1, eps=eps)
    safe_bearing = torch.where(
        dist.unsqueeze(-1) > min_bearing_dist_m, dir_from_bearing, dir_from_heading
    )
    # w=0 far (pure bearing); w→1 near goal (more final heading), smooth in between
    w = torch.clamp((dist_heading_blend_m - dist) / max(dist_heading_blend_m, eps), 0.0, 1.0)
    dir_desired = torch.nn.functional.normalize(
        (1.0 - w).unsqueeze(-1) * safe_bearing + w.unsqueeze(-1) * dir_from_heading,
        dim=-1,
        eps=eps,
    )

    forward_b = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, forward_b)
    forward_xy = torch.nn.functional.normalize(forward_w[:, :2], dim=-1, eps=eps)

    cos_align = torch.sum(forward_xy * dir_desired, dim=-1).clamp(-1.0, 1.0)
    misalign = 0.5 * (1.0 - cos_align)
    v_xy_sq = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)
    dead = max(float(planar_speed_deadband), 0.0)
    v_excess_sq = torch.clamp(v_xy_sq - dead * dead, min=0.0)
    return misalign * v_excess_sq


def track_lin_vel_xy_near_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    pose_command_name: str = "pose_command",
    std: float = 0.5,
    goal_radius: float = 3.5,
    blend: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Go2 速度跟踪 × go1 定点门控：离目标较远时接近 0，接近目标点后按 base_velocity 强化快速运动。"""
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as vel_mdp

    r = vel_mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std)
    target_pos_2d, _ = get_pose_command_world(env, pose_command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    dist = torch.linalg.norm(asset.data.root_pos_w[:, :2] - target_pos_2d, dim=-1)
    gate = torch.sigmoid((goal_radius - dist) / max(blend, 1e-3))
    return r * gate


def track_ang_vel_z_near_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    pose_command_name: str = "pose_command",
    std: float = 0.5,
    goal_radius: float = 3.5,
    blend: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """角速度跟踪仅在接近 pose 目标时生效，与 track_lin_vel_xy_near_goal 配套。"""
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as vel_mdp

    r = vel_mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std)
    target_pos_2d, _ = get_pose_command_world(env, pose_command_name)
    if target_pos_2d is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    dist = torch.linalg.norm(asset.data.root_pos_w[:, :2] - target_pos_2d, dim=-1)
    gate = torch.sigmoid((goal_radius - dist) / max(blend, 1e-3))
    return r * gate


"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
