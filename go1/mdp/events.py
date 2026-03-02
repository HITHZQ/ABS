# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim.schemas as sim_schemas
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_cylinders_kinematic_at_startup(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    obstacle_names: list[str] | None = None,
) -> None:
    """Ensure all cylinder obstacles are kinematic (fixed) from the start. Call at startup."""
    if obstacle_names is None:
        obstacle_names = []
    if isinstance(obstacle_names, str):
        obstacle_names = [obstacle_names]
    for name in obstacle_names:
        if not isinstance(name, str) or not name.startswith("cylinder"):
            continue
        try:
            asset = env.scene[name]
        except KeyError:
            continue
        if not isinstance(asset, RigidObject):
            continue
        if not hasattr(asset, "root_physx_view"):
            continue
        cfg = sim_schemas.schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        for path in asset.root_physx_view.prim_paths:
            sim_schemas.modify_rigid_body_properties(path, cfg)


def reset_cylinder_obstacles_from_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    buffer_attr: str = "_test_obstacle_positions",
    default_z: float = 0.25,
) -> None:
    """Reset cylinder obstacles from pre-computed positions stored on env.

    Reads from env.<buffer_attr> of shape (num_envs, 2, num_obstacles) where
    [:,0,:] = x, [:,1,:] = y in env local frame. Used for testbed evaluation.
    """
    if len(obstacle_names) == 0:
        return

    positions = getattr(env, buffer_attr, None)
    if positions is None or positions.shape[2] < len(obstacle_names):
        return

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    zeros_6 = torch.zeros((len(env_ids), 6), device=env.device)

    for i, name in enumerate(obstacle_names):
        if i >= positions.shape[2]:
            break
        asset: RigidObject = env.scene[name]
        base_pos = env.scene.env_origins[env_ids].clone()
        x = positions[env_ids, 0, i]
        y = positions[env_ids, 1, i]
        z = torch.full((len(env_ids),), default_z, device=env.device, dtype=torch.float32)
        pose = torch.cat(
            [base_pos + torch.stack([x, y, z], dim=-1), identity_quat],
            dim=-1,
        )
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zeros_6, env_ids=env_ids)


def reset_cylinder_obstacles_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_levels: int,
    max_obstacles: int,
    out_of_bounds_offset: tuple[float, float, float] = (1000.0, 1000.0, -10.0),
    min_obstacles: int = 0,
    grid_nx: int = 10,
    grid_ny: int = 5,
) -> None:
    """Reset cylinder obstacles with a difficulty curriculum tied to terrain levels.

    For each environment, sample a number of active obstacles in [0, max_obstacles_for_level],
    where max_obstacles_for_level increases with terrain level (0..num_levels-1).

    If min_obstacles > 0 (e.g. for play/eval), at least that many obstacles are shown per env.

    Obstacles are placed on a grid of slots so that no two cylinders overlap in the same env:
    grid_nx * grid_ny slots in [x_range] x [y_range]; each env gets num_active distinct slots.
    """
    if len(obstacle_names) == 0:
        return

    # Resolve terrain levels (0..num_levels-1). If terrain isn't present, fall back to zeros.
    terrain_levels = None
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        terrain_levels = env.scene.terrain.terrain_levels

    if terrain_levels is None:
        levels = torch.zeros((len(env_ids),), dtype=torch.long, device=env.device)
    else:
        levels = terrain_levels[env_ids].to(dtype=torch.long)
        levels = torch.clamp(levels, 0, max(0, num_levels - 1))

    # Compute per-env max obstacles increasing with difficulty.
    denom = max(1, num_levels - 1)
    max_for_level = torch.floor(levels.to(torch.float32) / float(denom) * float(max_obstacles)).to(torch.long)
    max_for_level = torch.clamp(max_for_level, 0, max_obstacles)

    num_active = torch.zeros((len(env_ids),), dtype=torch.long, device=env.device)
    has_any = max_for_level > 0
    if torch.any(has_any):
        num_active[has_any] = torch.randint(
            low=0,
            high=(max_for_level[has_any] + 1).to(torch.long),
            size=(int(has_any.sum().item()),),
            device=env.device,
        )
    if min_obstacles > 0:
        min_n = min(min_obstacles, max_obstacles)
        num_active = torch.maximum(num_active, torch.full_like(num_active, min_n))

    # Grid of slot centers so cylinders never overlap (each env: distinct slots).
    num_slots = grid_nx * grid_ny
    if num_slots < len(obstacle_names):
        grid_ny = max(grid_ny, (len(obstacle_names) + grid_nx - 1) // grid_nx)
        num_slots = grid_nx * grid_ny
    xs = torch.linspace(
        x_range[0] + (x_range[1] - x_range[0]) / (2 * grid_nx),
        x_range[1] - (x_range[1] - x_range[0]) / (2 * grid_nx),
        grid_nx,
        device=env.device,
    )
    ys = torch.linspace(
        y_range[0] + (y_range[1] - y_range[0]) / (2 * grid_ny),
        y_range[1] - (y_range[1] - y_range[0]) / (2 * grid_ny),
        grid_ny,
        device=env.device,
    )
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    slot_centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (num_slots, 2)

    # Per-env random permutation of slot indices so each obstacle gets a distinct slot.
    perm = torch.argsort(torch.rand(len(env_ids), num_slots, device=env.device), dim=1)  # (len(env_ids), num_slots)

    zeros_6 = torch.zeros((len(env_ids), 6), device=env.device)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    oob_offset = torch.tensor(out_of_bounds_offset, device=env.device).view(1, 3)

    for i, name in enumerate(obstacle_names):
        asset: RigidObject = env.scene[name]
        root_state = asset.data.default_root_state[env_ids].clone()
        base_pos = root_state[:, 0:3] + env.scene.env_origins[env_ids]

        # Each env gets a distinct slot for this obstacle index (no overlap).
        slot_idx = perm[:, i]  # (len(env_ids),)
        local_xy = slot_centers[slot_idx]  # (len(env_ids), 2)
        z = torch.zeros(len(env_ids), device=env.device)
        yaw = math_utils.sample_uniform(-math.pi, math.pi, (len(env_ids),), device=env.device)
        quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        pose_active = torch.cat([base_pos + torch.stack([local_xy[:, 0], local_xy[:, 1], z], dim=-1), quat], dim=-1)
        pose_inactive = torch.cat([base_pos + oob_offset, identity_quat], dim=-1)

        active_mask = (torch.full((len(env_ids),), i, device=env.device, dtype=torch.long) < num_active).unsqueeze(-1)
        pose = torch.where(active_mask, pose_active, pose_inactive)

        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zeros_6, env_ids=env_ids)


def sample_joint_encoder_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    bias_attr: str = "_encoder_joint_pos_bias",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample per-episode encoder bias for joint position observations.

    Domain randomization (Table II): randomly bias joint positions to model
    motor encoders' offset errors. Stored on env as (num_envs, num_joints).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Lazily create buffer on env
    bias = getattr(env, bias_attr, None)
    if bias is None or bias.shape != (env.num_envs, asset.num_joints) or bias.device != env.device:
        bias = torch.zeros((env.num_envs, asset.num_joints), device=env.device)
        setattr(env, bias_attr, bias)

    # Sample biases for selected envs/joints
    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        bias[env_ids] = math_utils.sample_uniform(bias_range[0], bias_range[1], (len(env_ids), asset.num_joints), env.device)
    else:
        # keep other joints unchanged
        sampled = math_utils.sample_uniform(
            bias_range[0], bias_range[1], (len(env_ids), len(joint_ids)), device=env.device
        )
        bias[env_ids[:, None], joint_ids] = sampled


def clear_joint_effort_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Clear any feed-forward joint effort targets (sets to zero)."""
    asset: Articulation = env.scene[asset_cfg.name]
    zeros = torch.zeros((len(env_ids), asset.num_joints if asset_cfg.joint_ids == slice(None) else len(asset_cfg.joint_ids)), device=env.device)
    asset.set_joint_effort_target(zeros, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def apply_erfi50_torque_perturbations(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    erfi_ratio: float = 0.5,
    num_levels: int = 10,
    max_scale: float = 1.0,
    erfi_torque_per_level: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """ERFI-50 [Campanaro et al.]: random torque perturbations for sim-to-real transfer.

    Domain randomization (Table II). Curriculum by terrain level.

    If erfi_torque_per_level is set (e.g. 0.78): Table II mode.
        tau_noise ~ U(-0.78 * level, +0.78 * level) N·m per joint, level = 0..num_levels-1.
    Else: tau_noise ~ U(-erfi_ratio * tau_lim, +erfi_ratio * tau_lim) * scale, scale from level.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    terrain_levels = None
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        terrain_levels = env.scene.terrain.terrain_levels

    if terrain_levels is None:
        levels = torch.zeros((len(env_ids),), dtype=torch.long, device=env.device)
    else:
        levels = terrain_levels[env_ids].to(dtype=torch.long)
        levels = torch.clamp(levels, 0, max(0, num_levels - 1))

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        nj = asset.num_joints
    else:
        nj = len(joint_ids)

    if erfi_torque_per_level is not None:
        # Table II: 0.78 N·m × difficulty level
        mag = erfi_torque_per_level * levels.to(torch.float32)
        mag = mag.unsqueeze(1).expand(-1, nj)
        u = 2.0 * torch.rand(len(env_ids), nj, device=env.device) - 1.0
        tau_noise = u * mag
    else:
        denom = float(max(1, num_levels - 1))
        scale = (levels.to(torch.float32) / denom).unsqueeze(1) * float(max_scale)
        if joint_ids == slice(None):
            limits = asset.data.joint_effort_limits[env_ids]
        else:
            limits = asset.data.joint_effort_limits[env_ids[:, None], joint_ids]
        u = 2.0 * torch.rand_like(limits) - 1.0
        tau_noise = u * (float(erfi_ratio) * limits) * scale

    asset.set_joint_effort_target(tau_noise, joint_ids=joint_ids, env_ids=env_ids)

