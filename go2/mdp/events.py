# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
Go2 目标到达任务：圆柱体障碍物事件（与 go1 agile 一致，便于避障训练）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim.schemas as sim_schemas
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject

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
    Obstacles are placed on a grid so that no two cylinders overlap in the same env.
    """
    if len(obstacle_names) == 0:
        return

    terrain_levels = None
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        terrain_levels = env.scene.terrain.terrain_levels

    if terrain_levels is None:
        levels = torch.zeros((len(env_ids),), dtype=torch.long, device=env.device)
    else:
        levels = terrain_levels[env_ids].to(dtype=torch.long)
        levels = torch.clamp(levels, 0, max(0, num_levels - 1))

    denom = max(1, num_levels - 1)
    max_for_level = torch.floor(levels.to(torch.float32) / float(denom) * float(max_obstacles)).to(torch.long)
    max_for_level = torch.clamp(max_for_level, 0, max_obstacles)

    num_active = torch.zeros((len(env_ids),), dtype=torch.long, device=env.device)
    has_any = max_for_level > 0
    if torch.any(has_any):
        # torch.randint expects scalar low/high; sample per-env upper bounds via scaled uniform.
        max_plus_one = (max_for_level[has_any] + 1).to(torch.float32)
        u = torch.rand(int(has_any.sum().item()), device=env.device)
        num_active[has_any] = torch.floor(u * max_plus_one).to(torch.long)
    if min_obstacles > 0:
        min_n = min(min_obstacles, max_obstacles)
        num_active = torch.maximum(num_active, torch.full_like(num_active, min_n))

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
    slot_centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    perm = torch.argsort(torch.rand(len(env_ids), num_slots, device=env.device), dim=1)

    zeros_6 = torch.zeros((len(env_ids), 6), device=env.device)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    oob_offset = torch.tensor(out_of_bounds_offset, device=env.device).view(1, 3)

    for i, name in enumerate(obstacle_names):
        asset: RigidObject = env.scene[name]
        root_state = asset.data.default_root_state[env_ids].clone()
        base_pos = root_state[:, 0:3] + env.scene.env_origins[env_ids]

        slot_idx = perm[:, i]
        local_xy = slot_centers[slot_idx]
        z = torch.zeros(len(env_ids), device=env.device)
        yaw = math_utils.sample_uniform(-math.pi, math.pi, (len(env_ids),), device=env.device)
        quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        pose_active = torch.cat([base_pos + torch.stack([local_xy[:, 0], local_xy[:, 1], z], dim=-1), quat], dim=-1)
        pose_inactive = torch.cat([base_pos + oob_offset, identity_quat], dim=-1)

        active_mask = (torch.full((len(env_ids),), i, device=env.device, dtype=torch.long) < num_active).unsqueeze(-1)
        pose = torch.where(active_mask, pose_active, pose_inactive)

        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zeros_6, env_ids=env_ids)


def reset_cylinder_obstacles_from_buffer(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    buffer_attr: str,
    default_z: float = 0.25,
    fixed_yaw: float | None = None,
    out_of_bounds_offset: tuple[float, float, float] = (1000.0, 1000.0, -10.0),
) -> None:
    """Reset cylinder obstacles from a per-env position buffer.

    The buffer is expected on the env object at attribute `buffer_attr` and can be:
    - shape (num_envs, 2, num_obj): [x;y] for each object
    - shape (num_envs, num_obj, 2): (x,y) for each object
    Values are interpreted in the env-local frame and will be shifted by env origins.
    Obstacles beyond provided num_obj are moved out-of-bounds.
    """
    if len(obstacle_names) == 0:
        return
    buf = getattr(env, buffer_attr, None)
    if buf is None:
        return
    if not torch.is_tensor(buf):
        buf = torch.tensor(buf, device=env.device, dtype=torch.float32)
    buf = buf.to(device=env.device, dtype=torch.float32)

    # normalize to (N, num_obj, 2)
    if buf.dim() != 3:
        return
    if buf.shape[1] == 2:
        # (N, 2, K)
        buf_xy = buf.permute(0, 2, 1).contiguous()
    elif buf.shape[2] == 2:
        # (N, K, 2)
        buf_xy = buf
    else:
        return

    n_env = int(env_ids.numel())
    num_obj = int(buf_xy.shape[1])
    zeros_6 = torch.zeros((n_env, 6), device=env.device)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(n_env, 1)
    oob_offset = torch.tensor(out_of_bounds_offset, device=env.device, dtype=torch.float32).view(1, 3)

    for i, name in enumerate(obstacle_names):
        asset: RigidObject = env.scene[name]
        root_state = asset.data.default_root_state[env_ids].clone()
        base_pos = root_state[:, 0:3] + env.scene.env_origins[env_ids]

        if i < num_obj:
            local_xy = buf_xy[env_ids, i]  # (n_env, 2)
            z = torch.full((n_env,), float(default_z), device=env.device)
            if fixed_yaw is None:
                yaw = math_utils.sample_uniform(-math.pi, math.pi, (n_env,), device=env.device)
            else:
                yaw = torch.full((n_env,), float(fixed_yaw), device=env.device)
            quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            pose = torch.cat([base_pos + torch.stack([local_xy[:, 0], local_xy[:, 1], z], dim=-1), quat], dim=-1)
        else:
            pose = torch.cat([base_pos + oob_offset, identity_quat], dim=-1)

        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zeros_6, env_ids=env_ids)
