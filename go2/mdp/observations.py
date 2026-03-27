from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .pose_utils import get_pose_command_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def lidar_distances_with_illusion(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "pose_command",
    margin: float = 0.3,
    max_distance: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_log: bool = True,
    log_eps: float = 0.01,
) -> torch.Tensor:
    """Ray-caster distances with illusion, optionally log-scaled."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    hits_w = sensor.data.ray_hits_w
    pos_w = sensor.data.pos_w
    d = torch.linalg.norm(hits_w - pos_w.unsqueeze(1), dim=-1)

    if max_distance is not None:
        d = torch.nan_to_num(d, nan=max_distance, posinf=max_distance, neginf=0.0)

    target_pos_2d, _ = get_pose_command_world(env, command_name)
    if target_pos_2d is not None:
        robot_xy = asset.data.root_pos_w[:, :2]
        d_goal = torch.linalg.norm(target_pos_2d - robot_xy, dim=-1)
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


def foot_contacts_binary(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
    foot_body_pattern: str = ".*_foot",
) -> torch.Tensor:
    """Binary foot contacts (E, num_feet) from contact sensor history.

    This is used to construct recovery-style proprioceptive observations without depending on go1 code.
    """
    import re

    sensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    body_names = list(asset.body_names)
    foot_ids = [i for i, n in enumerate(body_names) if re.match(foot_body_pattern, n)]
    if len(foot_ids) == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)
    # deterministic ordering by body name
    foot_ids = [i for _, i in sorted((body_names[i], i) for i in foot_ids)]
    forces_hist = sensor.data.net_forces_w_history[:, :, foot_ids]  # (E, H, F, 3)
    mags = torch.linalg.norm(forces_hist, dim=-1)  # (E, H, F)
    peak = torch.amax(mags, dim=1)  # (E, F)
    return (peak > float(threshold)).float()
