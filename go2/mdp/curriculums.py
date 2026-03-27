from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .pose_utils import get_pose_command_world

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg=None) -> torch.Tensor:
    """Faster terrain curriculum aligned with pose-goal navigation.

    This overrides IsaacLab's default `terrain_levels_vel` (velocity-command-based) with a goal-distance-based
    rule, so terrain difficulty increases when the robot reliably gets close to the random pose target.
    """
    # Resolve terrain object (generator-based terrain only).
    terrain = env.scene.terrain

    # Get goal position in world frame (2D).
    target_pos_2d, _ = get_pose_command_world(env, "pose_command")

    if target_pos_2d is None:
        # Fall back: don't change levels if we can't read pose target.
        return torch.mean(terrain.terrain_levels.float())

    # Current planar distance to target.
    robot = env.scene["robot"]
    dist = torch.linalg.norm(robot.data.root_pos_w[env_ids, :2] - target_pos_2d[env_ids], dim=1)
    current_levels = terrain.terrain_levels[env_ids]

    # Hysteresis curriculum (goal-reaching task):
    # - Easier "move_up" threshold to avoid getting stuck at low levels.
    # - Delay "move_down" until medium-high levels to prevent early oscillation.
    # - Use a wider down-threshold than up-threshold to reduce level chattering.
    move_up = dist < 1.8
    move_down = dist > 4.5
    move_down *= current_levels >= 6
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "distance_to_goal",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        # Lower bar + larger steps -> faster curriculum.
        if reward > reward_term.weight * 0.6:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "distance_to_goal",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.6:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
