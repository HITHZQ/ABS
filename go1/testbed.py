# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Migrated from Isaac Gym / legged_gym to Isaac Lab.

"""
Testbed for Go1 agile policy evaluation with obstacles and optional Recovery Assist (RA).
Run: isaaclab -p go1/testbed.py --num_envs 64 --policy path/to/policy.pt [--task go1_pos_rough]
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

import go1.mdp as mdp
from go1.agile import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    Go1PosRoughEnvCfg,
    Go1PosSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

import numpy as np
import torch
import torch.nn as nn

# Root dir for logs/policies (use ABS package dir)
ABS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECORD_FRAMES = False
MOVE_CAMERA = False
one_trial = False
train_ra = False
test_ra = False
visualize_ra = False

difficulty = 2  # 0: easy; 1: medium, 2: hard
init_obst_xy = [
    [-3.0, 8.0, -2.5, 2.5],
    [-3.0, 8.0, -2.5, 2.5],
    [1.5, 7.0, -2.0, 2.0],
]  # xmin, xmax, ymin, ymax

OBJECT_NAMES = [
    "cylinder_0",
    "cylinder_1",
    "cylinder_2",
    "cylinder_3",
    "cylinder_4",
    "cylinder_5",
    "cylinder_6",
    "cylinder_7",
]


def get_pos_integral(twist, tau):
    vx, vy, wz = twist[..., 0], twist[..., 1], twist[..., 2]
    theta = wz * tau
    x = vx * tau - 0.5 * vy * wz * tau * tau
    y = vy * tau + 0.5 * vx * wz * tau * tau
    return x, y, theta


twist_tau = 0.05
twist_eps = 0.05
twist_lam = 10.0
twist_lr = 0.5
twist_min = torch.tensor([-1.5, -0.3, -3.0])
twist_max = -twist_min


def _clip_grad(grad, thres):
    grad_norms = grad.norm(p=2, dim=-1).unsqueeze(-1)
    return grad * thres / torch.maximum(grad_norms, thres * torch.ones_like(grad_norms))


def sample_obstacle_test(xmin, xmax, ymin, ymax, n_env, n_obj, safedist=0.75):
    assert ymax - ymin > 2 * safedist
    assert xmax > xmin + 0.1
    obj_pos_sampled = torch.zeros(n_env, 2, n_obj, device="cuda")
    nodes = torch.zeros(n_env, 2, 4, device="cuda")
    nodes[:, 0, 0] = xmin
    nodes[:, 0, 1] = xmin * 0.67 + xmax * 0.33
    nodes[:, 0, 2] = xmin * 0.33 + xmax * 0.67
    nodes[:, 0, 3] = xmax
    nodes[:, 1, 0] = ymin * 0.5 + ymax * 0.5
    nodes[:, 1, 3] = ymin * 0.5 + ymax * 0.5
    nodes[:, 1, 1:3] = nodes[:, 1, 1:3].uniform_(ymin + safedist, ymax - safedist)
    A = torch.stack(
        [nodes[:, 0, :] ** 3, nodes[:, 0, :] ** 2, nodes[:, 0, :], torch.ones_like(nodes[:, 0, :])],
        dim=2,
    )
    coefficients = torch.linalg.lstsq(
        A, nodes[:, 1, :].unsqueeze(2)
    ).solution
    obj_pos_sampled[:, 0, :] = obj_pos_sampled[:, 0, :].uniform_(xmin, xmax)
    obj_pos_sampled[:, 1, :] = obj_pos_sampled[:, 1, :].uniform_(ymin, ymax)
    y_curve = (
        coefficients[:, 0] * obj_pos_sampled[:, 0, :] ** 3
        + coefficients[:, 1] * obj_pos_sampled[:, 0, :] ** 2
        + coefficients[:, 2] * obj_pos_sampled[:, 0, :]
        + coefficients[:, 3]
    )
    diffy = obj_pos_sampled[:, 1, :] - y_curve
    diffy[diffy == 0.0] = 0.001
    obj_pos_sampled[:, 1, :] = (
        obj_pos_sampled[:, 1, :] * (torch.abs(diffy) >= safedist)
        + (torch.sign(diffy) * safedist + y_curve) * (torch.abs(diffy) < safedist)
    )
    return obj_pos_sampled


# -----------------------------------------------------------------------------
# Testbed-specific env config
# -----------------------------------------------------------------------------

@configclass
class TestbedEventCfg(EventCfg):
    """Events for testbed: use obstacle positions from buffer instead of curriculum."""
    reset_cylinder_obstacles = EventTerm(
        func=mdp.reset_cylinder_obstacles_from_buffer,
        mode="reset",
        params={
            "obstacle_names": OBJECT_NAMES,
            "buffer_attr": "_test_obstacle_positions",
            "default_z": 0.25,
        },
    )


@configclass
class TestbedTerminationsCfg(TerminationsCfg):
    """Terminations for testbed: base + thighs + calves (original terminate_after_contacts_on), time_out."""
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("robot", body_names="trunk|.*thigh|.*calf"),
            "threshold": 1.0,
        },
    )


@configclass
class Go1TestbedEnvCfg(Go1PosRoughEnvCfg):
    """Testbed config: flat terrain, test obstacle layout, command ranges."""
    scene: Go1PosSceneCfg = replace(
        Go1PosSceneCfg(),
        num_envs=64,
        env_spacing=2.5,
    )

    def __post_init__(self):
        super().__post_init__()
        # Flat terrain for test
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            num_rows=3,
            num_cols=3,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.max_init_terrain_level = 0
        # Command ranges for test
        self.commands.pose_command.ranges = replace(
            self.commands.pose_command.ranges,
            pos_x=(6.0, 7.5),
            pos_y=(-1.5, 1.5),
            heading=(0.0, 0.0),
        )
        self.events = TestbedEventCfg()
        self.terminations = TestbedTerminationsCfg()


def _get_contact_forces(env):
    """Get body contact forces (num_envs, num_bodies, 3) from robot."""
    robot = env.scene["robot"]
    if hasattr(robot.data, "body_contact_forces_w"):
        return robot.data.body_contact_forces_w
    return torch.zeros(env.num_envs, len(robot.body_names), 3, device=env.device)


def _get_body_indices(robot, names):
    out = []
    for n in names:
        for i, bn in enumerate(robot.body_names):
            if n in bn or (isinstance(n, str) and n.replace("*", ".*") in bn):
                out.append(i)
                break
        else:
            out.append(-1)
    return out


def _get_obstacle_relpos(env, num_obj):
    """Robot-to-obstacle relative position (num_envs, num_obj, 3) in world frame."""
    robot = env.scene["robot"]
    rb_pos = robot.data.root_pos_w[:, :3]  # (N, 3)
    rel = []
    for i in range(min(num_obj, len(OBJECT_NAMES))):
        name = OBJECT_NAMES[i]
        if name in env.scene:
            obj = env.scene[name]
            obj_pos = obj.data.root_pos_w[:, :3]
            rel.append(obj_pos - rb_pos)
        else:
            rel.append(torch.zeros(env.num_envs, 3, device=env.device))
    if not rel:
        return torch.zeros(env.num_envs, num_obj, 3, device=env.device)
    return torch.stack(rel, dim=1)


def _get_goal_position(env):
    cmd = env.command_manager.get_command("pose_command")
    return cmd[:, :2]  # (N, 2)


def _get_obs_tensor(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    return torch.cat([v.flatten(1) for v in obs.values()], dim=1)


def play(env_cfg, policy_path, num_envs, train_ra_flag, test_ra_flag, device="cuda:0"):
    global difficulty, train_ra, test_ra
    train_ra = train_ra_flag
    test_ra = test_ra_flag
    if train_ra:
        difficulty = 1
    print("train RA:", train_ra, "; test RA:", test_ra)

    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device

    if difficulty == 0:
        object_num = 3
    else:
        object_num = 8

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    body_names = robot.body_names
    term_body_names = ["trunk", "FL_thigh", "FL_calf", "FR_thigh", "FR_calf"]
    feet_names = ["FL_foot", "FR_foot"]
    termination_contact_indices = []
    for n in term_body_names:
        for i, bn in enumerate(body_names):
            if n in bn or bn == n:
                termination_contact_indices.append(i)
                break
    feet_indices = []
    for n in feet_names:
        for i, bn in enumerate(body_names):
            if bn == n:
                feet_indices.append(i)
                break

    # Init obstacle buffer and sample
    test_obstacle_positions = torch.zeros(
        num_envs, 2, object_num, device=env.device, dtype=torch.float32
    )
    test_obstacle_positions[:, 0, :] = torch.rand(num_envs, object_num, device=env.device) * (
        init_obst_xy[difficulty][1] - init_obst_xy[difficulty][0]
    ) + init_obst_xy[difficulty][0]
    test_obstacle_positions[:, 1, :] = torch.rand(num_envs, object_num, device=env.device) * (
        init_obst_xy[difficulty][3] - init_obst_xy[difficulty][2]
    ) + init_obst_xy[difficulty][2]
    test_obstacle_positions[test_obstacle_positions == 0.0] = 0.01
    _too_near = (test_obstacle_positions.norm(dim=1) < 1.1).unsqueeze(1)
    test_obstacle_positions[:, 0:1, :] += (
        _too_near * torch.sign(test_obstacle_positions[:, 0:1, :]) * 0.9
    )
    test_obstacle_positions[:, 1:, :] += (
        _too_near * torch.sign(test_obstacle_positions[:, 1:, :]) * 0.9
    )
    env._test_obstacle_positions = test_obstacle_positions

    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        env.scene.terrain.terrain_levels[:] = 9

    # Load policy
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    print("Loaded policy from:", policy_path)

    # Observation buffer
    obs_dict, _ = env.reset()
    obs = _get_obs_tensor(obs_dict)

    camera_position = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array([0.0, 0.0, 0.0]) - camera_position

    ra_vf = nn.Sequential(
        nn.Linear(19, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Tanh(),
    )
    ra_vf.to(device)
    optimizer = torch.optim.SGD(ra_vf.parameters(), lr=0.002, momentum=0.0)

    policy_name = os.path.basename(policy_path)
    if train_ra:
        best_metric = 999.0
        path = os.path.join(ABS_ROOT, "logs", "exported", "RA", policy_name[:-3] + "_ra.pt")
        if os.path.isfile(path):
            _load = input("load existing value? y/n\n")
            if _load != "n":
                ra_vf = torch.load(path, map_location=device)
                print("loaded value from", path)
    if test_ra:
        path = os.path.join(ABS_ROOT, "logs", "exported", "RA", policy_name[:-3] + "_ra.pt")
        ra_vf = torch.load(path, map_location=device)
        print("loaded value from", path)
        rec_policy_path = os.path.join(ABS_ROOT, "resources", "policy", "recover_v4_twist.pt")
        rec_policy = torch.jit.load(rec_policy_path, map_location=device)
        print("loaded recovery policy from", rec_policy_path)
        mode_running = True

    standard_raobs_init = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 6.0, 0] + [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    standard_raobs_die = torch.tensor(
        [[5.0, 0, 0, 0, 0, 0, 6.0, 0] + [-2.5] * 11],
        device=device,
    )
    standard_raobs_turn = torch.tensor(
        [[0, 0, 0, 0, 0, 2.0, 0.5, 5.8] + [2.0] * 6 + [0.0] * 5],
        device=device,
    )
    ra_obs = standard_raobs_init.clone().repeat(env.num_envs, 1)
    collision = torch.zeros(env.num_envs, device=env.device).bool()

    queue_len = 1001
    batch_size = 200
    hindsight = 10
    s_queue = torch.zeros((queue_len, env.num_envs, 19), device=env.device, dtype=torch.float)
    g_queue = torch.zeros((queue_len, env.num_envs), device=env.device, dtype=torch.float)
    g_hs_queue = g_queue.clone()
    g_hs_span = torch.zeros((2, env.num_envs), device=env.device, dtype=torch.int)
    l_queue = torch.zeros((queue_len, env.num_envs), device=env.device, dtype=torch.float)
    done_queue = torch.zeros((queue_len, env.num_envs), device=env.device, dtype=torch.bool)
    alive = torch.ones(env.num_envs, device=env.device).bool()

    # Metrics
    (
        total_n_collision,
        total_n_reach,
        total_n_timeout,
    ) = 0, 0, 0
    total_n_episodic_recovery = 0
    total_n_episodic_recovery_success = 0
    total_n_episodic_recovery_fail = 0
    total_recovery_dist = 0
    total_recovery_timesteps = 0
    total_n_collision_when_ra_on = 0
    total_n_collision_when_ra_off = 0
    total_n_done = 0
    total_episode = 0
    last_obs = obs.clone()
    last_root_states = robot.data.root_pos_w[:, :3].clone()
    last_position_targets = _get_goal_position(env).clone()
    episode_travel_dist = torch.zeros(env.num_envs, device=env.device)
    episode_time = torch.zeros(env.num_envs, device=env.device)
    episode_max_velo = torch.zeros(env.num_envs, device=env.device)
    episode_max_velo_dist = 0
    episode_max_velo_dist_collision = 0
    episode_max_velo_dist_reach = 0
    episode_max_velo_dist_timeout = 0
    total_travel_dist = 0
    total_time = 0
    total_reach_dist = 0
    total_collision_dist = 0
    total_timeout_dist = 0
    total_reach_time = 0
    total_collision_time = 0
    total_timeout_time = 0
    episode_recovery_logging = torch.zeros(env.num_envs, device=env.device).bool()
    current_recovery_status = torch.zeros(env.num_envs, device=env.device).bool()

    max_episode_length = int(env.episode_length_buf[0].item()) if hasattr(env, "episode_length_buf") else 1600
    max_episode_length_s = env.step_dt * max_episode_length
    prev_where_done = torch.tensor([], device=env.device, dtype=torch.long)

    for i in range(300 * max_episode_length):
        current_recovery_status.zero_()
        where_recovery = torch.zeros(env.num_envs, device=env.device).bool()

        if i % 1000 == 0 and train_ra:
            env._test_obstacle_positions[:, 0, :] = torch.rand(
                num_envs, object_num, device=env.device
            ) * (init_obst_xy[difficulty][1] - init_obst_xy[difficulty][0]) + init_obst_xy[difficulty][0]
            env._test_obstacle_positions[:, 1, :] = torch.rand(
                num_envs, object_num, device=env.device
            ) * (init_obst_xy[difficulty][3] - init_obst_xy[difficulty][2]) + init_obst_xy[difficulty][2]
            env._test_obstacle_positions[env._test_obstacle_positions == 0.0] = 0.01
            _too_near = (env._test_obstacle_positions.norm(dim=1) < 1.1).unsqueeze(1)
            env._test_obstacle_positions[:, 0:1, :] += (
                _too_near * torch.sign(env._test_obstacle_positions[:, 0:1, :]) * 0.9
            )
            env._test_obstacle_positions[:, 1:, :] += (
                _too_near * torch.sign(env._test_obstacle_positions[:, 1:, :]) * 0.9
            )

        reset_ids = prev_where_done if i > 0 else torch.tensor([], device=env.device, dtype=torch.long)
        if reset_ids.numel() > 0 and not train_ra:
            n_reset = reset_ids.numel()
            sampled = sample_obstacle_test(
                init_obst_xy[difficulty][0],
                init_obst_xy[difficulty][1],
                init_obst_xy[difficulty][2],
                init_obst_xy[difficulty][3],
                n_reset,
                object_num,
            )
            env._test_obstacle_positions[reset_ids] = sampled
            env._test_obstacle_positions[env._test_obstacle_positions == 0.0] = 0.01
            _too_near = (env._test_obstacle_positions[reset_ids].norm(dim=1) < 1.1).unsqueeze(1)
            env._test_obstacle_positions[reset_ids, 0:1, :] += (
                _too_near * torch.sign(env._test_obstacle_positions[reset_ids, 0:1, :]) * 0.9
            )
            env._test_obstacle_positions[reset_ids, 1:, :] += (
                _too_near * torch.sign(env._test_obstacle_positions[reset_ids, 1:, :]) * 0.9
            )

        if one_trial and i == 1:
            pass  # Skip auto-reset handling for one_trial
        if one_trial:
            print("step", i, "survive rate", alive.float().mean().item())

        with torch.inference_mode():
            actions = policy(obs.detach().unsqueeze(0)).squeeze(0) * alive.unsqueeze(1).float()
        actions = actions.to(env.device)

        if test_ra:
            v_pred = ra_vf(ra_obs)
            start_v = ra_vf(standard_raobs_init).mean().item()
            die_v = ra_vf(standard_raobs_die).mean().item()
            turn_v = ra_vf(standard_raobs_turn).mean().item()
            recovery = (v_pred > -twist_eps).squeeze(-1)
            where_recovery = torch.where(torch.logical_and(recovery, ~collision))[0]

            if collision.sum().item() > 0 and mode_running and visualize_ra:
                cf = _get_contact_forces(env)
                print(torch.norm(cf[0, :, :], dim=-1))
                print("nooooooo i died!")
                time.sleep(5.0)

            if where_recovery.shape[0] > 0:
                episode_recovery_logging[where_recovery] = True
                current_recovery_status[where_recovery] = True
                mode_running = False
                base_lin = robot.data.root_lin_vel_b
                base_ang = robot.data.root_ang_vel_b
                twist_iter = torch.cat(
                    [base_lin[where_recovery, 0:2], base_ang[where_recovery, 2:3]],
                    dim=-1,
                ).clone()
                twist_iter.requires_grad = True
                for _iter in range(10):
                    twist_ra_obs = torch.cat(
                        [
                            twist_iter[..., 0:2],
                            base_lin[where_recovery, 2:3],
                            base_ang[where_recovery, 0:2],
                            twist_iter[..., 2:3],
                            obs[where_recovery, 10:12],
                            obs[where_recovery, -11:],
                        ],
                        dim=-1,
                    )
                    x_iter, y_iter, _ = get_pos_integral(twist_iter, twist_tau)
                    ra_value = ra_vf(twist_ra_obs)
                    loss_separate = (
                        twist_lam * (ra_value + 2 * twist_eps).clamp(min=0).squeeze(-1)
                        + 0.02
                        * (
                            (x_iter - obs[where_recovery, 10:11].squeeze(-1)) ** 2
                            + (y_iter - obs[where_recovery, 11:12].squeeze(-1)) ** 2
                        )
                    )
                    loss = loss_separate.sum()
                    loss.backward()
                    twist_iter.data = twist_iter.data - twist_lr * _clip_grad(
                        twist_iter.grad.data, 1.0
                    )
                    twist_iter.data = twist_iter.data.clamp(
                        min=twist_min.to(env.device), max=twist_max.to(env.device)
                    )
                    twist_iter.grad.zero_()
                twist_iter = twist_iter.detach()
                # Recovery policy expects (c_f, ω, g, tw_c, q, q̇, a) per recovery.py
                foot_contacts_ = mdp.foot_contacts(
                    env,
                    threshold=1.0,
                    asset_cfg=SceneEntityCfg("robot"),
                    foot_body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                )
                obs_rec = torch.cat(
                    [
                        foot_contacts_[where_recovery],  # 4
                        obs[where_recovery, 3:6],        # base_ang_vel 3
                        obs[where_recovery, 6:9],        # projected_gravity 3
                        twist_iter,                       # tw_c 3
                        obs[where_recovery, 12:24],      # joint_pos 12
                        obs[where_recovery, 24:36],      # joint_vel 12
                        obs[where_recovery, 36:48],      # last_action 12
                    ],
                    dim=-1,
                )
                with torch.inference_mode():
                    actions[where_recovery] = rec_policy(obs_rec.detach().unsqueeze(0)).squeeze(0)
            else:
                mode_running = True

        obs_dict, rew, terminated, truncated, info = env.step(actions.detach())
        obs = _get_obs_tensor(obs_dict)
        dones = terminated | truncated

        cf = _get_contact_forces(env)
        collision = torch.zeros(env.num_envs, device=env.device).bool()
        if termination_contact_indices:
            valid = [idx for idx in termination_contact_indices if 0 <= idx < cf.shape[1]]
            if valid:
                collision = torch.any(
                    torch.norm(cf[:, valid, :], dim=-1) > 1.0, dim=1
                )
        if feet_indices:
            hor_footforce = cf[:, feet_indices[:2], 0:2].norm(dim=-1)
            ver_footforce = torch.abs(cf[:, feet_indices[:2], 2])
            foot_hor_col = torch.any(hor_footforce > 2 * ver_footforce + 10.0, dim=-1)
            collision = torch.logical_or(collision, foot_hor_col)
        minobjdist = _get_obstacle_relpos(env, object_num).norm(dim=-1)
        _near_obj = torch.any(minobjdist < 0.95, dim=-1)
        base_vel = robot.data.root_lin_vel_w[:, :2]
        _near_obj = torch.logical_and(_near_obj, base_vel.norm(dim=-1) > 0.5)
        base_contact = cf[:, 0, :].norm(dim=-1) > 1.0 if cf.shape[1] > 0 else torch.zeros(env.num_envs, device=env.device).bool()
        _near_obj = torch.logical_or(_near_obj, base_contact)
        collision = torch.logical_and(collision, _near_obj)

        where_done = torch.where(dones)[0]
        prev_where_done = where_done
        where_collision = torch.where(torch.logical_and(dones, collision))[0]
        distance_to_goal = torch.norm(
            last_position_targets[:, 0:2] - last_root_states[:, 0:2], dim=-1
        )
        where_reach = torch.where(
            torch.logical_and(
                distance_to_goal < 0.65,
                torch.logical_and(dones, ~collision),
            )
        )[0]
        where_timeout = torch.where(
            torch.logical_and(
                distance_to_goal >= 0.65,
                torch.logical_and(dones, ~collision),
            )
        )[0]
        not_in_goal = torch.logical_and(distance_to_goal >= 0.65, ~dones)

        total_episode += where_done.shape[0]
        total_n_done += where_done.shape[0]
        total_n_collision += where_collision.shape[0]
        total_n_reach += where_reach.shape[0]
        total_n_timeout += where_timeout.shape[0]

        if i > 0:
            root_pos = robot.data.root_pos_w[:, 0:2]
            onestep_dist = (
                torch.norm(root_pos - last_root_states[:, 0:2], dim=-1) * not_in_goal.float()
            )
            episode_travel_dist += onestep_dist
            episode_time += env.step_dt * not_in_goal.float()
            episode_max_velo = torch.maximum(episode_max_velo, onestep_dist / env.step_dt)
            total_recovery_dist += onestep_dist[where_recovery].sum().item()
            total_recovery_timesteps += where_recovery.shape[0]

        if where_done.shape[0] > 0:
            total_n_episodic_recovery += episode_recovery_logging[where_done].sum().item()
            total_n_episodic_recovery_fail += (
                episode_recovery_logging[where_done] * collision[where_done]
            ).sum().item()
            total_n_episodic_recovery_success += (
                episode_recovery_logging[where_done] * ~collision[where_done]
            ).sum().item()
            episode_recovery_logging[where_done] = False
            total_n_collision_when_ra_on += (
                torch.logical_and(collision[where_done], current_recovery_status[where_done])
            ).sum().item()
            total_n_collision_when_ra_off += (
                torch.logical_and(collision[where_done], ~current_recovery_status[where_done])
            ).sum().item()

            episode_max_velo_dist += episode_max_velo[where_done].sum().item()
            episode_max_velo_dist_collision += episode_max_velo[where_collision].sum().item()
            episode_max_velo_dist_reach += episode_max_velo[where_reach].sum().item()
            episode_max_velo_dist_timeout += episode_max_velo[where_timeout].sum().item()
            episode_max_velo[where_done] = 0

            collision_dist = episode_travel_dist[where_collision].sum().item()
            collision_time = episode_time[where_collision].sum().item()
            total_collision_dist += collision_dist
            total_collision_time += collision_time
            total_travel_dist += collision_dist
            total_time += collision_time

            reach_dist = episode_travel_dist[where_reach].sum().item()
            reach_time = episode_time[where_reach].sum().item()
            total_reach_dist += reach_dist
            total_reach_time += reach_time
            total_travel_dist += reach_dist
            total_time += reach_time

            timeout_dist = episode_travel_dist[where_timeout].sum().item()
            timeout_time = episode_time[where_timeout].sum().item()
            total_timeout_dist += timeout_dist
            total_timeout_time += timeout_time
            total_travel_dist += timeout_dist
            total_time += timeout_time

            episode_time[where_done] = 0
            episode_travel_dist[where_done] = 0

            avg_collision_dist = total_collision_dist / (total_n_collision + 1e-8)
            avg_collision_time = total_collision_time / (total_n_collision + 1e-8)
            avg_reach_dist = total_reach_dist / (total_n_reach + 1e-8)
            avg_reach_time = total_reach_time / (total_n_reach + 1e-8)
            avg_timeout_dist = total_timeout_dist / (total_n_timeout + 1e-8)
            avg_timeout_time = total_timeout_time / (total_n_timeout + 1e-8)
            avg_total_dist = total_travel_dist / (total_episode + 1e-8)
            avg_total_time = total_time / (total_episode + 1e-8)
            avg_total_velocity = avg_total_dist / avg_total_time
            avg_collision_velocity = avg_collision_dist / (avg_collision_time + 1e-8)
            avg_reach_velocity = avg_reach_dist / (avg_reach_time + 1e-8)
            avg_timeout_velocity = avg_timeout_dist / (avg_timeout_time + 1e-8)
            avg_recovery_velocity = total_recovery_dist / (
                (total_recovery_timesteps + 1e-8) * env.step_dt
            )

        if (
            total_episode % 1 == 0
            and total_episode > 0
            and (not train_ra)
            and where_done.shape[0] > 0
        ):
            print("========= Episode {} =========".format(total_episode))
            print("Total Episode:                         {}".format(total_episode))
            print("Total Collision:                       {}".format(total_n_collision))
            print("Total Reach:                           {}".format(total_n_reach))
            print("Total Timeout:                         {}".format(total_n_timeout))
            print("Collision Rate:                        {:.2%}".format(total_n_collision / total_episode))
            print("Reach Rate:                            {:.2%}".format(total_n_reach / total_episode))
            print("Timeout Rate:                          {:.2%}".format(total_n_timeout / total_episode))
            print("Average Total Velocity:                {:.2f}".format(avg_total_velocity))
            print("Average Recovery Velocity:             {:.2f}".format(avg_recovery_velocity))
            print("Episode that activated recovery:         {}".format(total_n_episodic_recovery))
            print("RA activation rate for collision moments: {:.2%}".format(total_n_collision_when_ra_on / (total_n_collision + 1e-8)))

        last_obs = obs.clone()
        last_root_states = robot.data.root_pos_w[:, :3].clone()
        last_position_targets = _get_goal_position(env).clone()

        if one_trial:
            alive = torch.logical_and(alive, ~collision)

        base_lin = robot.data.root_lin_vel_b
        base_ang = robot.data.root_ang_vel_b
        lidar_len = 11
        if obs.shape[1] >= 10 + 2 + lidar_len:
            ra_obs = torch.cat(
                [base_lin, base_ang, obs[:, 10:12], obs[:, -lidar_len:]], dim=-1
            )
        else:
            ra_obs = torch.cat(
                [base_lin, base_ang, obs[:, 10:12], torch.zeros(env.num_envs, lidar_len, device=env.device)],
                dim=-1,
            )

        if train_ra:
            gs = collision.float() * 2 - 1
            ls = torch.tanh(
                torch.log2(torch.norm(obs[:, 10:12], dim=-1) / 0.65 + 1e-8)
            )
            s_queue[:-1] = s_queue[1:].clone()
            g_queue[:-1] = g_queue[1:].clone()
            l_queue[:-1] = l_queue[1:].clone()
            done_queue[:-1] = done_queue[1:].clone()
            s_queue[-1] = ra_obs.clone()
            g_queue[-1] = gs.clone()
            l_queue[-1] = ls.clone()
            done_queue[-1] = dones.clone()
            g_hs_queue[:-1] = g_hs_queue[1:].clone()
            g_hs_queue[-1] = gs.clone()
            g_hs_span[:] -= 1
            g_hs_span[0][dones] = g_hs_span[1][dones].clone() + 1
            g_hs_span[1][dones] = queue_len - 1
            g_hs_span[0] = torch.maximum(g_hs_span[0], g_hs_span[1] - hindsight)
            g_hs_span = g_hs_span * (g_hs_span >= 0)
            range_tensor = torch.arange(queue_len, device=env.device).unsqueeze(1)
            mask = (range_tensor >= g_hs_span[0:1]) & (range_tensor < g_hs_span[1:2])
            new_values = gs.clone().repeat(queue_len, 1)
            mask = mask & (new_values > 0)
            new_values -= (g_hs_span[1:2] - range_tensor) * 2 / hindsight * mask
            g_hs_queue[mask] = new_values[mask].clone()

            if i > queue_len and i % 20 == 0:
                false_safe, false_reach, n_fail, n_reach = 0, 0, 0, 0
                accu_loss = []
                total_n_fail = torch.logical_and(g_queue[1:] > 0, done_queue[1:]).sum().item()
                total_n_reach = torch.logical_and(l_queue[:-1] <= 0, done_queue[1:]).sum().item()
                start_v = ra_vf(standard_raobs_init).mean().item()
                die_v = ra_vf(standard_raobs_die).mean().item()
                turn_v = ra_vf(standard_raobs_turn).mean().item()
                gamma = 0.999999
                for _start in range(0, queue_len - 1, batch_size):
                    vs_old = ra_vf(s_queue[_start : _start + batch_size]).squeeze(-1)
                    with torch.no_grad():
                        vs_new = (
                            ra_vf(s_queue[_start + 1 : _start + batch_size + 1]).squeeze(-1)
                            * (~done_queue[_start + 1 : _start + batch_size + 1])
                            + 1.0 * done_queue[_start + 1 : _start + batch_size + 1]
                        )
                        vs_discounted_old = (
                            gamma
                            * torch.maximum(
                                g_hs_queue[_start + 1 : _start + batch_size + 1],
                                torch.minimum(
                                    l_queue[_start : _start + batch_size], vs_new
                                ),
                            )
                            + (1 - gamma)
                            * torch.maximum(
                                l_queue[_start : _start + batch_size],
                                g_hs_queue[_start + 1 : _start + batch_size + 1],
                            )
                        )
                    v_loss = 100 * torch.mean(
                        torch.square(vs_old - vs_discounted_old)
                    )
                    optimizer.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ra_vf.parameters(), 1.0)
                    optimizer.step()
                    false_safe += torch.logical_and(
                        g_queue[_start + 1 : _start + batch_size + 1] > 0, vs_old <= 0
                    ).sum().item()
                    false_reach += torch.logical_and(
                        l_queue[_start : _start + batch_size] <= 0, vs_old > 0
                    ).sum().item()
                    n_fail += (g_queue[_start + 1 : _start + batch_size + 1] > 0).sum().item()
                    n_reach += (l_queue[_start : _start + batch_size] <= 0).sum().item()
                    accu_loss.append(v_loss.item())
                new_loss = np.mean(accu_loss)
                print(
                    "value RA loss %.4f, false safe rate %.2f in %d, false reach rate %.2f in %d, step %d"
                    % (new_loss, false_safe / (n_fail + 1e-8), n_fail, false_reach / (n_reach + 1e-8), n_reach, i)
                )
                if (
                    false_safe / (n_fail + 1e-8) < best_metric
                    and die_v > 0.2
                    and start_v < -0.1
                    and turn_v < -0.1
                    and i > 3000
                ):
                    best_metric = false_safe / (n_fail + 1e-8)
                    os.makedirs(os.path.join(ABS_ROOT, "logs", "exported", "RA"), exist_ok=True)
                    torch.save(ra_vf, path)
                    print("\x1b[6;30;42m saving ra model to", path, "\x1b[0m")

        if RECORD_FRAMES and i % 2 == 0:
            pass  # Isaac Lab: use env.render() or similar if available

        if MOVE_CAMERA:
            camera_position += camera_vel * env.step_dt
            # env.set_camera(...) - Isaac Lab viewer API may differ

    env.close()


def main():
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Go1 testbed - agile policy eval with obstacles")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--policy", type=str, required=True, help="Path to JIT policy (.pt)")
    parser.add_argument("--task", type=str, default="go1_pos_rough")
    parser.add_argument("--trainRA", action="store_true")
    parser.add_argument("--testRA", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env_cfg = Go1TestbedEnvCfg()
    play(
        env_cfg,
        policy_path=args.policy,
        num_envs=args.num_envs,
        train_ra_flag=args.trainRA,
        test_ra_flag=args.testRA,
        device=args.device,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
