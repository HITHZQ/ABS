# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Migrated from Isaac Gym / legged_gym to Isaac Lab.
"""
Ray-prediction network data collection: depth images + lidar labels.
Run: isaaclab -p go1/camrec.py --policy path/to/policy.pt --enable_cameras [--log_root DIR] [--shift N]

Tips:
  1. --shift and --log_root: use different values for parallel runs, merge by moving files.
     Obstacles: edit cylinder config in go1/agile.py or Go1CamRecSceneCfg below.
  2. After collection, run: python go1/train_depth_resnet.py --data_folder <log_root>
  3. Camera: resolution, FOV, depth range in CAMREC_CAMERA below.
"""

from __future__ import annotations

import math
import os
import pickle
import shutil
from dataclasses import replace

import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

import numpy as np
import torch

from go1.agile import Go1AgileSceneCfg
from go1.testbed import Go1TestbedEnvCfg

ABS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Camera config (Tip 3): resolution, position, FOV, depth range
# -----------------------------------------------------------------------------
CAMREC_CAMERA = {
    "width": 80,
    "height": 60,
    "fov": 90.0,  # degrees, ~pi/2 to align with lidar
    "clipping_range": (0.1, 6.0),  # depth range in meters
    "pos": (-0.05, 0.0, 0.0),  # on robot base, same as lidar
}


@configclass
class Go1CamRecSceneCfg(Go1AgileSceneCfg):
    """Scene with depth camera for ray-prediction data collection."""

    def __post_init__(self):
        super().__post_init__()
        # Flat terrain for camrec
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            num_rows=4,
            num_cols=4,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.terrain.max_init_terrain_level = 0

        # Depth camera: same mount as lidar, similar FOV
        focal_length = CAMREC_CAMERA["width"] / (2 * math.tan(math.radians(CAMREC_CAMERA["fov"] / 2)))
        self.depth_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/depth_cam",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=20.955,
                vertical_aperture=20.955 * CAMREC_CAMERA["height"] / CAMREC_CAMERA["width"],
                clipping_range=CAMREC_CAMERA["clipping_range"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=CAMREC_CAMERA["pos"],
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
            data_types=["depth"],
            width=CAMREC_CAMERA["width"],
            height=CAMREC_CAMERA["height"],
            depth_clipping_behavior="max",
        )


@configclass
class Go1CamRecEnvCfg(Go1TestbedEnvCfg):
    """CamRec env: fewer envs, depth camera enabled, for data collection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene = replace(Go1CamRecSceneCfg(), num_envs=9, env_spacing=2.5)
        self.episode_length_s = 5.0


def _get_obs_tensor(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    return torch.cat([v.flatten(1) for v in obs.values()], dim=1)


def play(
    env_cfg,
    policy_path: str,
    log_root: str,
    shift: int = 10,
    obstacle_assets: dict | None = None,
    device: str = "cuda:0",
):
    """
    Collect (depth_image, ray2d_label) pairs for ray-prediction training.

    Args:
        env_cfg: Go1CamRecEnvCfg
        policy_path: Path to agile policy .pt
        log_root: Root dir for logs (Tip 1: change for parallel runs)
        shift: Filename shift to avoid conflicts (Tip 1: change for parallel runs)
        obstacle_assets: Not used in Isaac Lab (obstacles from scene cfg). Kept for API compat.
        device: torch device
    """
    env_cfg.sim.device = device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    print("Loaded policy from:", policy_path)

    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
        env.scene.terrain.terrain_levels[:] = 9

    obs_dict, _ = env.reset()
    obs = _get_obs_tensor(obs_dict)

    os.makedirs(log_root, exist_ok=True)
    existing = len([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])
    log_folder_name = "test" + str(existing + shift)
    last_folder_name = "test" + str(existing + shift - 1)
    last_success = os.path.isfile(os.path.join(log_root, last_folder_name, "label.pkl"))
    print("last recording succeed?", last_success)
    if not last_success and os.path.isdir(os.path.join(log_root, last_folder_name)):
        try:
            shutil.rmtree(os.path.join(log_root, last_folder_name))
            log_folder_name = last_folder_name
        except Exception:
            pass

    log_folder = os.path.join(log_root, log_folder_name)
    os.makedirs(log_folder, exist_ok=True)
    print("created folder", log_folder)

    depth_cam = env.scene["depth_cam"] if "depth_cam" in env.scene else None
    lidar = env.scene.sensors["lidar"] if "lidar" in env.scene.sensors else None

    if depth_cam is None:
        raise RuntimeError("depth_cam not in scene. Use Go1CamRecEnvCfg and --enable_cameras.")

    labels = {}
    n_steps = 50 * 5 * 4
    sample_every = 5
    sample_at = 2

    for i in range(n_steps):
        if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "terrain_levels"):
            env.scene.terrain.terrain_levels[:] = torch.randint_like(
                env.scene.terrain.terrain_levels, low=0, high=10
            )

        with torch.inference_mode():
            actions = policy(obs.detach().unsqueeze(0)).squeeze(0)
        obs_dict, _, terminated, truncated, _ = env.step(actions.detach())
        obs = _get_obs_tensor(obs_dict)

        if i % sample_every == sample_at:
            depth_data = depth_cam.data.output["depth"]
            if depth_data is None:
                continue
            depth_np = depth_data.detach().cpu().numpy()
            if len(depth_np.shape) == 4:
                depth_np = depth_np[..., 0]

            lidar_data = None
            if lidar is not None:
                hits_w = lidar.data.ray_hits_w
                pos_w = lidar.data.pos_w
                d = torch.linalg.norm(hits_w - pos_w.unsqueeze(1), dim=-1)
                d = torch.nan_to_num(d, nan=6.0, posinf=6.0, neginf=0.0)
                lidar_data = d.detach().cpu().numpy()

            for robot_id in range(env.num_envs):
                save_name = "robot_%d_step%d" % (robot_id, i)
                if depth_np.ndim == 4:
                    cam_data = depth_np[robot_id].squeeze()
                elif depth_np.ndim == 3:
                    cam_data = depth_np[robot_id].squeeze() if depth_np.shape[0] == env.num_envs else depth_np.squeeze()
                else:
                    cam_data = depth_np.squeeze()
                if lidar_data is not None:
                    if lidar_data.ndim == 2:
                        ray2d_label = lidar_data[robot_id]
                    else:
                        ray2d_label = lidar_data
                    labels[save_name] = np.array(ray2d_label, dtype=np.float32).flatten()
                np.save(os.path.join(log_folder, save_name + ".npy"), cam_data.astype(np.float32))
            print("timestep %d save done" % i)

    with open(os.path.join(log_folder, "label.pkl"), "wb") as f:
        pickle.dump(labels, f)
    with open(os.path.join(log_folder, "label.pkl"), "rb") as f:
        _labels = pickle.load(f)
        k = list(_labels.keys())[-1] if _labels else ""
        print(_labels.get(k, "no labels"))
    print("saved labels")
    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CamRec: collect depth + lidar for ray-prediction")
    parser.add_argument("--policy", type=str, required=True, help="Path to agile policy .pt")
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Root dir for logs (Tip 1). Default: ABS_ROOT/logs/rec_cam",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=10,
        help="Filename shift for parallel runs (Tip 1). Default: 10",
    )
    parser.add_argument("--num_envs", type=int, default=9)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    log_root = args.log_root or os.path.join(ABS_ROOT, "logs", "rec_cam")

    env_cfg = Go1CamRecEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    play(
        env_cfg=env_cfg,
        policy_path=args.policy,
        log_root=log_root,
        shift=args.shift,
        device=args.device,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
