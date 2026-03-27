"""
Go2 ray-prediction data collection: depth images + lidar-observation labels.

Run:
  isaaclab -p go2/camrec.py --policy path/to/policy.pt --enable_cameras [--log_root DIR]

Output structure (compatible with go2/train_depth_resnet.py):
  <log_root>/testXX/robot_<env_id>_step<step>.npy
  <log_root>/testXX/label_raw.pkl   # dict[name -> log(raw ray distances)]
  <log_root>/testXX/label_obs.pkl   # dict[name -> log(illusioned ray distances)]
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
from dataclasses import replace

import numpy as np
import torch
from isaaclab.app import AppLauncher

# Launch the app before importing simulation-dependent Isaac Lab modules.
parser = argparse.ArgumentParser(description="Go2 CamRec: collect depth + lidar pairs for depth->ray training.")
parser.add_argument("--policy", type=str, required=True, help="Path to torchscript policy .pt")
parser.add_argument("--log_root", type=str, default=None, help="Root dir for logs. Default: ABS_ROOT/logs/rec_cam_go2")
parser.add_argument("--shift", type=int, default=10, help="Folder index shift for parallel runs")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--sample_every", type=int, default=5)
parser.add_argument("--sample_offset", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from go2 import mdp as go2_mdp
from go2.velocity_env_cfg import RobotEnvCfg, RobotSceneCfg

ABS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMREC_CAMERA = {
    "width": 80,
    "height": 60,
    "fov": 90.0,
    "clipping_range": (0.1, 6.0),
    "pos": (-0.05, 0.0, 0.0),
}

DEPTH_MIN = CAMREC_CAMERA["clipping_range"][0]
DEPTH_MAX = CAMREC_CAMERA["clipping_range"][1]


@configclass
class Go2CamRecSceneCfg(RobotSceneCfg):
    """Go2 scene with depth camera for depth->ray data collection."""

    def __post_init__(self):
        super().__post_init__()

        focal_length = CAMREC_CAMERA["width"] / (2.0 * math.tan(math.radians(CAMREC_CAMERA["fov"] / 2.0)))
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
class Go2CamRecEnvCfg(RobotEnvCfg):
    """CamRec env for Go2: depth camera enabled, fewer envs, short episodes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene = replace(Go2CamRecSceneCfg(), num_envs=16, env_spacing=2.5)
        self.episode_length_s = 6.0


def _get_obs_tensor(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    return torch.cat([v.flatten(1) for v in obs.values()], dim=1)


def _make_log_folder(log_root: str, shift: int) -> str:
    os.makedirs(log_root, exist_ok=True)
    existing = len([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])
    log_folder_name = "test" + str(existing + shift)
    last_folder_name = "test" + str(existing + shift - 1)
    last_raw_ok = os.path.isfile(os.path.join(log_root, last_folder_name, "label_raw.pkl"))
    last_obs_ok = os.path.isfile(os.path.join(log_root, last_folder_name, "label_obs.pkl"))
    last_success = last_raw_ok and last_obs_ok
    if not last_success and os.path.isdir(os.path.join(log_root, last_folder_name)):
        try:
            shutil.rmtree(os.path.join(log_root, last_folder_name))
            log_folder_name = last_folder_name
        except Exception:
            pass
    log_folder = os.path.join(log_root, log_folder_name)
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def collect(
    env_cfg: Go2CamRecEnvCfg,
    policy_path: str,
    log_root: str,
    shift: int = 10,
    n_steps: int = 1000,
    sample_every: int = 5,
    sample_offset: int = 2,
    device: str = "cuda:0",
):
    """Collect (depth_image, lidar_observation_label) pairs from Go2 rollouts."""
    env_cfg.sim.device = device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()

    obs_dict, _ = env.reset()
    obs = _get_obs_tensor(obs_dict)

    log_folder = _make_log_folder(log_root, shift)
    print("Collecting into:", log_folder)

    try:
        depth_cam = env.scene["depth_cam"]
    except KeyError:
        depth_cam = None
    try:
        lidar = env.scene.sensors["lidar"]
    except KeyError:
        lidar = None

    if depth_cam is None:
        raise RuntimeError("depth_cam not found. Launch with --enable_cameras and Go2CamRecEnvCfg.")
    if lidar is None:
        raise RuntimeError("lidar not found in scene. Check RobotSceneCfg lidar sensor.")

    labels_raw = {}
    labels_obs = {}
    for i in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs_dict, _, _, _, _ = env.step(actions.detach())
        obs = _get_obs_tensor(obs_dict)

        if i % sample_every != sample_offset:
            continue

        depth_data = depth_cam.data.output["depth"]
        if depth_data is None:
            continue
        depth_np = depth_data.detach().cpu().numpy()
        if depth_np.ndim == 4:
            depth_np = depth_np[..., 0]

        # 1) Raw lidar (deterministic): distance -> log(distance)
        hits_w = lidar.data.ray_hits_w
        pos_w = lidar.data.pos_w
        d_raw = torch.linalg.norm(hits_w - pos_w.unsqueeze(1), dim=-1)
        d_raw = torch.nan_to_num(d_raw, nan=6.0, posinf=6.0, neginf=0.0)
        d_raw = torch.clamp(d_raw, min=0.01, max=6.0)
        lidar_raw_np = torch.log(d_raw).detach().cpu().numpy()

        # 2) Policy-observation lidar (stochastic): illusion + natural log.
        lidar_obs = go2_mdp.lidar_distances_with_illusion(
            env,
            sensor_cfg=SceneEntityCfg("lidar"),
            command_name="pose_command",
            margin=0.3,
            max_distance=6.0,
            asset_cfg=SceneEntityCfg("robot"),
            use_log=True,
            log_eps=0.01,
        )
        lidar_obs_np = lidar_obs.detach().cpu().numpy()

        for env_id in range(env.num_envs):
            save_name = f"robot_{env_id}_step{i}"
            if depth_np.ndim == 3:
                cam_data = depth_np[env_id].squeeze()
            else:
                cam_data = depth_np.squeeze()
            # Depth sensor can output inf; sanitize and keep within training range.
            cam_data = np.nan_to_num(cam_data, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=DEPTH_MIN)
            cam_data = np.clip(cam_data, float(DEPTH_MIN), float(DEPTH_MAX))
            raw_label = lidar_raw_np[env_id] if lidar_raw_np.ndim == 2 else lidar_raw_np
            obs_label = lidar_obs_np[env_id] if lidar_obs_np.ndim == 2 else lidar_obs_np
            labels_raw[save_name] = np.asarray(raw_label, dtype=np.float32).flatten()
            labels_obs[save_name] = np.asarray(obs_label, dtype=np.float32).flatten()
            np.save(os.path.join(log_folder, save_name + ".npy"), cam_data.astype(np.float32))
        if i % (sample_every * 10) == sample_offset:
            print(f"Saved samples up to step {i}")

    raw_label_path = os.path.join(log_folder, "label_raw.pkl")
    obs_label_path = os.path.join(log_folder, "label_obs.pkl")
    with open(raw_label_path, "wb") as f:
        pickle.dump(labels_raw, f)
    with open(obs_label_path, "wb") as f:
        pickle.dump(labels_obs, f)
    print(f"Done. Saved {len(labels_raw)} raw labels to {raw_label_path}")
    print(f"Done. Saved {len(labels_obs)} obs labels to {obs_label_path}")
    env.close()


def main():
    log_root = args_cli.log_root or os.path.join(ABS_ROOT, "logs", "rec_cam_go2")
    env_cfg = Go2CamRecEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    collect(
        env_cfg=env_cfg,
        policy_path=args_cli.policy,
        log_root=log_root,
        shift=args_cli.shift,
        n_steps=args_cli.n_steps,
        sample_every=args_cli.sample_every,
        sample_offset=args_cli.sample_offset,
        device=args_cli.device,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
