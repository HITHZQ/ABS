"""
Go2 testbed: deterministic evaluation playground for policies (and future RA variants).

Key idea:
- fixed obstacle layouts (from buffer)
- fixed command distribution
- report success/failure/timeout rates and mean episode length

Run:
  ./isaaclab.sh -p source/isaaclab/ABS-main/go2/testbed.py --policy /path/to/exported/policy.pt --headless
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import torch
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

# -----------------------------------------------------------------------------
# App launch first (required by Isaac Sim runtime)
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Go2 testbed: fixed-layout evaluation.")
parser.add_argument("--policy", type=str, required=True, help="TorchScript policy.pt (exported)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=4000, help="Total steps to run")
parser.add_argument("--seed", type=int, default=0, help="Single-seed run (ignored if --seeds is provided)")
parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for benchmark runs, e.g. 0,1,2,3")
parser.add_argument("--difficulty", type=int, default=1, choices=[0, 1, 2], help="Obstacle layout difficulty preset")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--fixed_obstacle_yaw", action="store_true", help="Use fixed yaw=0 for obstacles (more deterministic)")
parser.add_argument("--out_json", type=str, default=None, help="Optional JSON output path for results")

# RA (depth->ray) evaluation: replace policy lidar obs with predicted rays
parser.add_argument("--ra_model", type=str, default=None, help="TorchScript depth->ray model (.pt). Enables RA mode.")
parser.add_argument("--ra_mode", type=str, default="none", choices=["none", "pred_only", "fuse"], help="How to use RA rays")
parser.add_argument("--ra_num_rays", type=int, default=11)
parser.add_argument("--ra_fuse_alpha", type=float, default=1.0, help="fuse: alpha*pred + (1-alpha)*true")

# Recovery + gate (Suggestion B: train recovery on true sensors, deploy with RA rays)
parser.add_argument("--recovery_policy", type=str, default=None, help="Optional TorchScript recovery policy.pt")
parser.add_argument("--gate_enter_m", type=float, default=0.9, help="Enter recovery if min ray < this")
parser.add_argument("--gate_exit_m", type=float, default=1.2, help="Exit recovery if min ray > this")
parser.add_argument("--gate_hold_steps", type=int, default=12, help="Minimum hold steps after a switch")
parser.add_argument("--gate_blend_steps", type=int, default=8, help="Action blend steps at switching edge")
parser.add_argument("--twist_v_back_max", type=float, default=1.2)
parser.add_argument("--twist_wz_max", type=float, default=2.5)
parser.add_argument("--twist_d_stop", type=float, default=0.35)
parser.add_argument("--twist_turn_gain", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# imports after app launch
from go2.velocity_env_cfg import RobotPlayEnvCfg, RobotSceneCfg  # noqa: E402
from go2 import mdp  # noqa: E402
from go2.mdp.twist_from_rays import twist_command_from_rays  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

# -----------------------------------------------------------------------------
# Layout presets
# -----------------------------------------------------------------------------

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

CAMREC_CAMERA = {
    "width": 80,
    "height": 60,
    "fov": 90.0,
    "clipping_range": (0.1, 6.0),
    "pos": (-0.05, 0.0, 0.0),
}


def _make_layout_buffer(num_envs: int, difficulty: int, device: str) -> torch.Tensor:
    """Return obstacle xy buffer of shape (num_envs, 2, num_obj)."""
    num_obj = len(OBJECT_NAMES)
    buf = torch.zeros((num_envs, 2, num_obj), device=device, dtype=torch.float32)

    # Simple hand-crafted presets: put cylinders roughly along the corridor.
    if difficulty == 0:
        xs = torch.linspace(2.5, 6.5, num_obj, device=device)
        ys = torch.zeros(num_obj, device=device)
    elif difficulty == 1:
        xs = torch.linspace(2.0, 7.0, num_obj, device=device)
        ys = torch.tensor([0.0, 0.7, -0.7, 1.1, -1.1, 0.4, -0.4, 0.0], device=device)
    else:
        xs = torch.linspace(2.0, 7.5, num_obj, device=device)
        ys = torch.tensor([1.2, -1.2, 0.9, -0.9, 0.6, -0.6, 0.3, -0.3], device=device)

    buf[:, 0, :] = xs.unsqueeze(0).repeat(num_envs, 1)
    buf[:, 1, :] = ys.unsqueeze(0).repeat(num_envs, 1)
    return buf


# -----------------------------------------------------------------------------
# Testbed env config
# -----------------------------------------------------------------------------

@configclass
class Go2TestbedSceneCfg(RobotSceneCfg):
    """Go2 scene + depth camera (for RA evaluation)."""

    def __post_init__(self):
        super().__post_init__()
        # Depth camera for RA: mount similar to lidar (same as camrec)
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
class TestbedEventCfg:
    """Events for testbed: use obstacle positions from buffer instead of curriculum."""

    set_cylinders_kinematic = EventTerm(
        func=mdp.set_cylinders_kinematic_at_startup,
        mode="startup",
        params={"obstacle_names": OBJECT_NAMES},
    )

    reset_cylinder_obstacles = EventTerm(
        func=mdp.reset_cylinder_obstacles_from_buffer,
        mode="reset",
        params={
            "obstacle_names": OBJECT_NAMES,
            "buffer_attr": "_test_obstacle_positions",
            "default_z": 0.25,
            "fixed_yaw": 0.0 if args_cli.fixed_obstacle_yaw else None,
        },
    )


@configclass
class Go2TestbedEnvCfg(RobotPlayEnvCfg):
    """Play cfg + deterministic obstacles + easier visualization defaults."""

    scene: RobotSceneCfg = replace(Go2TestbedSceneCfg(), num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.events = TestbedEventCfg()
        # Benchmark-style fixed command distribution (reproducible A/B).
        self.commands.pose_command.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.pose_command.ranges = replace(
            self.commands.pose_command.ranges,
            pos_x=(6.0, 7.5),
            pos_y=(-1.5, 1.5),
            heading=(0.0, 0.0),
        )


def _get_obs_tensor(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    # If we ever get a dict of terms here, keep ordering deterministic.
    if isinstance(obs, dict):
        return torch.cat([obs[k].flatten(1) for k in sorted(obs.keys())], dim=1)
    raise TypeError(f"Unsupported obs type: {type(obs)}")

def _parse_seeds() -> list[int]:
    if args_cli.seeds is None or str(args_cli.seeds).strip() == "":
        return [int(args_cli.seed)]
    return [int(s.strip()) for s in str(args_cli.seeds).split(",") if s.strip() != ""]


def _safe_mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return float(m), float(v**0.5)


def _compute_ra_rays_log(depth_cam, device: torch.device, ra_model, num_rays: int) -> torch.Tensor:
    """Return predicted rays in log-space, shape (num_envs, num_rays)."""
    depth_data = depth_cam.data.output["depth"]
    if depth_data is None:
        raise RuntimeError("depth_cam depth output is None. Ensure --enable_cameras and RTX sensors enabled.")
    depth = depth_data
    if depth.dim() == 4:
        depth = depth[..., 0]
    # sanitize and log
    depth = torch.nan_to_num(depth, nan=CAMREC_CAMERA["clipping_range"][1], posinf=CAMREC_CAMERA["clipping_range"][1], neginf=CAMREC_CAMERA["clipping_range"][0])
    depth = torch.clamp(depth, min=float(CAMREC_CAMERA["clipping_range"][0]), max=float(CAMREC_CAMERA["clipping_range"][1]))
    depth = torch.log(torch.clamp(depth, min=0.01))
    x = depth.to(device).unsqueeze(1).repeat(1, 3, 1, 1)
    y = ra_model(x)
    if y.shape[-1] != num_rays:
        raise RuntimeError(f"RA model output dim {y.shape[-1]} != ra_num_rays {num_rays}")
    return y


def _rays_log_to_m(rays_log: torch.Tensor) -> torch.Tensor:
    # rays in our pipeline are log-scaled distances
    return torch.exp(rays_log)


def _build_recovery_obs(env: ManagerBasedRLEnv, twist_cmd: torch.Tensor) -> torch.Tensor:
    """Construct recovery-style observation (E,49) from env state.

    Layout: foot_contacts(4) + base_ang_vel(3) + projected_gravity(3) + twist(3)
            + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)
    """
    from isaaclab.managers import SceneEntityCfg

    fc = mdp.foot_contacts_binary(env, sensor_cfg=SceneEntityCfg("contact_forces"), asset_cfg=SceneEntityCfg("robot"))
    ang = mdp.base_ang_vel(env)
    g = mdp.projected_gravity(env)
    q = mdp.joint_pos_rel(env)
    qd = mdp.joint_vel_rel(env)
    a = mdp.last_action(env)
    return torch.cat([fc, ang, g, twist_cmd, q, qd, a], dim=1)


def _gate_update(
    mode_is_recovery: torch.Tensor,
    hold: torch.Tensor,
    dmin: torch.Tensor,
    *,
    enter_m: float,
    exit_m: float,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hysteresis gate with hold timer. mode_is_recovery is (E,) bool tensor."""
    hold = torch.clamp(hold - 1, min=0)
    can_switch = hold == 0
    to_recovery = can_switch & (~mode_is_recovery) & (dmin < float(enter_m))
    to_agile = can_switch & (mode_is_recovery) & (dmin > float(exit_m))
    mode_is_recovery = torch.where(to_recovery, torch.ones_like(mode_is_recovery, dtype=torch.bool), mode_is_recovery)
    mode_is_recovery = torch.where(to_agile, torch.zeros_like(mode_is_recovery, dtype=torch.bool), mode_is_recovery)
    switched = to_recovery | to_agile
    hold = torch.where(switched, torch.full_like(hold, int(hold_steps)), hold)
    return mode_is_recovery, hold


def _run_once(seed: int) -> dict:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    # create env
    env_cfg = Go2TestbedEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.seed = int(seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # attach obstacle buffer to env (read by reset event)
    env._test_obstacle_positions = _make_layout_buffer(args_cli.num_envs, args_cli.difficulty, env.device)

    # load policy
    policy = torch.jit.load(args_cli.policy, map_location=args_cli.device)
    policy.eval()

    recovery_policy = None
    if args_cli.recovery_policy is not None:
        recovery_policy = torch.jit.load(args_cli.recovery_policy, map_location=args_cli.device)
        recovery_policy.eval()

    # optional RA model
    use_ra = args_cli.ra_mode != "none" and args_cli.ra_model is not None
    ra_model = None
    depth_cam = None
    if use_ra:
        ra_model = torch.jit.load(args_cli.ra_model, map_location=args_cli.device)
        ra_model.eval()
        try:
            depth_cam = env.scene["depth_cam"]
        except KeyError as e:
            raise RuntimeError("depth_cam not found in scene. Ensure testbed scene includes CameraCfg.") from e

    # reset
    obs, _ = env.reset()
    obs_t = _get_obs_tensor(obs)

    # stats
    ep_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    n_done = 0
    n_success = 0
    done_steps_sum = 0
    success_steps_sum = 0
    n_timeout = 0
    fail_counts = {k: 0 for k in env.termination_manager.active_terms}

    # policy lidar slice (by construction of go2 policy obs: lidar_ranges is last 11 dims)
    ray_dim = int(args_cli.ra_num_rays)
    if obs_t.shape[1] < ray_dim:
        raise RuntimeError(f"obs dim {obs_t.shape[1]} < ra_num_rays {ray_dim}")
    ray_slice = slice(-ray_dim, None)

    # gate state
    mode_is_recovery = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    hold = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    blend = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    for step in range(int(args_cli.n_steps)):
        # compute rays (log) either from policy obs (true lidar) or RA model
        rays_log = obs_t[:, ray_slice]
        if use_ra:
            pred = _compute_ra_rays_log(depth_cam, env.device, ra_model, ray_dim)
            if args_cli.ra_mode == "pred_only":
                rays_log = pred
            else:
                alpha = float(args_cli.ra_fuse_alpha)
                rays_log = alpha * pred + (1.0 - alpha) * rays_log

        # update gate based on min ray distance (meters)
        rays_m = _rays_log_to_m(rays_log)
        dmin = torch.amin(rays_m, dim=1)
        if recovery_policy is not None:
            prev = mode_is_recovery.clone()
            mode_is_recovery, hold = _gate_update(
                mode_is_recovery,
                hold,
                dmin,
                enter_m=float(args_cli.gate_enter_m),
                exit_m=float(args_cli.gate_exit_m),
                hold_steps=int(args_cli.gate_hold_steps),
            )
            switched = mode_is_recovery ^ prev
            blend = torch.where(switched, torch.full_like(blend, int(args_cli.gate_blend_steps)), blend)
            blend = torch.clamp(blend - 1, min=0)

        with torch.inference_mode():
            actions_agile = policy(obs_t)
            actions = actions_agile
            if recovery_policy is not None:
                twist_cmd = twist_command_from_rays(
                    rays_m,
                    v_back_max=float(args_cli.twist_v_back_max),
                    wz_max=float(args_cli.twist_wz_max),
                    d_stop=float(args_cli.twist_d_stop),
                    d_enter=float(args_cli.gate_enter_m),
                    turn_gain=float(args_cli.twist_turn_gain),
                )
                obs_rec = _build_recovery_obs(env, twist_cmd)
                actions_rec = recovery_policy(obs_rec)

                # blend at switch boundary, otherwise hard gate
                w = (blend.float() / max(float(args_cli.gate_blend_steps), 1.0)).clamp(0.0, 1.0).unsqueeze(1)
                use_rec = mode_is_recovery.unsqueeze(1).float()
                # when in recovery: action = (1-w)*rec + w*agile (fade-in)
                # when in agile:    action = (1-w)*agile + w*rec (fade-out)
                actions = torch.where(
                    use_rec > 0.5,
                    (1.0 - w) * actions_rec + w * actions_agile,
                    (1.0 - w) * actions_agile + w * actions_rec,
                )

        obs, _, terminated, truncated, _ = env.step(actions)
        obs_t = _get_obs_tensor(obs)
        dones = torch.logical_or(terminated, truncated)
        ep_steps += 1

        if torch.any(dones):
            done_ids = torch.where(dones)[0]
            n_done += int(done_ids.numel())
            done_steps_sum += int(ep_steps[done_ids].sum().item())

            if "success" in env.termination_manager.active_terms:
                succ = env.termination_manager.get_term("success")[done_ids]
                n_success += int(succ.sum().item())
                if torch.any(succ):
                    success_steps_sum += int(ep_steps[done_ids][succ].sum().item())

            n_timeout += int(truncated[done_ids].sum().item())

            for term in env.termination_manager.active_terms:
                term_hit = env.termination_manager.get_term(term)[done_ids]
                fail_counts[term] += int(term_hit.sum().item())

            ep_steps[done_ids] = 0

    sr = (n_success / max(1, n_done)) if n_done > 0 else 0.0
    timeout_rate = (n_timeout / max(1, n_done)) if n_done > 0 else 0.0
    mean_len = (done_steps_sum / max(1, n_done)) if n_done > 0 else 0.0
    mean_succ_len = (success_steps_sum / max(1, n_success)) if n_success > 0 else 0.0
    base_contact_rate = (fail_counts.get("base_contact", 0) / max(1, n_done)) if n_done > 0 else 0.0

    result = {
        "seed": int(seed),
        "policy": args_cli.policy,
        "difficulty": int(args_cli.difficulty),
        "num_envs": int(args_cli.num_envs),
        "n_steps": int(args_cli.n_steps),
        "ra_model": args_cli.ra_model,
        "ra_mode": args_cli.ra_mode,
        "ra_fuse_alpha": float(args_cli.ra_fuse_alpha),
        "episodes": int(n_done),
        "success_rate": float(sr),
        "timeout_rate": float(timeout_rate),
        "base_contact_rate": float(base_contact_rate),
        "mean_ep_len_steps": float(mean_len),
        "mean_success_steps": float(mean_succ_len),
        "termination_counts": {k: int(v) for k, v in fail_counts.items()},
    }

    env.close()
    return result


def main():
    seeds = _parse_seeds()
    runs = []
    for s in seeds:
        r = _run_once(s)
        runs.append(r)
        print(
            f"[seed {s}] episodes={r['episodes']} success_rate={r['success_rate']:.4f} "
            f"timeout_rate={r['timeout_rate']:.4f} base_contact_rate={r['base_contact_rate']:.4f} "
            f"mean_success_steps={r['mean_success_steps']:.1f}"
        )

    # aggregate
    agg = {}
    for k in ["success_rate", "timeout_rate", "base_contact_rate", "mean_ep_len_steps", "mean_success_steps"]:
        xs = [float(r[k]) for r in runs]
        m, sd = _safe_mean_std(xs)
        agg[k] = {"mean": m, "std": sd}

    out = {
        "config": {
            "policy": args_cli.policy,
            "difficulty": int(args_cli.difficulty),
            "num_envs": int(args_cli.num_envs),
            "n_steps": int(args_cli.n_steps),
            "seeds": seeds,
            "fixed_obstacle_yaw": bool(args_cli.fixed_obstacle_yaw),
            "ra_model": args_cli.ra_model,
            "ra_mode": args_cli.ra_mode,
            "ra_num_rays": int(args_cli.ra_num_rays),
            "ra_fuse_alpha": float(args_cli.ra_fuse_alpha),
        },
        "runs": runs,
        "aggregate": agg,
    }

    print("============================================================")
    print("Go2 Testbed Benchmark Summary (aggregate)")
    print("policy   :", args_cli.policy)
    print("seeds    :", seeds)
    for k, v in agg.items():
        print(f"{k}: mean={v['mean']:.4f} std={v['std']:.4f}")
    print("============================================================")

    if args_cli.out_json is not None:
        with open(args_cli.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Saved JSON to:", args_cli.out_json)

    simulation_app.close()


if __name__ == "__main__":
    main()

