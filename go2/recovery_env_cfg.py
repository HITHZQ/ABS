"""
Go2 Recovery Policy Environment (modeled after Go1 recovery).

Recovery policy goal: robustly regain stable locomotion / braking under disturbed states.
- Observation: no exteroception (no lidar). Uses contacts + proprioception + twist command.
- Command: twist_command (vx, vy, wz) in a moderate range.
- Episode: short (2 s) with randomized initial orientation and velocities.

This file intentionally mirrors the structure of `go1/recovery.py`, but uses the Go2 robot
and keeps the implementation self-contained within the Go2 package.
"""

from __future__ import annotations

import math
import re

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG

from . import mdp


# -----------------------------------------------------------------------------
# Terrain (flat + rough + low obstacles)
# -----------------------------------------------------------------------------

RECOVERY_TERRAIN_MAX_HEIGHT_DIFF_M = 0.07

GO2_RECOVERY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=False,
    size=(11.0, 5.0),
    num_rows=64,
    num_cols=64,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.34),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.33,
            noise_range=(0.0, RECOVERY_TERRAIN_MAX_HEIGHT_DIFF_M / 2.0),
            noise_step=0.01,
            border_width=0.25,
        ),
        "low_stumbling_blocks": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.33,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.12, 0.4),
            obstacle_height_range=(0.0, RECOVERY_TERRAIN_MAX_HEIGHT_DIFF_M),
            num_obstacles=40,
            platform_width=1.0,
        ),
    },
)


# -----------------------------------------------------------------------------
# Local MDP helpers (to avoid go1 dependency)
# -----------------------------------------------------------------------------

def _alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    m = getattr(env, "termination_manager", None)
    if m is None:
        return torch.ones(env.num_envs, device=env.device)
    t = getattr(m, "terminated", None)
    if t is None:
        t = getattr(m, "_terminated_buf", None)
    if t is None:
        return torch.ones(env.num_envs, device=env.device)
    return 1.0 - t.float()


def _angular_velocity_tracking_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    actual = asset.data.root_ang_vel_w[:, 2]
    err = actual - cmd[:, 2]
    return torch.square(err)


def _linear_velocity_tracking_reward_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "twist_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_linvel: float = 0.5,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    base_quat = asset.data.root_quat_w
    base_lin_vel_w = asset.data.root_lin_vel_w[:, :2]
    cos_yaw = base_quat[:, 0] ** 2 + base_quat[:, 3] ** 2 - base_quat[:, 1] ** 2 - base_quat[:, 2] ** 2
    sin_yaw = 2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2])
    vx = base_lin_vel_w[:, 0] * cos_yaw + base_lin_vel_w[:, 1] * sin_yaw
    vy = -base_lin_vel_w[:, 0] * sin_yaw + base_lin_vel_w[:, 1] * cos_yaw
    ex = cmd[:, 0] - vx
    ey = cmd[:, 1] - vy
    s2 = float(sigma_linvel) ** 2
    return torch.exp(-((ex**2 + ey**2) / s2))


def _posture_penalty_recovery(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    joint_names = asset.joint_names
    if nominal_joint_pos is None:
        nominal_joint_pos = {
            "FL_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 1.0,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.7,
            "RL_calf_joint": -1.7,
            "FR_calf_joint": -1.7,
            "RR_calf_joint": -1.7,
        }
    q_bar = torch.zeros_like(q)
    default = asset.data.default_joint_pos
    for i, name in enumerate(joint_names):
        val = nominal_joint_pos.get(name)
        q_bar[:, i] = float(val) if val is not None else default[0, i]
    return torch.sum(torch.abs(q - q_bar), dim=-1)


def _foot_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_pattern: str = ".*_foot",
) -> torch.Tensor:
    """Binary foot contact vector (num_envs, num_feet)."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    body_names = asset.body_names
    foot_ids = [i for i, n in enumerate(body_names) if re.match(foot_body_pattern, n)]
    if len(foot_ids) == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)
    # sort by foot name for determinism
    foot_ids = [i for _, i in sorted((body_names[i], i) for i in foot_ids)]
    forces_hist = sensor.data.net_forces_w_history[:, :, foot_ids]  # (E, H, F, 3)
    mags = torch.linalg.norm(forces_hist, dim=-1)  # (E, H, F)
    peak = torch.amax(mags, dim=1)  # (E, F)
    return (peak > float(threshold)).float()


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class Go2RecoverySceneCfg(InteractiveSceneCfg):
    """Terrain + contact sensor + Go2 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_RECOVERY_TERRAINS_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# -----------------------------------------------------------------------------
# Actions / Observations / Commands
# -----------------------------------------------------------------------------


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """o_Rec = (c_f, ω, g, tw_c, q, q̇, a). No lidar."""

    @configclass
    class PolicyCfg(ObsGroup):
        foot_contacts = ObsTerm(
            func=_foot_contacts,
            params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces"), "asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoiseCfg(mean=0.0, std=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    twist_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-3.0, 3.0),
        ),
    )


# -----------------------------------------------------------------------------
# Events / Rewards / Terminations
# -----------------------------------------------------------------------------


@configclass
class EventCfg:
    """Recovery DR: no pushes; random init roll/pitch + velocities."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.4, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.37, 0.37),
                "roll": (-math.pi / 6, math.pi / 6),
                "pitch": (-math.pi / 6, math.pi / 6),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 5.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
        },
    )


@configclass
class RewardsCfg:
    # Task: 10*r_linvel - 0.5*r_angvel + 5*r_alive - 0.1*r_posture
    r_linvel = RewTerm(
        func=_linear_velocity_tracking_reward_xy,
        weight=10.0,
        params={"command_name": "twist_command", "asset_cfg": SceneEntityCfg("robot"), "sigma_linvel": 0.5},
    )
    r_angvel = RewTerm(
        func=_angular_velocity_tracking_penalty,
        weight=-0.5,
        params={"command_name": "twist_command", "asset_cfg": SceneEntityCfg("robot")},
    )
    r_alive = RewTerm(func=_alive_bonus, weight=5.0)
    r_posture = RewTerm(func=_posture_penalty_recovery, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")})

    # Collision penalty (allow calves for max deceleration; keep thighs + base)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base|.*_thigh"), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # Light regularization (keep recovery responsive)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)


@configclass
class TerminationsCfg:
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    time_out = TermTerm(func=mdp.time_out, time_out=True)


# -----------------------------------------------------------------------------
# Environment Config
# -----------------------------------------------------------------------------


@configclass
class Go2RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 recovery env (twist tracking) for training recovery policy."""

    scene: Go2RecoverySceneCfg = Go2RecoverySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 2.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt

