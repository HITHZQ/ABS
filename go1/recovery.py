# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.

"""
Recovery Policy Configuration

The recovery policy tracks twist commands (tw_c) for fast backup/shielding.
- Observation o_Rec: c_f, ω, g, tw_c, q, q̇, a. No exteroception.
- Action space: same as agile — 12-D joint targets; MLP.
- Commands: v^c_x U(−1.5,1.5), v^c_y U(−0.3,0.3), ω^c_z U(−3,3).
- Sim vs agile: episode 2 s; init roll/pitch U(−π/6,π/6), vx U(−0.5,5.5), ω U(−1,1);
  obs noises & dynamics same as agile (Table II). Curriculum: promote if vel error < 0.7 σ_linvel, demote if fall.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.noise import GaussianNoiseCfg

# Import MDP functions
import isaaclab.envs.mdp as mdp
import go1.mdp as mdp

##
# Scene Configuration
##

@configclass
class Go1RecoverySceneCfg(InteractiveSceneCfg):
    """Configuration for the Go1 Recovery scene - simplified for fast twist tracking."""

    # 1. Terrain: Flat ground only for fast recovery
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            num_rows=1,
            num_cols=1,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=1.0),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 2. Robot - Go1 (same as agile)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/adam/IsaacLab/source/extension/robot/go1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.37),
            joint_pos={
                "FL_hip_joint": 0.0, "RL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.8, "RL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RR_thigh_joint": 0.8,
                "FL_calf_joint": -1.5, "RL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RR_calf_joint": -1.5,
            },
        ),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_joint"],
                effort_limit=32,
                saturation_effort=32,
                velocity_limit=31.4159,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
        },
    )

    # 3. Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# Actions
##

@configclass
class ActionsCfg:
    """Action space (same as agile policy).

    Recovery uses the same action space as agile: 12-D joint position targets.
    Policy network: MLP.
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )

##
# Observations
##

@configclass
class ObservationsCfg:
    """Recovery observation space o_Rec (no exteroception).

    Paper: o_Rec = (c_f, ω, g, tw_c, q, q̇, a).
    - c_f: foot contacts (4)
    - ω: base angular velocities (3)
    - g: projected gravity in base frame (3)
    - tw_c: twist commands [lin_vel_x, lin_vel_y, ang_vel_yaw] (3; “only non-zero” → we use all 3)
    - q: joint positions (12)
    - q̇: joint velocities (12)
    - a: previous-frame actions (12)

    Total: 4 + 3 + 3 + 3 + 12 + 12 + 12 = 49. No lidar or other exteroception.
    """
    @configclass
    class PolicyCfg(ObsGroup):
        foot_contacts = ObsTerm(
            func=mdp.foot_contacts,
            params={
                "threshold": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            },
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_with_encoder_bias, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoiseCfg(mean=0.0, std=1.5))
        actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.policy = self.PolicyCfg()

##
# Events (Domain Randomization)
##

@configclass
class EventCfg:
    """Recovery DR: same obs noises & dynamics as agile (Table II). Episode 2 s; init roll/pitch U(−π/6,π/6);
    vx U(−0.5,5.5) m/s, ω U(−1,1) rad/s. Curriculum: promote if vel error < 0.7 σ_linvel, demote if fall."""

    # Dynamics (same as agile Table II): friction U(0.4, 1.1), added mass U(−1.5, 1.5), encoder bias U(−0.08, 0.08)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.4, 1.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 2.5),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # Init: x=0, y=0; roll, pitch U(−π/6, π/6); yaw U(−π, π); vx U(−0.5, 5.5), vy,vz ±0.5; ω U(−1, 1)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
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

    sample_encoder_bias = EventTerm(
        func=mdp.sample_joint_encoder_bias,
        mode="reset",
        params={
            "bias_range": (-0.08, 0.08),
            "bias_attr": "_encoder_joint_pos_bias",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

##
# Commands
##

@configclass
class CommandsCfg:
    """Twist commands tw_c. Ranges: v^c_x U(−1.5,1.5) m/s, v^c_y U(−0.3,0.3) m/s, ω^c_z U(−3,3) rad/s."""
    twist_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_yaw=(-3.0, 3.0),
        ),
    )

##
# Rewards
##

@configclass
class RewardsCfg:
    """Recovery rewards: penalty, task, regularization (same as agile except allow knee contacts).
    Task: r_task = 10·r_linvel − 0.5·r_angvel + 5·r_alive − 0.1·r_posture (twist tracking, alive, posture).
    Penalty: collision (base, thighs, feet; no calf — allow knee for max deceleration), termination.
    Regularization: same as agile (unified regularization_reward).
    """
    # -- Task rewards (twist tracking): 10·r_linvel − 0.5·r_angvel + 5·r_alive − 0.1·r_posture
    r_linvel = RewTerm(
        func=mdp.linear_velocity_tracking_reward_xy,
        weight=10.0,
        params={
            "command_name": "twist_command",
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma_linvel": 0.5,
        },
    )
    r_angvel = RewTerm(
        func=mdp.angular_velocity_tracking_penalty,
        weight=-0.5,
        params={"command_name": "twist_command", "asset_cfg": SceneEntityCfg("robot")},
    )
    r_alive = RewTerm(func=mdp.alive_bonus, weight=5.0)
    r_posture = RewTerm(
        func=mdp.posture_penalty_recovery,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Penalty rewards (same as agile except allow knee/calf contact)
    collision = RewTerm(
        func=mdp.undesired_contacts_comprehensive,
        weight=-100.0,
        params={
            "threshold": 1.0,
            "horizontal_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "base_body_name": "trunk",
            "thigh_body_pattern": ".*thigh",
            "calf_body_pattern": ".*calf",
            "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "penalize_calf": False,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # -- Regularization (same as agile)
    regularization = RewTerm(
        func=mdp.regularization_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_limit": 31.4159,
            "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "contact_threshold": 1.0,
        },
    )

##
# Terminations
##

@configclass
class TerminationsCfg:
    """Recovery terminations. Base contact = fall (used for curriculum demotion). Episode 2 s."""
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot", body_names="trunk"), "threshold": 1.0},
    )
    time_out = TermTerm(func=mdp.time_out, params={"time_out": 2.0})

##
# Environment Config
##

@configclass
class Go1RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go1 Recovery environment - fast twist command tracking."""
    
    # Scene settings
    scene: Go1RecoverySceneCfg = Go1RecoverySceneCfg(num_envs=1280, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # Physics settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )
    
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
