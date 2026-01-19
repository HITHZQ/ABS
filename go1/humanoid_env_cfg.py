# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.

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
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.noise import  GaussianNoiseCfg

# 引入标准库中的奖励和观测函数
import isaaclab.envs.mdp as mdp
import go1.mdp as mdp
##
# Scene Configuration
##

@configclass
class Go1PosSceneCfg(InteractiveSceneCfg):
    """Configuration for the Go1 Position Tracking scene."""

    # 1. 地形 (Terrain) - 对应 terrain_types = ['flat', 'rough', 'low_obst']
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 2. 机器人 (Robot) - Go1
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
                enabled_self_collisions=False, # 对应 self_collisions = 0 (enabled in bitwise usually means 0 is enabled, but here explicit False matches config intent of no self collision)
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.37), # 对应 init_state.pos
            joint_pos={
                # 对应 default_joint_angles
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
                stiffness=30.0, # 对应 control.stiffness
                damping=0.65,   # 对应 control.damping
                friction=0.0,
            ),
        },
    )

    # 3. 传感器 - 对应 Ray2d
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", # 挂载在躯干上
        offset=RayCasterCfg.OffsetCfg(pos=(-0.05, 0.0, 0.0)), # 对应 x_0=-0.05
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            # 对应 theta_start (-pi/4) 到 theta_end (pi/4)
            horizontal_fov_range=(-math.pi/4, math.pi/4), 
            horizontal_res=math.pi/20, # 对应 theta_step
        ),
        max_distance=6.0, # 对应 max_dist
        debug_vis=False,
        mesh_prim_paths=["/World/ground"], # 射线只检测地形
    )

    # 4. 障碍物 - 对应 loaded objects (cylinders)
    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.1,
            height=0.5,
            mass_props=sim_utils.MassPropertiesCfg(0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.25)), # 初始位置将被随机化覆盖
    )

    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, # 对应 action_scale
        use_default_offset=True
    )

##
# Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        # 对应 num_observations = 61 (近似组合)
        
        # 1. Base Linear Velocity (3)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.1))
        
        # 2. Base Angular Velocity (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.2))
        # 3. Projected Gravity (3)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        # 4. Commands (3) - x, y, heading
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        # 5. Joint Positions (12)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        # 6. Joint Velocities (12)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoiseCfg(mean=0.0, std=1.5))
        # 7. Actions (12)
        actions = ObsTerm(func=mdp.last_action)
        # 8. Ray Cast / Lidar (approx 11-15 dims based on resolution)
        lidar_ranges = ObsTerm(
            func=mdp.observations.RayCasterCamera, ####if need, add more
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.2),
            clip=(-100.0, 100.0)
        )

    def __post_init__(self):
        self.policy = self.PolicyCfg()

##
# Events (Domain Randomization)
##

@configclass
class EventCfg:
    """Configuration for events."""
    
    # 摩擦力随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25), # 对应 friction_range ~[-0.2, 1.25] logic adjusted
            "dynamic_friction_range": (0.8, 1.25),
            "num_buckets": 64,
        },
    )

    # 质量随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.5, 1.5), # 对应 added_mass_range
            "operation": "add",
        },
    )

    # 推力 (Push Robots)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 2.5), # 对应 push_interval_s
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    
    # 初始状态随机化
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, # 对应 init_x/y/yaw_range
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )

##
# Commands
##

@configclass
class CommandsCfg:
    """Command specifications for the environment."""
    # 对应原配置中的 pos_1, pos_2 和 heading
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0), # 需根据 curriculum 调整
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-1.5, 7.5), # 对应 pos_1 
            pos_y=(-2.0, 2.0), # 对应 pos_2 (如果非 polar)
            heading=(-0.3, 0.3),
        ),
    )

##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for the environment."""
    # -- Penalty Rewards --
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0, # 极大的惩罚
        params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]),
        "threshold": 1.0,
        },
    )
    
    # -- Task Rewards --
    # 对应 reach_pos_target_soft = 60.0 (使用 L2 距离的 exp 形式近似 soft)
    # possoft
    possoft = RewTerm(
        func=mdp.possoft,
        weight=60.0,
        params={"env": "pose_command", "sigma": 2, "tr_steps": 2},
    )
    # postight
    postight = RewTerm(
        func=mdp.postight,
        weight=60.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    # heading
    heading = RewTerm(
        func=mdp.heading,
        weight=30.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    # stand
    stand = RewTerm(
        func=mdp.stand,
        weight=-10.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    # agile
    agile = RewTerm(
        func=mdp.agile,
        weight=10.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    # stall
    stall = RewTerm(
        func=mdp.stall,
        weight=-20.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    # -- regularization Rewards --

    regularization = RewTerm(
        func=mdp.position_command_error,
        weight=60.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )



    track_pos_l2 = RewTerm(
        func=mdp.position_command_error,
        weight=60.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )
    
    # 对应 reach_heading_target = 30.0
    track_heading_l2 = RewTerm(
        func=mdp.heading_command_error,
        weight=30.0,
        params={"command_name": "pose_command", "target_obj_name": "robot"},
    )

    # -- Penalties --
    # 对应 lin_vel_z = -2.0
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    
    # 对应 ang_vel_xy = -0.05
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # 对应 torques = -0.0005
    torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0005)
    
    # 对应 dof_acc = -2.0e-7
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    

    # 对应 action_rate = -0.01
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Termination logic is handled in Termination Terms, usually not as a negative reward unless specified
    # 对应 termination = -100 (在终止时给予惩罚)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the environment."""
    # 对应 terminate_after_contacts_on = ["base"]
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot", body_names="trunk"), "threshold": 1.0},
    )
    # 超时
    time_out = TermTerm(func=mdp.time_out, params={"time_out": 9.0}) # 对应 episode_length_s

##
# Environment Config
##

@configclass
class Go1PosRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go1 Pos Rough environment."""
    # Scene settings
    scene: Go1PosSceneCfg = Go1PosSceneCfg(num_envs=1280, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # Physics settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, # 对应 sim DT
        render_interval=4, # 对应 decimation=4
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