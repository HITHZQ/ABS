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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfDiscreteObstaclesTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg, MeshRandomGridTerrainCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.noise import  GaussianNoiseCfg

# 引入标准库中的奖励和观测函数
import isaaclab.envs.mdp as mdp
import go1.mdp as mdp

# --------------------------------------------------------------------------------------
# Terrain curriculum (flat / rough / low stumbling blocks)
# --------------------------------------------------------------------------------------
# Target behavior (per your description):
# - Terrains are randomly sampled from: flat, rough, low stumbling blocks
# - Curriculum difficulty level increases from 0 to 9 (10 levels total)
# - For rough terrains and stumbling blocks, peak-to-peak height difference increases from 0cm to 7cm
#
# In Isaac Lab, curriculum difficulty is a float in [0, 1] varying along `num_rows`.
# We map (level 0..9) -> (difficulty 0..1) by setting `num_rows=10` and `difficulty_range=(0, 1)`.
AGILE_TERRAIN_LEVELS = 10  # levels 0..9
AGILE_TERRAIN_MAX_HEIGHT_DIFF_M = 0.07  # 7 cm

# For "rough", we use MeshRandomGridTerrainCfg (trimesh). It perturbs grid heights in [-h, +h].
# To keep peak-to-peak <= 0.07m, use h_max = 0.07 / 2 = 0.035m.
GO1_AGILE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    # Each tile (sub-terrain) size in meters.
    size=(8.0, 8.0),
    # One row per curriculum level.
    num_rows=AGILE_TERRAIN_LEVELS,
    # More columns => more random variety at the same difficulty.
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        # 1) Flat ground
        "flat": MeshPlaneTerrainCfg(proportion=0.34),
        # 2) Rough terrain: low-amplitude random grid heights
        "rough": MeshRandomGridTerrainCfg(
            proportion=0.33,
            # grid_width must not perfectly tile size to keep a positive border width.
            # With size=8.0, grid_width=0.24 → border_width ≈ 0.08 > 0.
            grid_width=0.24,
            grid_height_range=(0.0, AGILE_TERRAIN_MAX_HEIGHT_DIFF_M / 2.0),
            platform_width=1.0,
            holes=False,
        ),
        # 3) Low stumbling blocks: positive-only discrete cuboid obstacles
        "low_stumbling_blocks": HfDiscreteObstaclesTerrainCfg(
            proportion=0.33,
            obstacle_height_mode="fixed",  # fixed => positive-only; choice includes negative pits
            obstacle_width_range=(0.12, 0.4),
            obstacle_height_range=(0.0, AGILE_TERRAIN_MAX_HEIGHT_DIFF_M),
            num_obstacles=40,
            platform_width=1.0,
        ),
    },
)
##
# Scene Configuration
##

@configclass
class Go1AgileSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the Go1 agile (position-tracking) task."""

    # 1. 地形 (Terrain): flat / rough / low stumbling blocks with curriculum (levels 0..9)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO1_AGILE_TERRAINS_CFG,
        # Start from easy levels (0 is easiest). Increase if you want a harder initial mix.
        max_init_terrain_level=0,
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

    # 3. 传感器 - 对应 Ray2d + 接触力
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", # 挂载在躯干上
        offset=RayCasterCfg.OffsetCfg(pos=(-0.05, 0.0, 0.0)), # 对应 x_0=-0.05
        # align rays with robot yaw (replacement for deprecated attach_yaw_only)
        ray_alignment="yaw",
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

    # Contact sensor on all robot links, used for termination and potential rewards.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 4. 障碍物 (Cylinders)
    # Paper: 0~8 cylinders per episode, radius 40cm, uniformly placed in a 11m x 5m rectangle
    # covering origin and goal, with a curriculum (harder -> more obstacles).
    #
    # Implementation detail: we spawn 8 cylinders and at reset we activate a subset (others moved far away).
    cylinder_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=1.0,
            mass_props=sim_utils.MassPropertiesCfg(50.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.25)),
    )
    cylinder_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_1",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_2",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_3",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_4",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_5",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_6",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_7",
        spawn=cylinder_0.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
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
        
        # Observations — Table II noise (Gaussian approx of U(−a,a) via std=a)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.2))   # U(−0.2,0.2) rad/s
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))  # U(−0.05,0.05)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_with_encoder_bias, noise=GaussianNoiseCfg(mean=0.0, std=0.01))  # U(−0.01,0.01) rad
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoiseCfg(mean=0.0, std=1.5))     # U(−1.5,1.5) rad/s
        actions = ObsTerm(func=mdp.last_action)
        # Lidar: illusion + log(ray); Table II log(ray) noise U(−0.2, 0.2)
        lidar_ranges = ObsTerm(
            func=mdp.lidar_distances_with_illusion,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "command_name": "pose_command",
                "margin": 0.3,
                "max_distance": 6.0,
                "asset_cfg": SceneEntityCfg("robot"),
                "use_log": True,
                "log_eps": 0.01,
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.2),
            clip=(-5.0, 2.0),
        )

    def __post_init__(self):
        self.policy = self.PolicyCfg()

##
# Events (Domain Randomization)
##

@configclass
class EventCfg:
    """Domain randomization (Table II): Observation (illusion, noises), Dynamics (ERFI-50, friction, mass, biases), Episode (init, goal, length)."""

    # Table II Dynamics: Friction factor U(0.4, 1.1)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.4, 1.1),
            # In line with other tasks (e.g., locomotion velocity), use a fixed restitution range.
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Table II: Added base mass U(−1.5, 1.5) kg
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

    # Table II Episode: Initial x=0, y=0; yaw U(−π, π); twist U(−0.5, 0.5) m/s or rad/s
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )

    # --- Domain Randomization (Table II). Critical: illusion, ERFI-50, encoder bias. ---
    # Illusion: lidar_distances_with_illusion (obs). Overwrites ray d with U(d_goal+0.3, d) when d > d_goal+0.3.
    # ERFI-50 [Campanaro et al.]: torque perturbations; curriculum by terrain level to avoid impeding early learning.
    # Encoder bias: random joint-position offset to model motor encoders' errors.

    # Table II: Joint position biases U(−0.08, 0.08) rad
    sample_encoder_bias = EventTerm(
        func=mdp.sample_joint_encoder_bias,
        mode="reset",
        params={
            "bias_range": (-0.08, 0.08),
            "bias_attr": "_encoder_joint_pos_bias",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    clear_erfi_torques = EventTerm(
        func=mdp.clear_joint_effort_targets,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Table II: ERFI-50 — 0.78 N·m × difficulty level
    erfi50_torque_perturb = EventTerm(
        func=mdp.apply_erfi50_torque_perturbations,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "erfi_torque_per_level": 0.78,
            "num_levels": 10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 障碍物重置 + curriculum：level 越高，障碍物上限越大（0..8）
    reset_cylinder_obstacles = EventTerm(
        func=mdp.reset_cylinder_obstacles_curriculum,
        mode="reset",
        params={
            "obstacle_names": [
                "cylinder_0",
                "cylinder_1",
                "cylinder_2",
                "cylinder_3",
                "cylinder_4",
                "cylinder_5",
                "cylinder_6",
                "cylinder_7",
            ],
            # 11m x 5m rectangle covering origin (x=0) and typical goal (x up to ~7.5)
            # We use x in [-2.5, 8.5] and y in [-2.5, 2.5].
            "x_range": (-2.5, 8.5),
            "y_range": (-2.5, 2.5),
            "num_levels": 10,
            "max_obstacles": 8,
        },
    )

##
# Commands
##

@configclass
class CommandsCfg:
    """Command specifications for the environment. Table II Episode: Goal and heading."""
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(1.5, 7.5),   # Table II: x_goal ~ U(1.5, 7.5) m
            pos_y=(-2.0, 2.0), # Table II: y_goal ~ U(−2, 2) m
            heading=(-0.3, 0.3),  # Table II: arctan2(y_goal,x_goal) + U(−0.3, 0.3); approx via indep.
        ),
    )

##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for the environment."""
    # -- Penalty Rewards --
    # Undesired contacts: collisions on base, thighs, calves, and horizontal collisions on feet
    collision = RewTerm(
        func=mdp.undesired_contacts_comprehensive,
        weight=-100.0, # 极大的惩罚
        params={
            "threshold": 1.0,
            "horizontal_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "base_body_name": "trunk",
            "thigh_body_pattern": ".*thigh",
            "calf_body_pattern": ".*calf",
            "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        },
    )
    
    # -- Task Rewards --
    # 对应 reach_pos_target_soft = 60.0 (使用 L2 距离的 exp 形式近似 soft)
    # possoft - Soft position tracking to encourage exploration for goal reaching
    # Formula: r(possoft) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr)
    # Only active in the last Tr seconds of the episode (t > T - Tr)
    # 
    # Purpose: Robot only needs to reach goal before T - Tr to maximize tracking rewards,
    #          freeing from explicit motion constraints (e.g., target velocities) that limit agility.
    # Parameters:
    #   - σsoft = 2 m (normalizes tracking errors)
    #   - Tr = 2 s (time threshold)
    #   - Error: distance to goal
    possoft = RewTerm(
        func=mdp.possoft,
        weight=60.0,
        params={
            "command_name": "pose_command",
            "sigma": 2.0,  # σsoft = 2 m (normalization parameter)
            "tr_steps": 2.0,  # Tr = 2 s (time threshold)
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # postight - Tight position tracking for precise goal reaching
    # Formula: r(postight) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr)
    # Only active in the last Tr seconds of the episode (t > T - Tr)
    # 
    # Purpose: Provides tighter position tracking with smaller sigma and shorter time window
    #          compared to possoft, encouraging more precise goal reaching.
    # Parameters:
    #   - σtight = 0.5 m (normalizes tracking errors, tighter than possoft)
    #   - Tr = 1 s (time threshold, shorter than possoft)
    #   - Error: distance to goal
    postight = RewTerm(
        func=mdp.postight,
        weight=60.0,
        params={
            "command_name": "pose_command",
            "sigma": 0.5,  # σtight = 0.5 m (normalization parameter, tighter than possoft)
            "tr_steps": 1.0,  # Tr = 1 s (time threshold, shorter than possoft)
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # heading - Heading tracking for goal orientation
    # Formula: r(heading) = 1/(1+||error/sigma||^2) * (1/Tr) * (if t>T-Tr) * (if dist <= σsoft)
    # Only active in the last Tr seconds of the episode (t > T - Tr) AND when close to goal
    # 
    # Purpose: Tracks relative yaw angle to goal heading. Disabled when distance to goal > σsoft
    #          so that collision avoidance is not affected.
    # Parameters:
    #   - σheading = 1 rad (normalizes tracking errors)
    #   - Tr = 2 s (time threshold)
    #   - Error: relative yaw angle to the goal heading
    #   - Disabled when distance to goal > σsoft (2 m)
    heading = RewTerm(
        func=mdp.heading,
        weight=30.0,
        params={
            "command_name": "pose_command",
            "sigma": 1.0,  # σheading = 1 rad (normalization parameter)
            "tr_steps": 2.0,  # Tr = 2 s (time threshold)
            "sigma_soft": 2.0,  # σsoft = 2 m (distance threshold to disable heading reward)
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # stand - Stand reward to encourage maintaining standing pose when close to goal
    # Formula: r(stand) = ||q - q̄||₁ * (1(if t > T - Tr,stand) / Tr,stand) * 1(d_goal < σtight)
    # Only active in the last Tr,stand seconds (t > T - Tr,stand) AND when very close to goal (d < σtight)
    # 
    # Purpose: Penalizes deviation from nominal standing joint positions when very close to goal
    #          in the final seconds, encouraging a stable standing pose.
    # Parameters:
    #   - Tr,stand = 1 s (time threshold)
    #   - σtight = 0.5 m (distance threshold, same as postight)
    #   - q̄: nominal joint positions for standing
    stand = RewTerm(
        func=mdp.stand,
        weight=-10.0,
        params={
            "command_name": "pose_command",
            "tr_stand": 1.0,  # Tr,stand = 1 s (time threshold)
            "sigma_tight": 0.5,  # σtight = 0.5 m (distance threshold)
            "asset_cfg": SceneEntityCfg("robot"),
            "nominal_joint_pos": {
                "FL_hip_joint": 0.0,
                "RL_hip_joint": 0.0,
                "FR_hip_joint": 0.0,
                "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.8,
                "RL_thigh_joint": 0.8,
                "FR_thigh_joint": 0.8,
                "RR_thigh_joint": 0.8,
                "FL_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RR_calf_joint": -1.5,
            },
        },
    )
    # agile - Agile reward to encourage fast forward motion or staying at goal
    # Formula: r(agile) = max{relu(vx/vmax) * 1(correct direction), 1(d_goal < σtight)}
    # 
    # Purpose: Encourages robot to either run fast forward in correct direction or stay at goal.
    #          To maximize this term, the robot has to either run fast or stay at the goal.
    # Parameters:
    #   - vx: forward velocity in robot base frame
    #   - vmax = 4.5 m/s (upper bound of forward velocity, cannot be reached)
    #   - correct direction: angle between robot heading and robot-goal line < 105°
    #   - σtight = 0.5 m (distance threshold, same as postight)
    agile = RewTerm(
        func=mdp.agile,
        weight=10.0,
        params={
            "command_name": "pose_command",
            "vmax": 4.5,  # vmax = 4.5 m/s (upper bound)
            "sigma_tight": 0.5,  # σtight = 0.5 m (distance threshold)
            "correct_direction_threshold": 105.0,  # 105° angle threshold
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # stall - Stall penalty to penalize robot for time waste
    # Formula: r(stall) = 1 if (robot is static) AND (d_goal > σsoft) AND (not correct direction)
    # 
    # Purpose: Penalizes robot for staying static when far from goal and not heading toward goal.
    #          This prevents the robot from wasting time by stalling when it should be moving.
    # Conditions:
    #   - Robot is static (velocity < threshold)
    #   - d_goal > σsoft (2 m) - far from goal
    #   - NOT correct direction (angle >= 105°) - not heading toward goal
    stall = RewTerm(
        func=mdp.stall,
        weight=-20.0,
        params={
            "command_name": "pose_command",
            "sigma_soft": 2.0,  # σsoft = 2 m (distance threshold)
            "velocity_threshold": 0.1,  # Velocity threshold to consider static (m/s)
            "correct_direction_threshold": 105.0,  # 105° angle threshold
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # -- Regularization reward (paper formula) --
    # r(reg) = -2·v_z² - 0.05·(ω_x²+ω_y²) - 20·(g_x²+g_y²)
    #          - 0.0005·‖τ‖² - 20·Σ ReLU(|τ_i|-0.85·τ_lim) - 0.0005·‖q̇‖² - 20·Σ ReLU(|q̇_i|-0.9·q̇_lim)
    #          - 20·Σ ReLU(|q_i|-0.95·q_lim) - 2e-7·‖q̈‖² - 4e-6·‖ȧ‖² - 20·1(fly)
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
    """Termination terms for the environment."""
    # 对应 terminate_after_contacts_on = ["base"]
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # Table II: Episode length U(7, 9) s — we set fixed episode_length_s in env cfg and
    # use the built-in time_out termination flag.
    time_out = TermTerm(func=mdp.time_out, time_out=True)

##
# Environment Config
##

@configclass
class Go1AgileEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go1 agile (position-tracking) environment."""
    # Scene settings
    scene: Go1AgileSceneCfg = Go1AgileSceneCfg(num_envs=1280, env_spacing=2.5)
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
        # General timing settings: 200 Hz physics (dt=0.005), 4 control steps per action, 8 s episodes.
        self.decimation = 4
        self.episode_length_s = 8.0
        # Keep sim config consistent with decimation.
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
