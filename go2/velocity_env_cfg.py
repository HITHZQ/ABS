"""
Go2 融合任务：以「随机平面目标快速接近」为主（pose_command + distance/agile），
辅以定点后按 base_velocity 跟踪与课程。

- 主目标：UniformPose2dCommand 每回合一个随机 2D 点（重采样周期 ≥ episode），distance_to_goal + agile 推快贴目标。
- go1：圆柱避障、11 维 lidar（illusion）、possoft / postight / heading 等。
- go2：base_velocity；近目标门控 track_*（权重略低以免干扰冲刺）；feet_air_time；lin_vel_cmd_levels。
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG

from . import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
         ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


# 圆柱体障碍物通用 spawn 配置（与 go1 agile 一致：半径 0.4 m，kinematic）
_CYLINDER_SPAWN = sim_utils.CylinderCfg(
    radius=0.4,
    height=1.0,
    mass_props=sim_utils.MassPropertiesCfg(50.0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Go2 目标到达 + 避障：地形、机器人、射线雷达、圆柱体障碍、接触传感器、灯光。"""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 前向 11 射线 lidar（与 go1 agile 一致），用于避障与接近目标；仅检测地面（圆柱由碰撞惩罚学习避让）
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.05, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-math.pi / 4, math.pi / 4),
            horizontal_res=math.pi / 20,
        ),
        max_distance=6.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # 圆柱体障碍物（0~8 个，reset 时按 curriculum 放置，避免与机器人/目标重叠）
    cylinder_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_0",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.25)),
    )
    cylinder_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_1",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_2",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_3",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_4",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_5",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_6",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cylinder_7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder_7",
        spawn=_CYLINDER_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """目标到达 + 避障：圆柱体 kinematic、curriculum 重置、机器人从原点 reset、域随机化与 push。"""

    # 圆柱体：启动时设为 kinematic，reset 时按 terrain level 放置 0~max 个
    set_cylinders_kinematic = EventTerm(
        func=mdp.set_cylinders_kinematic_at_startup,
        mode="startup",
        params={
            "obstacle_names": [
                "cylinder_0", "cylinder_1", "cylinder_2", "cylinder_3",
                "cylinder_4", "cylinder_5", "cylinder_6", "cylinder_7",
            ],
        },
    )

    reset_cylinder_obstacles = EventTerm(
        func=mdp.reset_cylinder_obstacles_curriculum,
        mode="reset",
        params={
            "obstacle_names": [
                "cylinder_0", "cylinder_1", "cylinder_2", "cylinder_3",
                "cylinder_4", "cylinder_5", "cylinder_6", "cylinder_7",
            ],
            "x_range": (-2.5, 8.5),
            "y_range": (-2.5, 2.5),
            "num_levels": 10,
            "max_obstacles": 8,
            "min_obstacles": 0,
            "grid_nx": 10,
            "grid_ny": 5,
        },
    )

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset：机器人从原点附近出发，朝向随机；z 增量须为 0（default_root_state 已含 Go2 站立高度 0.4m，再加会悬空掉落）
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """pose：每回合一个随机平面目标（重采样 ≥ episode，冲刺中途不换点）；base_velocity：到点附近再采样的速度指令。"""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        # 与 RobotEnvCfg.episode_length_s 对齐，整局同一随机目标，便于学「快速接近当前点」
        resampling_time_range=(10.0, 10.0),
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(1.5, 6.0),
            pos_y=(-2.5, 2.5),
            heading=(-0.3, 0.3),
        ),
    )

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """观测：pose_command + base_velocity + lidar + 本体/关节。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        pose_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "pose_command"}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # 11 维射线（log 尺度 + illusion），用于避障与接近目标
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
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-5.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        pose_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "pose_command"}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """主：快速接近随机平面目标；辅：步态与少量正则（强调快 + 正向行走 + 少碰撞）。"""

    # -- 与 Isaac-Velocity-Rough 基类一致：大腿非期望接触（Go2 链名 * _thigh；官方 Go2 rough 常关此项，此处按「原 velocity 奖励」启用）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    # -- 抑制“膝盖/小腿顶着走”：对小腿（calf）接触也加惩罚（比大腿略弱，避免过度保守）
    undesired_contacts_calf = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )

    # -- 任务奖励：接近目标、最后时段位姿/朝向、朝目标快速运动
    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal_reward,
        weight=115.0,
        params={
            "command_name": "pose_command",
            "sigma": 6.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    
    # vel_toward_goal_scale：压低 agile 的 term1b（倒走仍可 toward-goal，易 max(term1a,term1b) 压过正走）
    agile = RewTerm(
        func=mdp.agile,
        weight=45.0,
        params={
            "command_name": "pose_command",
            "vmax": 6.0,
            "sigma_tight": 0.8,
            "correct_direction_threshold": 120.0,
            "vel_toward_goal_scale": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    upright = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=14.0,
        params={"threshold": 0.92, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 抑制“倒着跑也能快速接近”的捷径：只在确实倒退且仍在接近目标时惩罚
    backward_toward_goal = RewTerm(
        func=mdp.backward_toward_goal_penalty,
        weight=-4.0,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot"),
            "vx_deadband": 0.05,
            "dist_min_m": 0.3,
        },
    )
    # 先转向再前进，减少通过倒退“抄近路”接近目标
    align_heading_before_motion = RewTerm(
        func=mdp.align_heading_before_motion_penalty,
        weight=-2.0,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot"),
            "dist_heading_blend_m": 0.9,
            "min_bearing_dist_m": 0.12,
            "planar_speed_deadband": 0.12,
        },
    )

    # -- 正则与步态
    # 机体稳定性：抑制侧翻/俯仰摆动与离地弹跳（小权重，避免把速度压死）
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.06)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.8)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.28,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "pose_command",
            "threshold": 0.45,
        },
    )
    bound_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.7,
        params={
            "period": 0.55,
            # 前两条腿同相、后两条腿同相（假设顺序为：前左、前右、后左、后右）
            "offset": [0.0, 0.0, 0.5, 0.5],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "command_name": "pose_command",
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success（用于 success rate 可视化）：以“快速接近目标”为主，不强制末端航向对齐；短保持避免抖动误判
    success = DoneTerm(
        func=mdp.goal_reached_success,
        params={
            "command_name": "pose_command",
            "pos_threshold_m": 0.8,
            "yaw_threshold_rad": 3.14,
            "min_hold_s": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """地形难度（圆柱体等）+ 速度指令上限课程（go2 快速运动）。"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Go2：随机平面目标快速接近为主；近目标速度跟踪与课程为辅。"""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        # 略长于原 8s：远目标(至 ~5 m)+ 冲刺需要可学习窗口；与 pose_command 重采样 10s 对齐
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Avoid GPU narrowphase patch-buffer overflow under many envs + obstacle contacts.
        self.sim.physx.gpu_max_rigid_patch_count = 24 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
