# Go1 / Go2 奖励函数对比与快速移动设计

## 一、任务类型对比

| 维度 | Go1 (Agile) | Go2 (Velocity) |
|------|-------------|-----------------|
| **指令类型** | 位姿指令 (pose_command)：目标点 + 朝向 | 速度指令 (base_velocity)：线速度 xy + 角速度 z |
| **目标行为** | 尽快到达目标点并保持站立/朝向 | 跟踪给定线速度/角速度 |
| **主任务奖励** | distance_to_goal, possoft, postight, heading, agile | track_lin_vel_xy, track_ang_vel_z |

## 二、Go1 Agile 奖励结构（节选）

- **任务类**：distance_to_goal (90), possoft/postight (60), heading (30), agile (30), upright (8), leg_symmetry (4)
- **惩罚类**：collision (-25), stall (-25), stand (-3), regularization (4)
- **特点**：强目标导向、鼓励“朝目标快速移动”(agile)、静止重罚(stall)、接触约束(collision)

## 三、Go2 Velocity 默认奖励结构（原）

| 项 | 权重 | 作用 |
|----|------|------|
| track_lin_vel_xy | 1.5 | 跟踪 xy 线速度 |
| track_ang_vel_z | 0.75 | 跟踪 yaw 角速度 |
| lin_vel_z_l2 | -2.0 | 抑制竖直速度（防跳） |
| ang_vel_xy_l2 | -0.05 | 抑制 roll/pitch 角速度 |
| joint_vel_l2 | -0.001 | 关节速度正则 |
| joint_acc_l2 | -2.5e-7 | 关节加速度正则 |
| joint_torques_l2 | -2e-4 | 关节力矩正则 |
| action_rate_l2 | -0.1 | 动作变化率 |
| dof_pos_limits | -10.0 | 关节限位 |
| energy | -2e-5 | 能耗 |
| flat_orientation_l2 | -2.5 | 保持躯干水平 |
| joint_pos (stand_still) | -0.7 | 站立时关节偏离惩罚 |
| feet_air_time | 0.1 | 抬脚时长（促步态） |
| air_time_variance | -1.0 | 四足抬脚时间方差 |
| feet_slide | -0.1 | 脚打滑 |
| undesired_contacts | -1 | 非足端接触 |

## 四、快速移动设计原则

1. **提高速度跟踪收益**：加大 track_lin_vel_xy / track_ang_vel_z，让“跟上/超过指令速度”更值得。
2. **放宽动力约束**：适度减小 joint_torques、joint_acc、energy 的惩罚，允许更大发力与更快运动。
3. **强化步态**：提高 feet_air_time，鼓励明显腾空步态（跑而非拖步）。
4. **保持稳定性**：保留 flat_orientation、lin_vel_z、undesired_contacts 等，避免翻车与乱跳。
5. **避免过度正则**：joint_vel、action_rate 不宜过大，否则会压制快速、动态动作。

## 五、快速移动推荐权重（已用于 Go2 Velocity）

| 项 | 原权重 | 快速移动权重 | 说明 |
|----|--------|--------------|------|
| track_lin_vel_xy | 1.5 | **2.2** | 主任务，强化线速度跟踪 |
| track_ang_vel_z | 0.75 | **1.0** | 转向跟踪 |
| lin_vel_z_l2 | -2.0 | -2.0 | 保持，防跳 |
| ang_vel_xy_l2 | -0.05 | -0.05 | 保持 |
| joint_vel_l2 | -0.001 | **-0.0005** | 略放宽，允许更快摆腿 |
| joint_acc_l2 | -2.5e-7 | **-1.2e-7** | 放宽，更动态 |
| joint_torques_l2 | -2e-4 | **-1e-4** | 放宽，允许更大蹬地 |
| action_rate_l2 | -0.1 | **-0.06** | 略放宽，响应更快 |
| dof_pos_limits | -10.0 | -10.0 | 保持 |
| energy | -2e-5 | **-1e-5** | 略放宽 |
| flat_orientation_l2 | -2.5 | -2.5 | 保持，防倾覆 |
| joint_pos (stand_still) | -0.7 | -0.7 | 保持 |
| feet_air_time | 0.1 | **0.22** | 鼓励跑步步态 |
| air_time_variance | -1.0 | **-0.5** | 略放宽，避免过度对称约束 |
| feet_slide | -0.1 | -0.1 | 保持 |
| undesired_contacts | -1 | -1 | 保持 |

速度跟踪使用指数形式 `exp(-error^2/(2*std^2))`，std 保持 `sqrt(0.25)=0.5` 即可；若希望高速时对误差更宽容，可把 std 提到 0.6。

## 六、指令范围建议

要让机器人“快速移动”，除奖励外还需配合**速度指令范围**（CommandsCfg 中 base_velocity 的 ranges/limit_ranges）。适当提高 `lin_vel_x`、`lin_vel_y` 的上限（例如 1.2~1.5 m/s），并配合课程或固定大速度指令，才能与上述奖励一起学到快速跑。
