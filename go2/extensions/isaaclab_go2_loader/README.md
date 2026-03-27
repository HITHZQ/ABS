# isaaclab_go2_loader

使官方 `train.py` / `play.py` 能识别 `Unitree-Go2-Velocity` 任务：扩展加载时会执行 `import go2` 完成 Gym 注册。

## 1. 安装 `go2`

```bash
pip install -e /path/to/ABS-main/go2
```

## 2. 把本目录链到 Isaac Lab 的 `source/extensions`

```bash
ln -s /path/to/ABS-main/go2/extensions/isaaclab_go2_loader \
  /path/to/IsaacLab/source/extensions/isaaclab_go2_loader
```

（Windows 可用目录联接 `mklink /J`，或复制整个 `isaaclab_go2_loader` 文件夹。）

## 3. 使用官方脚本

在 Isaac Lab 仓库根目录：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Unitree-Go2-Velocity --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Unitree-Go2-Velocity
```

若扩展未自动启用，在 Isaac Sim 扩展管理器中启用 **ABS Go2 Gym registration**（或同名扩展）。

## 若仍报 `NameNotFound: Environment Unitree-Go2-Velocity doesn't exist`

headless 训练时扩展有时**不会**在 `gym.spec()` 之前加载你的包。最稳妥做法：

1. 使用仓库根目录上一级的 `train_go2.py`（先 `import go2` 再执行官方 `train.py`），或  
2. 在官方 `train.py` / `play.py` 里、`import isaaclab_tasks` **下一行**增加：  
   `import go2  # noqa: F401`

详见：`go2/OFFICIAL_TRAIN_PLAY_REGISTER.txt`。
