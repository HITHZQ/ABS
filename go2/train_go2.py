# SPDX-License-Identifier: BSD-3-Clause
"""
Go2 强化学习训练入口（RSL-RL）。

必须先注册 Gym 环境，因此本脚本在调用 Isaac Lab 的 train.py 之前执行 ``import go2``。

前置条件
--------
1. 已安装 Isaac Lab / Isaac Sim，并能使用 ``isaaclab`` / ``isaaclab.bat`` 启动。
2. 已安装本包（在 ``go2`` 目录下）::

       pip install -e .

3. 设置环境变量 ``ISAACLAB_ROOT`` 为 Isaac Lab 仓库根目录（含 ``scripts`` 文件夹）。

用法（Windows PowerShell 示例）
------------------------------
::

    $env:ISAACLAB_ROOT = "C:\\path\\to\\IsaacLab"
    isaaclab.bat -p C:\\path\\to\\ABS-main\\go2\\train_go2.py --headless

常用参数会传给官方 ``train.py``，例如 ``--num_envs``、``--seed``、``--max_iterations``。
默认任务名为 ``Unitree-Go2-Velocity``；若命令行已含 ``--task``，则不再追加。

仅可视化 / Play（少并行环境）可使用任务 ``Unitree-Go2-Velocity-Play``::

    isaaclab.bat -p ...\\train_go2.py --task Unitree-Go2-Velocity-Play
"""

from __future__ import annotations

import os
import sys


def _find_train_script(isaaclab_root: str) -> str | None:
    root = isaaclab_root.rstrip("/\\")
    candidates = [
        os.path.join(root, "scripts", "reinforcement_learning", "rsl_rl", "train.py"),
        os.path.join(root, "source", "standalone", "workflows", "rsl_rl", "train.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main() -> None:
    # 注册 Unitree-Go2-Velocity / Unitree-Go2-Velocity-Play
    import go2  # noqa: F401

    isaaclab_root = os.environ.get("ISAACLAB_ROOT") or os.environ.get("ISAACLAB_PATH") or ""
    train_py = _find_train_script(isaaclab_root) if isaaclab_root else None

    user_args = list(sys.argv[1:])
    if "--task" not in user_args:
        user_args = ["--task", "Unitree-Go2-Velocity"] + user_args

    if not train_py:
        print("[go2/train_go2.py] 未找到 Isaac Lab 的 train.py。")
        print("  请设置环境变量 ISAACLAB_ROOT 为 Isaac Lab 仓库根目录。")
        print("  已尝试路径（在 ISAACLAB_ROOT 下）：")
        print("    scripts/reinforcement_learning/rsl_rl/train.py")
        print("    source/standalone/workflows/rsl_rl/train.py")
        print()
        print("也可手动执行（需保证已 pip install -e go2，且 Python 能 import go2）：")
        print('  isaaclab.bat -p "%ISAACLAB_ROOT%\\scripts\\reinforcement_learning\\rsl_rl\\train.py" '
              '--task Unitree-Go2-Velocity --headless')
        print()
        print("在 train 启动前于同一进程中执行一次: import go2")
        raise SystemExit(1)

    train_dir = os.path.dirname(train_py)
    if train_dir and train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    sys.argv = [train_py] + user_args
    with open(train_py, encoding="utf-8") as f:
        code = compile(f.read(), train_py, "exec")
    g = {"__name__": "__main__", "__file__": train_py}
    exec(code, g)


if __name__ == "__main__":
    main()
