# SPDX-License-Identifier: BSD-3-Clause
"""
Go2 官方 play.py 包装：先 ``import go2`` 注册 Gym，再执行 Isaac Lab 的 play.py。

用法::

    export ISAACLAB_ROOT=/home/user/IsaacLab
    cd /home/user/IsaacLab
    ./isaaclab.sh -p source/isaaclab/ABS-main/go2/play_go2.py --task Unitree-Go2-Velocity

默认任务 ``Unitree-Go2-Velocity``；可视化少 env 可用 ``--task Unitree-Go2-Velocity-Play``。
"""

from __future__ import annotations

import os
import sys


def _find_play_script(isaaclab_root: str) -> str | None:
    root = isaaclab_root.rstrip("/\\")
    candidates = [
        os.path.join(root, "scripts", "reinforcement_learning", "rsl_rl", "play.py"),
        os.path.join(root, "source", "standalone", "workflows", "rsl_rl", "play.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main() -> None:
    import go2  # noqa: F401

    isaaclab_root = os.environ.get("ISAACLAB_ROOT") or os.environ.get("ISAACLAB_PATH") or ""
    play_py = _find_play_script(isaaclab_root) if isaaclab_root else None

    user_args = list(sys.argv[1:])
    if "--task" not in user_args:
        user_args = ["--task", "Unitree-Go2-Velocity"] + user_args

    if not play_py:
        print("[go2/play_go2.py] 未找到 Isaac Lab 的 play.py。")
        print("  请设置 ISAACLAB_ROOT，并确认存在：")
        print("    scripts/reinforcement_learning/rsl_rl/play.py")
        raise SystemExit(1)

    play_dir = os.path.dirname(play_py)
    if play_dir and play_dir not in sys.path:
        sys.path.insert(0, play_dir)
    sys.argv = [play_py] + user_args
    with open(play_py, encoding="utf-8") as f:
        code = compile(f.read(), play_py, "exec")
    g = {"__name__": "__main__", "__file__": play_py}
    exec(code, g)


if __name__ == "__main__":
    main()
