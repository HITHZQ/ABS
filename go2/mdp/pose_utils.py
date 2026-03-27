from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_pose_command_world(env: "ManagerBasedRLEnv", command_name: str):
    """Return pose-command target (xy, heading) in world frame.

    For pose commands, `get_command()` is typically in base frame. The command term
    stores world-frame targets in `pos_command_w` and `heading_command_w`.
    """
    try:
        term = env.command_manager.get_term(command_name)
        if hasattr(term, "pos_command_w") and hasattr(term, "heading_command_w"):
            return term.pos_command_w[:, :2].clone(), term.heading_command_w.clone()
    except (KeyError, AttributeError):
        pass
    return None, None

