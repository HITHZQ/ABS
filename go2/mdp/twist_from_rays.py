from __future__ import annotations

import torch


def twist_command_from_rays(
    rays_m: torch.Tensor,
    *,
    v_back_max: float = 1.2,
    wz_max: float = 2.5,
    d_stop: float = 0.35,
    d_enter: float = 0.9,
    turn_gain: float = 2.0,
) -> torch.Tensor:
    """Heuristic twist command from (forward) ray distances.

    Args:
        rays_m: (E, R) ray distances in meters. Assumed ordered left->right across forward FOV.
        v_back_max: maximum backward speed when very close.
        wz_max: maximum yaw rate for turning away.
        d_stop: distance where we saturate to max avoidance.
        d_enter: distance where avoidance starts ramping in.
        turn_gain: gain for left-right asymmetry -> yaw rate.

    Returns:
        twist_cmd: (E, 3) = (vx, vy, wz) in base frame. vy is 0 (keep it simple & stable).
    """
    if rays_m.dim() != 2:
        raise ValueError(f"rays_m must be 2D (E,R), got shape={tuple(rays_m.shape)}")
    E, R = rays_m.shape
    if R < 3:
        raise ValueError("Need at least 3 rays to decide left/right turn")

    rays = torch.nan_to_num(rays_m, nan=float(d_enter), posinf=float(d_enter), neginf=0.0)
    rays = torch.clamp(rays, min=0.0)
    dmin = torch.amin(rays, dim=1)  # (E,)

    # ramp from 0 at d_enter to 1 at d_stop (clamped)
    denom = max(float(d_enter - d_stop), 1e-6)
    s = torch.clamp((float(d_enter) - dmin) / denom, 0.0, 1.0)  # (E,)
    vx = -float(v_back_max) * s

    # turn away from the closer side
    left = rays[:, : R // 2]
    right = rays[:, R // 2 :]
    left_min = torch.amin(left, dim=1)
    right_min = torch.amin(right, dim=1)
    # if left side is closer -> turn right (negative?), define wz positive = turn left (CCW)
    # choose wz = + when right side is closer (turn left away from right obstacle)
    asym = (left_min - right_min)  # >0 means right closer
    wz = float(wz_max) * torch.tanh(float(turn_gain) * asym) * s

    vy = torch.zeros_like(vx)
    return torch.stack([vx, vy, wz], dim=1)

