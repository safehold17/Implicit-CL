import math
from typing import Tuple


def _angle_sub(current_angle: float, target_angle: float) -> float:
    """Return the shortest signed angle from current_angle to target_angle."""
    diff = (target_angle - current_angle) % (2.0 * math.pi)
    if diff > math.pi:
        diff = -(2.0 * math.pi - diff)
    return diff


def safe_backward_action_from_states(
    prev_pos: Tuple[float, float],
    prev_theta: float,
    prev_vel: float,
    curr_pos: Tuple[float, float],
    curr_theta: float,
    curr_vel: float,
    wheel_base: float,
    dt: float,
    steer_limit: float = 0.7,
    curvature_eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Backward-compute (accel, steer) from two adjacent states with numeric guards.
    This mirrors `BicycleModel.backward` but clips curvature to avoid invalid sqrt.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    prev_x, prev_y = prev_pos
    curr_x, curr_y = curr_pos
    _ = (prev_x, prev_y, curr_x, curr_y)  # keep signature explicit for callers

    vel = prev_vel
    accel = (curr_vel - vel) / dt

    theta = prev_theta
    w = _angle_sub(theta, curr_theta) / dt
    C = 2.0 * wheel_base * w / (curr_vel + vel + 1e-10)

    max_abs_c = max(0.0, 2.0 - curvature_eps)
    C = max(-max_abs_c, min(max_abs_c, C))

    den = math.sqrt(max(4.0 - C * C, 1e-12))
    steer = math.atan(2.0 * C / den)

    if not math.isfinite(steer):
        steer = 0.0
    steer = max(-steer_limit, min(steer_limit, steer))

    return float(accel), float(steer)
