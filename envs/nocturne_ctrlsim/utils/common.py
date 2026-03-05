"""Common utility functions for Nocturne CtrlSim modules."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


_INVALID_POSITION_MARKERS = {
    (-10000.0, -10000.0),
    (-1000000.0, -1000000.0),
}


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def merge_episode_progress(
    previous_progress: float,
    current_progress: float,
    position_reached: bool,
) -> float:
    """Merge current step progress into an episode-level progress scalar."""
    if position_reached:
        return 1.0
    return max(clamp01(previous_progress), clamp01(current_progress))


def radians_to_degrees(radians: float) -> float:
    return float(radians) * 180.0 / math.pi


def angle_of_rotation(yaw: float) -> float:
    return (math.pi / 2.0) - float(yaw)


def angle_sub(current_angle: float, target_angle: float) -> float:
    diff = (target_angle - current_angle) % (2 * math.pi)
    if diff > math.pi:
        diff = -(2 * math.pi - diff)
    return diff


def to_local(dx: float, dy: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    return local_x, local_y


def is_valid_world_position(x: float, y: float, max_abs: float = 1e5) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    if abs(x) > max_abs or abs(y) > max_abs:
        return False
    if (float(x), float(y)) in _INVALID_POSITION_MARKERS:
        return False
    return True


def compute_square_view_bounds(
    positions: Sequence[Sequence[float]],
    padding: float = 25.0,
    default_half_extent: float = 50.0,
) -> tuple[float, float, float, float]:
    """Compute square axis bounds from positions."""
    if not positions:
        return (
            -default_half_extent,
            default_half_extent,
            -default_half_extent,
            default_half_extent,
        )

    points = np.asarray(positions, dtype=np.float32)
    x_min = float(np.min(points[:, 0]) - padding)
    x_max = float(np.max(points[:, 0]) + padding)
    y_min = float(np.min(points[:, 1]) - padding)
    y_max = float(np.max(points[:, 1]) + padding)

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        diff = (x_range - y_range) / 2.0
        y_min -= diff
        y_max += diff
    else:
        diff = (y_range - x_range) / 2.0
        x_min -= diff
        x_max += diff

    return x_min, x_max, y_min, y_max
