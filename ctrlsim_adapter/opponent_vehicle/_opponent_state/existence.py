"""
负责判断 Nocturne 仿真中的车辆位置是否表示“真实存在”。
该模块统一处理非法数值、哨兵坐标和存在性状态转移中的几个小规则。
Determines whether a vehicle position in Nocturne represents a valid simulated existence.
Normalizes sentinel coordinates, invalid numbers, and the small state-transition rules used by opponent existence logic.
"""

import math
from typing import Any, Dict, Optional

import numpy as np


def sim_position_exists(x: float, y: float) -> bool:
    try:
        xf = float(x)
        yf = float(y)
    except Exception:
        return False
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return False
    return not (xf == -1000000.0 and yf == -1000000.0)


def _compute_goal_hold_until(
    prev_hold_until: Optional[int],
    current_step: int,
    reached_goal: bool,
    hold_steps: int,
) -> Optional[int]:
    if prev_hold_until is not None or not reached_goal:
        return prev_hold_until
    return current_step + hold_steps


def _should_drop_after_goal(current_step: int, hold_until: Optional[int]) -> bool:
    return hold_until is not None and current_step >= hold_until


def _keep_exists_on_invalid(sim_exists: bool, prev_exists: bool) -> bool:
    return bool(sim_exists or prev_exists)


def resolve_vehicle_exists(
    adapter: Any,
    *,
    veh_id: int,
    t: int,
    is_controlled: bool,
    ego_id: Optional[int],
    gt_traj_data: np.ndarray,
    pos: Any,
    constant_state: Optional[Dict[str, Any]],
) -> int:
    if constant_state is not None:
        return int(constant_state["existence"])

    protected = (veh_id == ego_id) or is_controlled
    if not protected:
        return int(gt_traj_data[t, 4])
    if not is_controlled:
        return 1 if sim_position_exists(pos.x, pos.y) else 0

    sim_exists = sim_position_exists(pos.x, pos.y)
    prev_exists = adapter._opponent_vehicle_exits.get(veh_id, bool(sim_exists))
    hold_until = adapter._opponent_goal_hold_until.get(veh_id)
    exists = _keep_exists_on_invalid(sim_exists, prev_exists)
    if _should_drop_after_goal(t, hold_until):
        exists = False
    adapter._opponent_vehicle_exits[veh_id] = bool(exists)
    return 1 if exists else 0
