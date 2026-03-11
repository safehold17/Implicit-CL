"""
负责对手车辆存在性状态转移中的几个小判定规则。
该模块封装 goal hold、到点后消失以及非法状态回退等逻辑，供状态更新流程复用。
Encapsulates the small decision rules used in opponent-vehicle existence state transitions.
Covers goal-hold timing, post-goal disappearance, and fallback behavior on invalid states for reuse in updates.
"""

from typing import Optional


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
