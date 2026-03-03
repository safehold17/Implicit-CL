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
