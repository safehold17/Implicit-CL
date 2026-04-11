"""Pure helpers for PLR/ACCEL regret enhancement."""

from __future__ import annotations


def should_collect_ego_ctrlsim_rtg_step(
    *,
    t: int,
    history_steps: int,
    sparse_inference_action_repeat: bool,
    action_repeat_frequency: int,
) -> bool:
    """Return whether this env step should export ego teacher RTG."""

    anchor_t = int(history_steps) - 1
    if int(t) < anchor_t:
        return False
    if not bool(sparse_inference_action_repeat):
        return True

    phase = (int(t) - anchor_t) % int(action_repeat_frequency)
    return phase != int(action_repeat_frequency) - 1
