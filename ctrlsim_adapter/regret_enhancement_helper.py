"""Pure helpers for PLR/ACCEL regret enhancement."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def compute_weighted_positive_rtg_gap(
    teacher_k_step_rtg: Sequence[float],
    student_applied_return: Sequence[float],
    regret_component_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Return weighted positive teacher-student RTG gap."""
    teacher = np.asarray(teacher_k_step_rtg, dtype=np.float32)
    student = np.asarray(student_applied_return, dtype=np.float32)
    regret_component_weights_arr = np.asarray(
        regret_component_weights,
        dtype=np.float32,
    )
    return float(np.maximum(teacher - student, 0.0).dot(regret_component_weights_arr))


def compute_truncated_episode_rtg_gap(
    records: Sequence[Mapping[str, object]],
    *,
    method: str = "first_inference_step_gap",
    regret_component_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> float | None:
    """Compute the inference-to-last RTG gap in one (truncated) episode."""
    if len(records) < 2:
        return None

    last_record = records[-1]
    last_teacher = np.asarray(last_record["teacher_full_rtg_3d"], dtype=np.float32)
    last_student = np.asarray(
        last_record["student_applied_return_3d_before_step"],
        dtype=np.float32,
    )

    gaps = []
    for record in records[:-1]:
        teacher = np.asarray(record["teacher_full_rtg_3d"], dtype=np.float32)
        student = np.asarray(
            record["student_applied_return_3d_before_step"],
            dtype=np.float32,
        )
        gaps.append(
            compute_weighted_positive_rtg_gap(
                teacher_k_step_rtg=teacher - last_teacher,
                student_applied_return=last_student - student,
                regret_component_weights=regret_component_weights,
            )
        )

    if method == "first_inference_step_gap":
        return float(gaps[0])
    if method == "mean_gap":
        return float(np.mean(gaps))
    if method == "max_gap":
        return float(np.max(gaps))
    raise ValueError(f"Unsupported rtg gap aggregation method: {method}")


def resolve_rollout_seed(
    current_level_seeds: Sequence[int] | None,
    process_idx: int,
    info: Mapping[str, object],
) -> int | None:
    """Resolve the rollout seed from PLR state or step info."""
    if "seed" in info:
        return int(info["seed"])
    if current_level_seeds is not None:
        return int(current_level_seeds[process_idx])
    return None


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
