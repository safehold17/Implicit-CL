"""Rollout-local helpers for PLR/ACCEL regret enhancement."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch

from ctrlsim_adapter.regret_enhancement_helper import (
    combine_enhanced_regret_score,
    compute_learnability,
    compute_mean_delta_rtg,
    compute_truncated_episode_rtg_gap,
    normalize_by_running_mean,
    normalize_learnability,
)


def build_nocturne_rtg_record(info: Mapping[str, object]) -> dict[str, object] | None:
    """Build one RTG segment record from a step info payload."""
    teacher_rtg = info.get("ego_ctrlsim_pred_rtg")
    step_t = info.get("ego_ctrlsim_pred_rtg_step")
    student_return = info.get("student_component_applied_return")
    if teacher_rtg is None or step_t is None or student_return is None:
        return None
    return {
        "step_t": int(step_t),
        "teacher_full_rtg_3d": np.asarray(teacher_rtg, dtype=np.float32),
        "student_applied_return_3d_before_step": np.asarray(
            student_return,
            dtype=np.float32,
        ),
    }


def append_nocturne_rtg_segment(
    *,
    seed: int | None,
    records: Sequence[Mapping[str, object]],
    delta_rtg_segments_by_seed: defaultdict[int, list[float]],
    rtg_gap_method: str,
    regret_component_weights: Sequence[float],
) -> None:
    """Append one valid RTG segment gap to the seed aggregation."""
    if seed is None:
        return
    delta_rtg_segment = compute_truncated_episode_rtg_gap(
        records,
        method=rtg_gap_method,
        regret_component_weights=regret_component_weights,
    )
    if delta_rtg_segment is not None:
        delta_rtg_segments_by_seed[int(seed)].append(delta_rtg_segment)


def compute_nocturne_base_regret_by_seed(
    rollouts,
) -> tuple[dict[int, float], dict[int, int], int]:
    """Compute positive value-loss regret by rollout seed."""
    level_seeds = rollouts.level_seeds.cpu()
    total_steps, num_actors = rollouts.action_log_dist.shape[:2]
    done = (~(rollouts.masks > 0)).cpu()
    cliffhanger = (~(rollouts.cliffhanger_masks > 0)).cpu()
    returns = rollouts.returns.cpu()
    if rollouts.use_popart:
        value_preds = rollouts.denorm_value_preds.cpu()
    else:
        value_preds = rollouts.value_preds.cpu()

    regret_sum_by_seed = defaultdict(float)
    regret_count_by_seed = defaultdict(int)
    actor_seed_by_index = {}

    for actor_index in range(num_actors):
        start_t = 0
        done_steps = done[:, actor_index].nonzero()[:, 0]

        for done_t in done_steps:
            t = int(done_t.item())
            if not start_t < total_steps:
                break
            if t == 0:
                continue

            seed_t = int(level_seeds[start_t, actor_index].item())
            actor_seed_by_index[actor_index] = seed_t
            if not cliffhanger[t, actor_index]:
                positive_advantages = (
                    returns[start_t:t, actor_index]
                    - value_preds[start_t:t, actor_index]
                ).clamp(0)
                regret_sum_by_seed[seed_t] += float(positive_advantages.sum().item())
                regret_count_by_seed[seed_t] += int(positive_advantages.numel())

            start_t = t

        if start_t < total_steps:
            seed_t = int(level_seeds[start_t, actor_index].item())
            actor_seed_by_index[actor_index] = seed_t
            positive_advantages = (
                returns[start_t:, actor_index]
                - value_preds[start_t:, actor_index]
            ).clamp(0)
            regret_sum_by_seed[seed_t] += float(positive_advantages.sum().item())
            regret_count_by_seed[seed_t] += int(positive_advantages.numel())

    base_regret_by_seed = {
        seed: regret_sum_by_seed[seed] / float(regret_count)
        for seed, regret_count in regret_count_by_seed.items()
        if regret_count > 0
    }
    return base_regret_by_seed, actor_seed_by_index, num_actors


def compute_nocturne_enhanced_regret_scores(
    rollouts,
    rollout_info: Mapping[str, object],
    *,
    running_mean_base_regret: float | None = None,
    running_mean_delta_rtg: float | None = None,
    use_solvable_rate: bool = True,
    use_ctrlsim_rtg_gap: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build per-actor external sampler scores for enhanced regret."""
    base_regret_by_seed, actor_seed_by_index, num_actors = (
        compute_nocturne_base_regret_by_seed(rollouts)
    )
    attempt_count_by_seed = rollout_info.get("attempt_count_by_seed", {})
    success_count_by_seed = rollout_info.get("success_count_by_seed", {})
    delta_rtg_segments_by_seed = rollout_info.get("delta_rtg_segments_by_seed", {})

    all_seeds = (
        set(base_regret_by_seed.keys())
        | set(attempt_count_by_seed.keys())
        | set(delta_rtg_segments_by_seed.keys())
    )

    solvable_rate_by_seed = {}
    learnability_by_seed = {}
    delta_rtg_by_seed = {}
    base_regret_norm_by_seed = {}
    learnability_norm_by_seed = {}
    delta_rtg_norm_by_seed = {}
    enhanced_regret_score_by_seed = {}

    for seed in all_seeds:
        seed_int = int(seed)
        base_regret = float(base_regret_by_seed.get(seed, 0.0))
        attempt_count = int(attempt_count_by_seed.get(seed, 0))
        success_count = int(success_count_by_seed.get(seed, 0))
        solvable_rate = None
        if attempt_count > 0:
            solvable_rate = success_count / float(attempt_count)
        learnability = compute_learnability(
            success_count=success_count,
            attempt_count=attempt_count,
        )
        if solvable_rate is not None:
            solvable_rate_by_seed[seed_int] = solvable_rate
        if learnability is not None:
            learnability_by_seed[seed_int] = learnability

        delta_rtg = compute_mean_delta_rtg(delta_rtg_segments_by_seed.get(seed, ()))
        if delta_rtg is not None:
            delta_rtg_by_seed[seed_int] = delta_rtg

        base_regret_norm = normalize_by_running_mean(
            base_regret,
            running_mean_base_regret,
        )
        if base_regret_norm is not None:
            base_regret_norm_by_seed[seed_int] = base_regret_norm

        learnability_norm = normalize_learnability(learnability)
        if learnability_norm is not None:
            learnability_norm_by_seed[seed_int] = learnability_norm

        delta_rtg_norm = normalize_by_running_mean(
            delta_rtg,
            running_mean_delta_rtg,
        )
        if delta_rtg_norm is not None:
            delta_rtg_norm_by_seed[seed_int] = delta_rtg_norm

        enhanced_regret_score_by_seed[seed_int] = combine_enhanced_regret_score(
            base_regret=base_regret_norm if base_regret_norm is not None else 0.0,
            learnability=learnability_norm,
            delta_rtg=delta_rtg_norm,
            use_solvable_rate=use_solvable_rate,
            use_ctrlsim_rtg_gap=use_ctrlsim_rtg_gap,
        )

    external_scores = torch.zeros((num_actors, 1), dtype=torch.float32)
    for actor_index, seed in actor_seed_by_index.items():
        external_scores[actor_index, 0] = float(
            enhanced_regret_score_by_seed.get(int(seed), 0.0)
        )

    metrics = {
        "base_regret_by_seed": {
            int(seed): float(score)
            for seed, score in base_regret_by_seed.items()
        },
        "solvable_rate_by_seed": solvable_rate_by_seed,
        "learnability_by_seed": learnability_by_seed,
        "delta_rtg_by_seed": delta_rtg_by_seed,
        "base_regret_norm_by_seed": base_regret_norm_by_seed,
        "learnability_norm_by_seed": learnability_norm_by_seed,
        "delta_rtg_norm_by_seed": delta_rtg_norm_by_seed,
        "enhanced_regret_score_by_seed": enhanced_regret_score_by_seed,
        "use_enhanced_regret": True,
    }
    scalar_sources = {
        "base_regret": "base_regret_norm_by_seed",
        "solvable_rate": "solvable_rate_by_seed",
        "learnability": "learnability_norm_by_seed",
        "delta_rtg": "delta_rtg_norm_by_seed",
        "enhanced_regret_score": "enhanced_regret_score_by_seed",
    }
    for scalar_name, by_seed_name in scalar_sources.items():
        values = list(metrics[by_seed_name].values())
        if values:
            metrics[scalar_name] = float(np.mean(values))

    return external_scores, metrics


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
