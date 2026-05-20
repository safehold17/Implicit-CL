"""Rollout-local implementations for PLR/ACCEL regret enhancement."""

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
    normalize_by_running_standard_score,
)


ENHANCED_REGRET_RAW_TERM_KEYS = {
    "base_regret": "base_regret_by_seed",
    "learnability": "learnability_by_seed",
    "delta_rtg": "delta_rtg_by_seed",
}


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


def build_rollout_seed_index(rollouts) -> tuple[dict[int, int], int]:
    """Build the actor-to-seed index for one rollout batch."""
    level_seeds = rollouts.level_seeds.cpu()
    num_actors = int(rollouts.action_log_dist.shape[1])
    total_steps = int(level_seeds.shape[0])
    actor_seed_by_index = {}

    for actor_index in range(num_actors):
        actor_seed_by_index[actor_index] = int(
            level_seeds[total_steps - 1, actor_index].item()
        )

    return actor_seed_by_index, num_actors


def compute_nocturne_base_regret_by_seed(
    rollouts,
) -> tuple[dict[int, float], dict[int, int], int]:
    """Compute positive value-loss regret by rollout seed."""
    level_seeds = rollouts.level_seeds.cpu()
    actor_seed_by_index, num_actors = build_rollout_seed_index(rollouts)
    total_steps = int(rollouts.action_log_dist.shape[0])
    done = (~(rollouts.masks > 0)).cpu()
    cliffhanger = (~(rollouts.cliffhanger_masks > 0)).cpu()
    returns = rollouts.returns.cpu()
    if rollouts.use_popart:
        value_preds = rollouts.denorm_value_preds.cpu()
    else:
        value_preds = rollouts.value_preds.cpu()

    regret_sum_by_seed = defaultdict(float)
    regret_count_by_seed = defaultdict(int)

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


def compute_nocturne_enhanced_regret_raw_metrics(
    rollouts,
    rollout_info: Mapping[str, object],
    *,
    use_base_regret: bool = True,
    use_solvable_rate: bool = True,
    use_ctrlsim_rtg_gap: bool = True,
) -> tuple[dict[str, object], dict[int, int], int]:
    """Build raw per-seed enhanced regret metrics before normalization."""
    actor_seed_by_index, num_actors = build_rollout_seed_index(rollouts)
    base_regret_by_seed = {}
    if use_base_regret:
        base_regret_by_seed, _, _ = compute_nocturne_base_regret_by_seed(rollouts)
    attempt_count_by_seed = rollout_info.get("attempt_count_by_seed", {})
    success_count_by_seed = rollout_info.get("success_count_by_seed", {})
    delta_rtg_segments_by_seed = rollout_info.get("delta_rtg_segments_by_seed", {})

    all_seeds = set()
    if use_base_regret:
        all_seeds.update(base_regret_by_seed.keys())
    if use_solvable_rate:
        all_seeds.update(attempt_count_by_seed.keys())
        all_seeds.update(success_count_by_seed.keys())
    if use_ctrlsim_rtg_gap:
        all_seeds.update(delta_rtg_segments_by_seed.keys())

    solvable_rate_by_seed = {}
    learnability_by_seed = {}
    delta_rtg_by_seed = {}
    for seed in all_seeds:
        seed_int = int(seed)
        if use_solvable_rate:
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

        if use_ctrlsim_rtg_gap:
            delta_rtg = compute_mean_delta_rtg(delta_rtg_segments_by_seed.get(seed, ()))
            if delta_rtg is not None:
                delta_rtg_by_seed[seed_int] = delta_rtg

    metrics = {"use_enhanced_regret": True}
    if use_base_regret:
        metrics["base_regret_by_seed"] = {
            int(seed): float(score)
            for seed, score in base_regret_by_seed.items()
        }
    if use_solvable_rate:
        metrics["solvable_rate_by_seed"] = solvable_rate_by_seed
        metrics["learnability_by_seed"] = learnability_by_seed
    if use_ctrlsim_rtg_gap:
        metrics["delta_rtg_by_seed"] = delta_rtg_by_seed
    return metrics, actor_seed_by_index, num_actors


def normalize_nocturne_enhanced_regret_scores(
    raw_metrics: Mapping[str, object],
    actor_seed_by_index: Mapping[int, int],
    num_actors: int,
    *,
    running_stats_by_term: Mapping[str, Mapping[str, float]],
    use_base_regret: bool = True,
    use_solvable_rate: bool = True,
    use_ctrlsim_rtg_gap: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build per-actor sampler scores from raw metrics and running stats."""
    base_regret_by_seed = (
        raw_metrics.get("base_regret_by_seed", {}) if use_base_regret else {}
    )
    solvable_rate_by_seed = (
        raw_metrics.get("solvable_rate_by_seed", {}) if use_solvable_rate else {}
    )
    learnability_by_seed = (
        raw_metrics.get("learnability_by_seed", {}) if use_solvable_rate else {}
    )
    delta_rtg_by_seed = (
        raw_metrics.get("delta_rtg_by_seed", {}) if use_ctrlsim_rtg_gap else {}
    )

    all_seeds = set()
    if use_base_regret:
        all_seeds.update(base_regret_by_seed.keys())
    if use_solvable_rate:
        all_seeds.update(solvable_rate_by_seed.keys())
        all_seeds.update(learnability_by_seed.keys())
    if use_ctrlsim_rtg_gap:
        all_seeds.update(delta_rtg_by_seed.keys())
    if not all_seeds and not (
        use_base_regret or use_solvable_rate or use_ctrlsim_rtg_gap
    ):
        all_seeds.update(int(seed) for seed in actor_seed_by_index.values())

    base_regret_norm_by_seed = {}
    learnability_norm_by_seed = {}
    delta_rtg_norm_by_seed = {}
    enhanced_regret_score_by_seed = {}
    base_regret_stats = (
        running_stats_by_term["base_regret"] if use_base_regret else None
    )
    learnability_stats = (
        running_stats_by_term["learnability"] if use_solvable_rate else None
    )
    delta_rtg_stats = (
        running_stats_by_term["delta_rtg"] if use_ctrlsim_rtg_gap else None
    )

    for seed in all_seeds:
        seed_int = int(seed)

        base_regret = base_regret_by_seed.get(seed)
        base_regret_norm = None
        if use_base_regret and base_regret is not None:
            base_regret_norm = normalize_by_running_standard_score(
                base_regret,
                float(base_regret_stats["mean"]),
                float(base_regret_stats["std"]),
            )
        if base_regret_norm is not None:
            base_regret_norm_by_seed[seed_int] = base_regret_norm

        learnability = learnability_by_seed.get(seed)
        learnability_norm = None
        if use_solvable_rate and learnability is not None:
            learnability_norm = normalize_by_running_standard_score(
                learnability,
                float(learnability_stats["mean"]),
                float(learnability_stats["std"]),
            )
        if learnability_norm is not None:
            learnability_norm_by_seed[seed_int] = learnability_norm

        delta_rtg = delta_rtg_by_seed.get(seed)
        delta_rtg_norm = None
        if use_ctrlsim_rtg_gap and delta_rtg is not None:
            delta_rtg_norm = normalize_by_running_standard_score(
                delta_rtg,
                float(delta_rtg_stats["mean"]),
                float(delta_rtg_stats["std"]),
            )
        if delta_rtg_norm is not None:
            delta_rtg_norm_by_seed[seed_int] = delta_rtg_norm

        enhanced_regret_score_by_seed[seed_int] = combine_enhanced_regret_score(
            base_regret=base_regret_norm,
            learnability=learnability_norm,
            delta_rtg=delta_rtg_norm,
            use_base_regret=use_base_regret,
            use_solvable_rate=use_solvable_rate,
            use_ctrlsim_rtg_gap=use_ctrlsim_rtg_gap,
        )

    external_scores = torch.zeros((num_actors, 1), dtype=torch.float32)
    for actor_index, seed in actor_seed_by_index.items():
        external_scores[actor_index, 0] = float(
            enhanced_regret_score_by_seed.get(int(seed), 0.0)
        )

    metrics = {
        "enhanced_regret_score_by_seed": enhanced_regret_score_by_seed,
        "use_enhanced_regret": bool(raw_metrics.get("use_enhanced_regret", True)),
    }
    if use_base_regret:
        metrics["base_regret_by_seed"] = dict(base_regret_by_seed)
    if use_solvable_rate:
        metrics["solvable_rate_by_seed"] = dict(solvable_rate_by_seed)
        metrics["learnability_by_seed"] = dict(learnability_by_seed)
    if use_ctrlsim_rtg_gap:
        metrics["delta_rtg_by_seed"] = dict(delta_rtg_by_seed)
    if use_base_regret:
        metrics["base_regret_norm_by_seed"] = base_regret_norm_by_seed
    if use_solvable_rate:
        metrics["learnability_norm_by_seed"] = learnability_norm_by_seed
    if use_ctrlsim_rtg_gap:
        metrics["delta_rtg_norm_by_seed"] = delta_rtg_norm_by_seed

    scalar_sources = {"enhanced_regret_score": "enhanced_regret_score_by_seed"}
    if use_base_regret:
        scalar_sources["base_regret"] = "base_regret_norm_by_seed"
    if use_solvable_rate:
        scalar_sources["solvable_rate"] = "solvable_rate_by_seed"
        scalar_sources["learnability"] = "learnability_norm_by_seed"
    if use_ctrlsim_rtg_gap:
        scalar_sources["delta_rtg"] = "delta_rtg_norm_by_seed"
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
