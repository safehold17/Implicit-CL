"""Rollout-local helpers for Nocturne ego ctrl-sim teacher paths."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from envs.nocturne_ctrlsim.core.episode_runtime import split_prepared_pack_batch


def needs_ego_ctrlsim_dual_forward(args: Any) -> bool:
    """Return whether the rollout needs ego ctrl-sim logits or RTGs."""
    return bool(
        getattr(args, "use_ego_ctrlsim_kl_loss", False)
        or getattr(args, "use_policy_reweighting", False)
        or getattr(args, "use_enhanced_regret", False)
    )


def run_nocturne_batched_step(
    *,
    venv: Any,
    external_teacher: Any,
    args: Any,
    action: Any,
    reset_random: bool = False,
    auto_reset_on_done: bool = True,
) -> tuple[Any, Any, Any, Sequence[dict[str, Any]]]:
    """Run one Nocturne batched step with optional ego ctrl-sim outputs."""
    prepared_batch = venv.step_prepare(action)
    opponent_prepared, ego_ctrlsim_prepared = split_prepared_pack_batch(
        prepared_batch
    )

    ego_ctrlsim_logits = [None] * len(ego_ctrlsim_prepared)
    ego_ctrlsim_rtgs = [None] * len(ego_ctrlsim_prepared)
    ego_ctrlsim_rtg_metadata = [None] * len(ego_ctrlsim_prepared)
    if needs_ego_ctrlsim_dual_forward(args):
        if external_teacher is None:
            raise RuntimeError("Nocturne training requires an ExternalTeacher.")
        (
            model_outputs,
            ego_ctrlsim_logits,
            ego_ctrlsim_rtgs,
            ego_ctrlsim_rtg_metadata,
        ) = external_teacher.run_batched_forward_with_ego_logits(
            opponent_prepared,
            ego_ctrlsim_prepared,
        )
    elif all(item is None for item in opponent_prepared):
        model_outputs = [None] * len(opponent_prepared)
    else:
        if external_teacher is None:
            raise RuntimeError("Nocturne training requires an ExternalTeacher.")
        model_outputs = external_teacher.run_batched_forward(opponent_prepared)

    obs, reward, done, infos = venv.step_complete(
        model_outputs,
        reset_random=reset_random,
        auto_reset_on_done=auto_reset_on_done,
    )

    for info, logits, rtg, rtg_metadata in zip(
        infos,
        ego_ctrlsim_logits,
        ego_ctrlsim_rtgs,
        ego_ctrlsim_rtg_metadata,
    ):
        info["ego_ctrlsim_action_logits"] = logits
        info["ego_ctrlsim_pred_rtg"] = rtg
        info["ego_ctrlsim_pred_rtg_metadata"] = rtg_metadata
        info["ego_ctrlsim_pred_rtg_step"] = (
            None if rtg_metadata is None else int(rtg_metadata["step_t"])
        )

    return obs, reward, done, infos


def collect_ego_ctrlsim_action_logits(
    infos: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Collect ego ctrl-sim action logits and validity flags from step infos."""
    logits_list = [
        info.get("ego_ctrlsim_action_logits")
        for info in infos
    ]
    if not logits_list or all(logits is None for logits in logits_list):
        return None, None

    first_non_none = next(logits for logits in logits_list if logits is not None)
    action_dim = int(np.asarray(first_non_none).shape[0])
    batch = np.zeros((len(logits_list), action_dim), dtype=np.float32)
    valid = np.zeros((len(logits_list), 1), dtype="bool")
    for idx, logits in enumerate(logits_list):
        if logits is None:
            continue
        batch[idx] = np.asarray(logits, dtype=np.float32)
        valid[idx, 0] = True
    return (
        torch.from_numpy(batch),
        torch.from_numpy(valid),
    )
