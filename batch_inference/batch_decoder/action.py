"""
负责动作 logits 的批量解码与按车辆组织的结果回填。
该模块将 action 头输出转换成连续动作结果，并使用无状态 sampling tickets 保持可复现。
Handles batched action-logit decoding and per-vehicle result assembly.
Converts action-head outputs into continuous actions while using stateless sampling tickets for reproducibility.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from .sampling import (
    build_action_probabilities,
    build_row_keys,
    build_stateless_uniforms,
    sample_categorical_from_uniform,
)


_ACTION_STAGE_TAG = 1


def _undiscretize_action_indices(
    action_idx: torch.Tensor,
    teacher: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    accel_idx = torch.div(action_idx, int(teacher.steer_discretization), rounding_mode="floor")
    steer_idx = torch.remainder(action_idx, int(teacher.steer_discretization))

    accel = accel_idx.to(dtype=torch.float32) / float(teacher.accel_discretization - 1)
    steer = steer_idx.to(dtype=torch.float32) / float(teacher.steer_discretization - 1)
    accel = accel * float(teacher.max_accel - teacher.min_accel) + float(teacher.min_accel)
    steer = steer * float(teacher.max_steer - teacher.min_steer) + float(teacher.min_steer)
    return accel, steer


def _sample_action_indices(
    flat_logits: torch.Tensor,
    temperatures: torch.Tensor,
    nucleus_sampling: torch.Tensor,
    nucleus_thresholds: torch.Tensor,
    sampling_seed: np.ndarray,
    step_t: np.ndarray,
    veh_id: np.ndarray,
) -> torch.Tensor:
    action_probs = build_action_probabilities(
        logits=flat_logits,
        temperatures=temperatures,
        nucleus_sampling=nucleus_sampling,
        nucleus_thresholds=nucleus_thresholds,
    )
    row_keys = build_row_keys(step_t=step_t, veh_ids=veh_id, stage_tag=_ACTION_STAGE_TAG)
    uniforms = build_stateless_uniforms(
        base_seed=sampling_seed,
        row_keys=row_keys,
        draws_per_row=1,
        as_tensor=True,
        device=flat_logits.device,
        dtype=flat_logits.dtype,
    ).reshape((-1,))
    return sample_categorical_from_uniform(action_probs, uniforms)


def decode_action_jobs_batched_impl(
    teacher: Any,
    action_logits: torch.Tensor,
    decode_meta: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """批量解码 action logits，并返回 flat 结果数组。

    Decode action logits in batch and return the flat result arrays.
    """
    job_offsets = decode_meta["job_offsets"]
    if decode_meta["job_idx"].size == 0:
        return {
            "veh_id": np.zeros((0,), dtype=np.int64),
            "values": np.zeros((0, 2), dtype=np.float32),
            "job_offsets": np.asarray(job_offsets, dtype=np.int64),
            "job_count": int(decode_meta["job_count"][0]),
        }

    device = action_logits.device
    row_batch_idx = decode_meta.get("job_idx_t")
    row_idx_in_model = decode_meta.get("idx_in_model_t")
    row_token_index = decode_meta.get("token_index_t")
    if row_batch_idx is None or row_idx_in_model is None or row_token_index is None:
        row_batch_idx = torch.as_tensor(decode_meta["job_idx"], dtype=torch.long, device=device)
        row_idx_in_model = torch.as_tensor(decode_meta["idx_in_model"], dtype=torch.long, device=device)
        row_token_index = torch.as_tensor(decode_meta["token_index"], dtype=torch.long, device=device)
    flat_logits = action_logits[row_batch_idx, row_idx_in_model, row_token_index]

    temperatures = decode_meta.get("temperature_t")
    if temperatures is None:
        temperatures = torch.as_tensor(decode_meta["temperature"], dtype=flat_logits.dtype, device=device)
    nucleus_sampling = decode_meta.get("nucleus_sampling_t")
    if nucleus_sampling is None:
        nucleus_sampling = torch.as_tensor(decode_meta["nucleus_sampling"], dtype=torch.bool, device=device)
    nucleus_thresholds = decode_meta.get("nucleus_threshold_t")
    if nucleus_thresholds is None:
        nucleus_thresholds = torch.as_tensor(
            decode_meta["nucleus_threshold"],
            dtype=flat_logits.dtype,
            device=device,
        )

    action_idx = _sample_action_indices(
        flat_logits=flat_logits,
        temperatures=temperatures,
        nucleus_sampling=nucleus_sampling,
        nucleus_thresholds=nucleus_thresholds,
        sampling_seed=decode_meta["sampling_seed"],
        step_t=decode_meta["step_t"],
        veh_id=decode_meta["veh_id"],
    )
    accel_t, steer_t = _undiscretize_action_indices(action_idx, teacher)
    return {
        "veh_id": np.asarray(decode_meta["veh_id"], dtype=np.int64),
        "values": torch.stack((accel_t, steer_t), dim=1).detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        ),
        "job_offsets": np.asarray(job_offsets, dtype=np.int64),
        "job_count": int(decode_meta["job_count"][0]),
    }


def export_selected_action_logits_by_job_impl(
    action_logits: torch.Tensor,
    decode_meta: Dict[str, np.ndarray],
    selected_job_indices: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """按选中的 job 导出 raw action logits。 / Export raw action logits for selected jobs only."""
    if not selected_job_indices:
        return {}

    job_count = int(decode_meta["job_count"][0])
    device = action_logits.device
    row_batch_idx = decode_meta.get("job_idx_t")
    row_idx_in_model = decode_meta.get("idx_in_model_t")
    row_token_index = decode_meta.get("token_index_t")
    if row_batch_idx is None or row_idx_in_model is None or row_token_index is None:
        row_batch_idx = torch.as_tensor(
            decode_meta["job_idx"],
            dtype=torch.long,
            device=device,
        )
        row_idx_in_model = torch.as_tensor(
            decode_meta["idx_in_model"],
            dtype=torch.long,
            device=device,
        )
        row_token_index = torch.as_tensor(
            decode_meta["token_index"],
            dtype=torch.long,
            device=device,
        )

    flat_logits = action_logits[row_batch_idx, row_idx_in_model, row_token_index]
    job_offsets = decode_meta["job_offsets"]

    logits_by_job: Dict[int, torch.Tensor] = {}
    for job_idx in selected_job_indices:
        job_idx_int = int(job_idx)
        if job_idx_int < 0 or job_idx_int >= job_count:
            raise ValueError("Selected action-logit job index out of range.")
        start = int(job_offsets[job_idx_int])
        end = int(job_offsets[job_idx_int + 1])
        if end - start != 1:
            raise ValueError(
                "ego_ctrlsim action-logit export expects exactly one action row per selected job."
            )
        logits_by_job[job_idx_int] = flat_logits[start].detach().clone()

    if len(logits_by_job) != len(selected_job_indices):
        raise ValueError("Selected action-logit export job/result count mismatch.")
    return logits_by_job


def decode_action_stage_batched_impl(
    teacher: Any,
    batched_data: Any,
    batch_meta: Dict[str, Any],
    return_logits: bool = False,
    logits_job_indices: Sequence[int] = (),
    cached_scene_enc: Any = None,
):
    """执行 action 阶段前向与 batched 解码。

    Run the action-stage forward pass and batched decode.
    """
    with torch.inference_mode():
        with teacher.model_forward_context():
            preds = teacher.model(
                batched_data, eval=True, cached_scene_enc=cached_scene_enc
            )
        action_logits = preds["action_preds"].float()

        jobs = batch_meta["jobs"]
        flat_action_results = decode_action_jobs_batched_impl(
            teacher=teacher,
            action_logits=action_logits,
            decode_meta=batch_meta["decode_meta"]["action"],
        )
        job_count = len(jobs)
        if int(flat_action_results["job_count"]) != job_count:
            raise ValueError("Action stage job/result count mismatch.")
        if not return_logits:
            return flat_action_results

        selected_logits_by_job = export_selected_action_logits_by_job_impl(
            action_logits=action_logits,
            decode_meta=batch_meta["decode_meta"]["action"],
            selected_job_indices=logits_job_indices,
        )
        action_logits_by_job = [None] * job_count
        for job_idx, logits in selected_logits_by_job.items():
            action_logits_by_job[job_idx] = logits

        if len(action_logits_by_job) != job_count:
            raise ValueError("Action-logit export job/result count mismatch.")
        return flat_action_results, action_logits_by_job
