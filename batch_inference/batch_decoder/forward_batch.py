"""
负责单个 flat job batch 的批量前向执行，以及 RTG/Action 解码阶段的调度。
该模块连接 collate、模型前向与 flat decode 结果切片，是单批次推理主流程。
Runs batched forward execution for one flat job batch and orchestrates RTG/action decode stages.
It connects collation, model forward, and flat decode-result slicing as the single-batch inference flow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _slice_flat_rows(
    veh_ids: Any,
    values: Any,
    job_offsets: Any,
    job_idx: int,
    *,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 job 索引切一段 flat 结果数组。

    Slice one job's segment from a flat result array using its job index.
    """
    veh_ids_array = np.asarray(veh_ids, dtype=np.int64).reshape((-1,))
    values_array = np.asarray(values, dtype=np.float32).reshape((-1, width))
    offsets_array = np.asarray(job_offsets, dtype=np.int64).reshape((-1,))
    start = int(offsets_array[job_idx])
    end = int(offsets_array[job_idx + 1])
    return veh_ids_array[start:end], values_array[start:end]


def _slice_flat_ids(
    values: Any,
    job_offsets: Any,
    job_idx: int,
) -> np.ndarray:
    """按 job 索引切一段 flat 一维 ID 数组。

    Slice one job's segment from a flat 1D id array using its job index.
    """
    values_array = np.asarray(values, dtype=np.int64).reshape((-1,))
    offsets_array = np.asarray(job_offsets, dtype=np.int64).reshape((-1,))
    start = int(offsets_array[job_idx])
    end = int(offsets_array[job_idx + 1])
    return values_array[start:end]


def batch_predict_rtgs_mode(jobs: List[Dict[str, Any]]) -> bool:
    """检查一批 jobs 是否统一启用 RTG 预测。

    Check whether a batch of jobs consistently enables RTG prediction.
    """
    predict_rtgs_flags = {bool(job.get("predict_rtgs", True)) for job in jobs}
    if not predict_rtgs_flags:
        return True
    if len(predict_rtgs_flags) != 1:
        raise ValueError("Mixed predict_rtgs modes in the same batch are not supported.")
    return predict_rtgs_flags.pop()


def forward_job_batch_impl(
    teacher: Any,
    jobs: List[Dict[str, Any]],
    batch_predict_rtgs_mode_fn: Any,
    decode_rtg_jobs_batched_fn: Any,
) -> List[Dict[str, Any]]:
    """执行单个 flat job batch 的两阶段批量前向。

    Execute the batched forward pass for one flat job batch, including the optional RTG stage.
    """
    if not jobs:
        return []

    batched_data, batch_meta = teacher._collate_jobs_with_padding(jobs)
    logits_job_indices = [
        idx
        for idx, job in enumerate(jobs)
        if bool(job.get("return_action_logits", False))
    ]
    need_action_logits = bool(logits_job_indices)

    if not batch_predict_rtgs_mode_fn(jobs):
        action_decode_outputs = teacher._decode_action_stage_batched(
            batched_data=batched_data,
            batch_meta=batch_meta,
            return_logits=need_action_logits,
            logits_job_indices=logits_job_indices,
        )
        if need_action_logits:
            flat_action_results, action_logits_by_job = action_decode_outputs
        else:
            flat_action_results = action_decode_outputs
            action_logits_by_job = [None] * len(jobs)
        job_outputs: List[Dict[str, Any]] = []
        for idx, job in enumerate(jobs):
            action_veh_ids, action_values = _slice_flat_rows(
                flat_action_results["veh_id"],
                flat_action_results["values"],
                flat_action_results["job_offsets"],
                idx,
                width=2,
            )
            job_outputs.append(
                {
                    "env_idx": job["env_idx"],
                    "prepared": job["prepared"],
                    "job_type": job.get("job_type", "opponent"),
                    "action_veh_ids": action_veh_ids,
                    "action_values": action_values,
                    "action_logits": (
                        action_logits_by_job[idx]
                        if bool(job.get("return_action_logits", False))
                        else None
                    ),
                    "rtg_veh_ids": np.zeros((0,), dtype=np.int64),
                    "rtg_values": np.zeros((0, 3), dtype=np.float32),
                    "processed_rtg_veh_ids": np.zeros((0,), dtype=np.int64),
                }
            )
        return job_outputs

    try:
        with torch.inference_mode():
            with teacher.model_forward_context():
                preds, scene_enc = teacher.model(batched_data, eval=True, return_enc=True)
    except RuntimeError as exc:
        print(
            "[forward_batch] teacher_forward_error "
            f"exc={type(exc).__name__}: {exc}"
        )
        raise
    rtg_logits = preds["rtg_preds"].float()

    batch_jobs = batch_meta["jobs"]
    flat_rtg_results = decode_rtg_jobs_batched_fn(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        decode_meta=batch_meta["decode_meta"]["rtg"],
    )
    ego_action_scales_by_job = teacher._compute_ego_action_scales_by_job(
        jobs=batch_jobs,
        rtg_logits=rtg_logits,
        decode_meta=batch_meta["decode_meta"]["rtg"],
        flat_rtg_results=flat_rtg_results,
    )

    action_decode_outputs = teacher._decode_action_stage_batched(
        batched_data=batched_data,
        batch_meta=batch_meta,
        return_logits=need_action_logits,
        logits_job_indices=logits_job_indices,
        cached_scene_enc=scene_enc,
    )
    if need_action_logits:
        flat_action_results, action_logits_by_job = action_decode_outputs
    else:
        flat_action_results = action_decode_outputs
        action_logits_by_job = [None] * len(batch_jobs)

    job_outputs: List[Dict[str, Any]] = []
    for idx, job in enumerate(batch_jobs):
        action_veh_ids, action_values = _slice_flat_rows(
            flat_action_results["veh_id"],
            flat_action_results["values"],
            flat_action_results["job_offsets"],
            idx,
            width=2,
        )
        rtg_veh_ids, rtg_values = _slice_flat_rows(
            flat_rtg_results["veh_id"],
            flat_rtg_results["values"],
            flat_rtg_results["job_offsets"],
            idx,
            width=3,
        )
        processed_rtg_veh_ids = _slice_flat_ids(
            flat_rtg_results["processed_veh_ids"],
            flat_rtg_results["processed_offsets"],
            idx,
        )
        job_outputs.append(
            {
                "env_idx": job["env_idx"],
                "prepared": job["prepared"],
                "job_type": job.get("job_type", "opponent"),
                "action_veh_ids": action_veh_ids,
                "action_values": action_values,
                "ego_action_scale": ego_action_scales_by_job[idx],
                "action_logits": (
                    action_logits_by_job[idx]
                    if bool(job.get("return_action_logits", False))
                    else None
                ),
                "rtg_veh_ids": rtg_veh_ids,
                "rtg_values": rtg_values,
                "processed_rtg_veh_ids": processed_rtg_veh_ids,
            }
        )
    return job_outputs
