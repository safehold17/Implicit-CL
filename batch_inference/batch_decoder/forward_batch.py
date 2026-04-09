"""
负责单个 flat job batch 的批量前向执行，以及 RTG/Action 解码阶段的调度。
该模块连接 collate、模型前向与 flat decode 结果汇总，是单批次推理主流程。
Runs batched forward execution for one flat job batch and orchestrates RTG/action decode stages.
It connects collation, model forward, and flat decode-result aggregation as the single-batch inference flow.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .action import decode_action_stage_batched_impl


def _build_empty_flat_rtg_results(job_count: int) -> Dict[str, Any]:
    """构造 action-only batch 的空 RTG flat 结果。

    当一批 jobs 不需要 RTG 预测时，前向路径仍然返回统一的 flat batch 结构。
    这个 helper 专门负责生成与 RTG decode 合约一致的空结果字典，这样后续的
    env 聚合逻辑可以无分支地消费 `flat_rtg_results`，避免在热路径里到处分叉判断。

    Build the empty RTG flat-result payload for an action-only batch.
    The action-only path still returns the same flat batch contract so later
    env-level aggregation can consume `flat_rtg_results` without branching all
    over the hot path.
    """
    empty_offsets = np.zeros((job_count + 1,), dtype=np.int64)
    return {
        "veh_id": np.zeros((0,), dtype=np.int64),
        "values": np.zeros((0, 3), dtype=np.float32),
        "job_offsets": empty_offsets,
        "processed_veh_ids": np.zeros((0,), dtype=np.int64),
        "processed_offsets": empty_offsets.copy(),
        "job_count": job_count,
    }


def _build_forward_batch_output(
    jobs: List[Dict[str, Any]],
    *,
    flat_action_results: Dict[str, Any],
    flat_rtg_results: Dict[str, Any],
    ego_action_scales_by_job: List[float],
    action_logits_by_job: List[torch.Tensor | None],
) -> Dict[str, Any]:
    """将单批次前向产物整理成统一 flat batch 返回结构。

    该 helper 把模型前向、RTG decode、action decode 的产物收口为一个字典，
    供上层直接做 env 级聚合。它显式保留每个 job 的 env 映射、类型、prepared
    引用和可选 logits，避免先切成 per-job 小字典，再重新按 env 拼接的重复搬运。

    Collect the outputs of one batched forward pass into a single flat batch
    structure for downstream env-level aggregation. It keeps the per-job env
    mapping, type, prepared reference, and optional logits explicitly so the
    caller does not need to materialize per-job mini dicts and stitch them back
    together again.
    """
    return {
        "jobs": jobs,
        "env_idx_by_job": np.asarray(
            [int(job["env_idx"]) for job in jobs],
            dtype=np.int64,
        ),
        "job_types": [str(job.get("job_type", "opponent")) for job in jobs],
        "prepared_by_job": [job["prepared"] for job in jobs],
        "ego_action_scales_by_job": [float(value) for value in ego_action_scales_by_job],
        "action_logits_by_job": action_logits_by_job,
        "flat_action_results": flat_action_results,
        "flat_rtg_results": flat_rtg_results,
    }


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
) -> Dict[str, Any]:
    """执行单个 flat job batch 的两阶段批量前向。

    该函数只做一次 collate，并返回统一的 flat batch 结果对象。调用方随后可以
    直接按 job offset 和 env 映射做一次性聚合，不再为每个 job 切片生成临时小数组。

    Execute the batched forward pass for one flat job batch and return a single
    flat batch result object. The caller can then aggregate by env using job
    offsets directly, avoiding per-job slicing and temporary mini arrays.
    """
    if not jobs:
        return _build_forward_batch_output(
            [],
            flat_action_results={
                "veh_id": np.zeros((0,), dtype=np.int64),
                "values": np.zeros((0, 2), dtype=np.float32),
                "job_offsets": np.zeros((1,), dtype=np.int64),
                "job_count": 0,
            },
            flat_rtg_results=_build_empty_flat_rtg_results(job_count=0),
            ego_action_scales_by_job=[],
            action_logits_by_job=[],
        )

    batched_data, batch_meta = teacher._collate_jobs_with_padding(jobs)
    logits_job_indices = [
        idx
        for idx, job in enumerate(jobs)
        if bool(job.get("return_action_logits", False))
    ]
    need_action_logits = bool(logits_job_indices)

    if not batch_predict_rtgs_mode_fn(jobs):
        action_decode_outputs = decode_action_stage_batched_impl(
            teacher=teacher,
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
        return _build_forward_batch_output(
            jobs,
            flat_action_results=flat_action_results,
            flat_rtg_results=_build_empty_flat_rtg_results(job_count=len(jobs)),
            ego_action_scales_by_job=[1.0] * len(jobs),
            action_logits_by_job=action_logits_by_job,
        )

    try:
        with torch.inference_mode():
            with teacher.model_forward_context():
                preds, scene_enc = teacher.model(
                    batched_data,
                    eval=True,
                    return_enc=True,
                    need_action=False,
                    need_rtg=True,
                    need_state=False,
                )
    except RuntimeError as exc:
        print(
            "[forward_batch] teacher_forward_error "
            f"exc={type(exc).__name__}: {exc}"
        )
        raise
    rtg_logits = preds["rtg_preds"]

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

    action_decode_outputs = decode_action_stage_batched_impl(
        teacher=teacher,
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

    return _build_forward_batch_output(
        batch_jobs,
        flat_action_results=flat_action_results,
        flat_rtg_results=flat_rtg_results,
        ego_action_scales_by_job=ego_action_scales_by_job,
        action_logits_by_job=action_logits_by_job,
    )
