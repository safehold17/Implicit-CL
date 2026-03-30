"""
负责单个 flat job batch 的批量前向执行，以及 RTG/Action 解码阶段的调度。
该模块连接 collate、模型前向与结果组装，是单批次推理主流程。
Runs batched forward execution for one flat job batch and orchestrates RTG/action decode stages.
Connects collation, model forward, and result assembly as the single-batch inference flow.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def batch_predict_rtgs_mode(jobs: List[Dict[str, Any]]) -> bool:
    predict_rtgs_flags = {
        bool(job["focal_batch"].get("predict_rtgs", True))
        for job in jobs
    }
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
            action_results_by_job, action_logits_by_job = action_decode_outputs
        else:
            action_results_by_job = action_decode_outputs
            action_logits_by_job = [None] * len(jobs)
        return [
            {
                "env_idx": job["env_idx"],
                "prepared": job["prepared"],
                "job_type": job.get("job_type", "opponent"),
                "action_results": action_results_by_job[idx],
                "action_logits": (
                    action_logits_by_job[idx]
                    if bool(job.get("return_action_logits", False))
                    else None
                ),
                "rtg_results": {},
                "processed_rtg_veh_ids": [],
            }
            for idx, job in enumerate(jobs)
        ]

    # first stage forward for RTG prediction
    try:
        with torch.no_grad():
            with teacher.model_forward_context():
                preds = teacher.model(batched_data, eval=True)
    except RuntimeError as exc:
        print(
            "[forward_batch] teacher_forward_error "
            f"exc={type(exc).__name__}: {exc}"
        )
        raise
    rtg_logits = preds["rtg_preds"].float()

    # RTG decode to get RTG-aware prepared meta for the second stage forward 
    batch_jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    rtg_results_by_job, processed_rtg_veh_ids_by_job = decode_rtg_jobs_batched_fn(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        decode_meta=batch_meta["decode_meta"]["rtg"],
    )
    ego_action_scales_by_job = teacher._compute_ego_action_scales_by_job(
        jobs=batch_jobs,
        rtg_logits=rtg_logits,
        decode_meta=batch_meta["decode_meta"]["rtg"],
        rtg_results_by_job=rtg_results_by_job,
    )

    # second stage decode with RTG-aware prepared meta
    action_decode_outputs = teacher._decode_action_stage_batched(
        batched_data=batched_data,
        batch_meta=batch_meta,
        return_logits=need_action_logits,
        logits_job_indices=logits_job_indices,
    )
    if need_action_logits:
        action_results_by_job, action_logits_by_job = action_decode_outputs
    else:
        action_results_by_job = action_decode_outputs
        action_logits_by_job = [None] * len(batch_jobs)
    return [
        {
            "env_idx": job["env_idx"],
            "prepared": job["prepared"],
            "job_type": job.get("job_type", "opponent"),
            "action_results": action_results_by_job[idx],
            "ego_action_scale": ego_action_scales_by_job[idx],
            "action_logits": (
                action_logits_by_job[idx]
                if bool(job.get("return_action_logits", False))
                else None
            ),
            "rtg_results": rtg_results_by_job[idx],
            "processed_rtg_veh_ids": processed_rtg_veh_ids_by_job[idx],
        }
        for idx, job in enumerate(batch_jobs)
    ]
