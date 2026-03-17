"""
负责单个 flat job batch 的批量前向执行，以及 RTG/Action 解码阶段的调度。
该模块连接 collate、模型前向、阶段 profiling 与结果组装，是单批次推理主流程。
Runs batched forward execution for one flat job batch and orchestrates RTG/action decode stages.
Connects collation, model forward, stage profiling, and result assembly as the single-batch inference flow.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple


def forward_job_batch_impl(
    teacher: Any,
    jobs: List[Dict[str, Any]],
    batch_predict_rtgs_mode_fn: Any,
    elapsed_ms_fn: Any,
    decode_rtg_jobs_batched_fn: Any,
) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    profile_enabled = bool(getattr(teacher, "_profile_enabled", False))
    total_start = time.perf_counter() if profile_enabled else 0.0

    collate_start = time.perf_counter() if profile_enabled else 0.0
    batched_data, batch_meta = teacher._collate_jobs_with_padding(jobs)
    collate_ms = elapsed_ms_fn(collate_start, profile_enabled)

    if not batch_predict_rtgs_mode_fn(jobs):
        action_results_by_job = teacher._decode_action_stage_batched(
            batched_data=batched_data,
            batch_meta=batch_meta,
        )
        action_stage_profile = getattr(teacher, "_last_action_stage_profile", {})
        teacher._last_forward_batch_profile = {
            "batch_jobs": len(jobs),
            "stage_ms": {
                "collate": collate_ms,
                "model_rtg": 0.0,
                "rtg_decode": 0.0,
                "model_action": float(action_stage_profile.get("stage_ms", {}).get("model_action", 0.0)),
                "action_decode": float(action_stage_profile.get("stage_ms", {}).get("action_decode", 0.0)),
                "total": elapsed_ms_fn(total_start, profile_enabled),
            },
            "detail_ms": {
                "collate": dict(batch_meta.get("collate_profile", {})),
                "action": dict(action_stage_profile.get("detail_ms", {})),
            },
        }
        return [
            {
                "env_idx": job["env_idx"],
                "prepared": job["prepared"],
                "action_results": action_results_by_job[idx],
                "rtg_results": {},
                "processed_rtg_veh_ids": [],
            }
            for idx, job in enumerate(jobs)
        ]

    model_rtg_start = time.perf_counter() if profile_enabled else 0.0
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    model_rtg_ms = elapsed_ms_fn(model_rtg_start, profile_enabled)
    rtg_logits = preds["rtg_preds"].float()

    batch_jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    rtg_decode_start = time.perf_counter() if profile_enabled else 0.0
    rtg_results_by_job, processed_rtg_veh_ids_by_job = decode_rtg_jobs_batched_fn(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        decode_meta=batch_meta["decode_meta"]["rtg"],
    )
    rtg_decode_ms = elapsed_ms_fn(rtg_decode_start, profile_enabled)

    action_results_by_job = teacher._decode_action_stage_batched(
        batched_data=batched_data,
        batch_meta=batch_meta,
    )
    action_stage_profile = getattr(teacher, "_last_action_stage_profile", {})

    teacher._last_forward_batch_profile = {
        "batch_jobs": len(jobs),
        "stage_ms": {
            "collate": collate_ms,
            "model_rtg": model_rtg_ms,
            "rtg_decode": rtg_decode_ms,
            "model_action": float(action_stage_profile.get("stage_ms", {}).get("model_action", 0.0)),
            "action_decode": float(action_stage_profile.get("stage_ms", {}).get("action_decode", 0.0)),
            "total": elapsed_ms_fn(total_start, profile_enabled),
        },
        "detail_ms": {
            "collate": dict(batch_meta.get("collate_profile", {})),
            "action": dict(action_stage_profile.get("detail_ms", {})),
        },
    }
    return [
        {
            "env_idx": job["env_idx"],
            "prepared": job["prepared"],
            "action_results": action_results_by_job[idx],
            "rtg_results": rtg_results_by_job[idx],
            "processed_rtg_veh_ids": processed_rtg_veh_ids_by_job[idx],
        }
        for idx, job in enumerate(batch_jobs)
    ]
