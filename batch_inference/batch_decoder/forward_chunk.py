from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import torch

from .rtg import RTGCache


def forward_chunk_batched_impl(
    teacher: Any,
    chunk: List[Dict[str, Any]],
    chunk_predict_rtgs_mode_fn: Any,
    elapsed_ms_fn: Any,
    get_env_sampling_generator_fn: Any,
    decode_rtg_for_job_fn: Any,
    reserve_action_rng_states_for_job_fn: Any,
    decode_rtg_jobs_batched_fn: Any = None,
) -> List[Dict[str, Any]]:
    if not chunk:
        return []

    profile_enabled = bool(getattr(teacher, "_profile_enabled", False))
    total_start = time.perf_counter() if profile_enabled else 0.0

    collate_start = time.perf_counter() if profile_enabled else 0.0
    batched_data, batch_meta = teacher._collate_chunk_with_padding(chunk)
    collate_ms = elapsed_ms_fn(collate_start, profile_enabled)

    if not chunk_predict_rtgs_mode_fn(chunk):
        action_results_by_job = teacher._decode_action_stage_batched(
            batched_data=batched_data,
            batch_meta=batch_meta,
            reserved_rng_states_by_job=None,
        )
        action_stage_profile = getattr(teacher, "_last_action_stage_profile", {})
        teacher._last_forward_chunk_profile = {
            "chunk_jobs": len(chunk),
            "stage_ms": {
                "collate": collate_ms,
                "model_rtg": 0.0,
                "rtg_decode": 0.0,
                "rng_reserve": 0.0,
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
            for idx, job in enumerate(chunk)
        ]

    model_rtg_start = time.perf_counter() if profile_enabled else 0.0
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    model_rtg_ms = elapsed_ms_fn(model_rtg_start, profile_enabled)
    rtg_logits = preds["rtg_preds"].float()

    rtg_cache: RTGCache = {}
    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
    processed_rtg_veh_ids_by_job: List[List[int]] = []
    reserved_action_rng_states_by_job: List[Dict[int, torch.Tensor]] = []
    rtg_decode_ms = 0.0
    rng_reserve_ms = 0.0

    rtg_decode_start = time.perf_counter() if profile_enabled else 0.0
    if decode_rtg_jobs_batched_fn is not None:
        rtg_results_by_job, processed_rtg_veh_ids_by_job = decode_rtg_jobs_batched_fn(
            teacher=teacher,
            batched_data=batched_data,
            rtg_logits=rtg_logits,
            jobs=jobs,
            token_index_per_job=token_index_per_job,
            rtg_cache=rtg_cache,
        )
    else:
        for batch_idx, job in enumerate(jobs):
            token_index = int(token_index_per_job[batch_idx])
            env_generator = get_env_sampling_generator_fn(
                teacher=teacher,
                env_idx=int(job["env_idx"]),
                step_t=int(job["prepared"]["step_t"]),
                worker_rng_state=job["prepared"].get("worker_rng_state"),
            )
            rtg_results, processed_rtg_veh_ids = decode_rtg_for_job_fn(
                teacher=teacher,
                batched_data=batched_data,
                rtg_logits=rtg_logits,
                batch_idx=batch_idx,
                job=job,
                token_index=token_index,
                rtg_cache=rtg_cache,
                env_generator=env_generator,
            )
            rtg_results_by_job.append(rtg_results)
            processed_rtg_veh_ids_by_job.append(processed_rtg_veh_ids)
    rtg_decode_ms = elapsed_ms_fn(rtg_decode_start, profile_enabled)

    for job in jobs:
        env_generator = get_env_sampling_generator_fn(
            teacher=teacher,
            env_idx=int(job["env_idx"]),
            step_t=int(job["prepared"]["step_t"]),
            worker_rng_state=job["prepared"].get("worker_rng_state"),
        )
        rng_reserve_start = time.perf_counter() if profile_enabled else 0.0
        reserved_action_rng_states_by_job.append(
            reserve_action_rng_states_for_job_fn(
                teacher=teacher,
                job=job,
                env_generator=env_generator,
            )
        )
        rng_reserve_ms += elapsed_ms_fn(rng_reserve_start, profile_enabled)

    action_results_by_job = teacher._decode_action_stage_batched(
        batched_data=batched_data,
        batch_meta=batch_meta,
        reserved_rng_states_by_job=reserved_action_rng_states_by_job,
    )
    action_stage_profile = getattr(teacher, "_last_action_stage_profile", {})

    teacher._last_forward_chunk_profile = {
        "chunk_jobs": len(chunk),
        "stage_ms": {
            "collate": collate_ms,
            "model_rtg": model_rtg_ms,
            "rtg_decode": rtg_decode_ms,
            "rng_reserve": rng_reserve_ms,
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
        for idx, job in enumerate(chunk)
    ]
