"""ExternalTeacher 的两阶段解码入口（RTG -> Action）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from adapters.ctrlsim_discretization import decode_predicted_action, decode_predicted_rtg, get_tilt_logits
from utils.data import MotionData

from . import action as action_impl
from .action import decode_action_stage_batched_impl
from .forward_chunk import forward_chunk_batched_impl
from .profile import chunk_predict_rtgs_mode as _chunk_predict_rtgs_mode
from .profile import elapsed_ms as _elapsed_ms
from .rng import (
    get_device_rng_state as _get_device_rng_state,
    get_env_sampling_generator as _get_env_sampling_generator,
    get_next_worker_rng_state,
    reserve_action_rng_states_for_job as _reserve_action_rng_states_for_job,
)
from . import rtg as rtg_impl
from .rtg import RTGCache


def _get_tilt_logits_tensor(
    teacher: Any,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
) -> torch.Tensor:
    return rtg_impl.get_tilt_logits_tensor_impl(
        teacher=teacher,
        goal_tilt=goal_tilt,
        veh_tilt=veh_tilt,
        road_tilt=road_tilt,
        get_tilt_logits_fn=get_tilt_logits,
    )


def _decode_rtg_for_job(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    rtg_cache: RTGCache,
    env_generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[int, Tuple[float, float, float]], List[int]]:
    return rtg_impl.decode_rtg_for_job_impl(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        batch_idx=batch_idx,
        job=job,
        token_index=token_index,
        rtg_cache=rtg_cache,
        get_tilt_logits_tensor_fn=_get_tilt_logits_tensor,
        decode_predicted_rtg_fn=decode_predicted_rtg,
        iter_resolved_vehicle_indices_fn=rtg_impl.iter_resolved_vehicle_indices,
        write_rtg_discrete_fn=rtg_impl.write_rtg_discrete,
        env_generator=env_generator,
    )


def _decode_rtg_jobs_batched(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    rtg_cache: RTGCache,
) -> Tuple[List[Dict[int, Tuple[float, float, float]]], List[List[int]]]:
    return rtg_impl.decode_rtg_jobs_batched_impl(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        jobs=jobs,
        token_index_per_job=token_index_per_job,
        rtg_cache=rtg_cache,
        get_env_sampling_generator_fn=_get_env_sampling_generator,
        get_tilt_logits_tensor_fn=_get_tilt_logits_tensor,
        decode_predicted_rtg_fn=decode_predicted_rtg,
        iter_resolved_vehicle_indices_fn=rtg_impl.iter_resolved_vehicle_indices,
    )


def _decode_action_for_job(
    teacher: Any,
    action_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    reserved_rng_states: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, Tuple[float, float]]:
    return action_impl.decode_action_for_job_impl(
        teacher=teacher,
        action_logits=action_logits,
        batch_idx=batch_idx,
        job=job,
        token_index=token_index,
        decode_predicted_action_fn=decode_predicted_action,
        get_decode_generator_fn=action_impl.get_decode_generator,
        iter_resolved_vehicle_indices_fn=action_impl.iter_resolved_vehicle_indices,
        reserved_rng_states=reserved_rng_states,
    )


def _decode_action_jobs_batched(
    teacher: Any,
    action_logits: torch.Tensor,
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    reserved_rng_states_by_job: Optional[List[Dict[int, torch.Tensor]]] = None,
) -> List[Dict[int, Tuple[float, float]]]:
    return action_impl.decode_action_jobs_batched_impl(
        teacher=teacher,
        action_logits=action_logits,
        jobs=jobs,
        token_index_per_job=token_index_per_job,
        decode_predicted_action_fn=decode_predicted_action,
        get_decode_generator_fn=action_impl.get_decode_generator,
        iter_resolved_vehicle_indices_fn=action_impl.iter_resolved_vehicle_indices,
        reserved_rng_states_by_job=reserved_rng_states_by_job,
    )


_DEFAULT_GET_TILT_LOGITS_TENSOR = _get_tilt_logits_tensor
_DEFAULT_DECODE_RTG_FOR_JOB = _decode_rtg_for_job
_DEFAULT_DECODE_ACTION_FOR_JOB = _decode_action_for_job
_DEFAULT_DECODE_PREDICTED_RTG = decode_predicted_rtg


@torch.no_grad()
def decode_rtg_stage_batched(
    teacher: Any,
    batched_data: MotionData,
    batch_meta: Dict[str, Any],
    rtg_cache: RTGCache,
) -> Tuple[MotionData, List[Dict[int, Tuple[float, float, float]]], List[List[int]]]:
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    rtg_logits = preds["rtg_preds"].float()

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    use_legacy_path = (
        _get_tilt_logits_tensor is not _DEFAULT_GET_TILT_LOGITS_TENSOR
        or _decode_rtg_for_job is not _DEFAULT_DECODE_RTG_FOR_JOB
        or decode_predicted_rtg is not _DEFAULT_DECODE_PREDICTED_RTG
    )
    if use_legacy_path:
        rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
        processed_rtg_veh_ids_by_job: List[List[int]] = []
        for batch_idx, job in enumerate(jobs):
            token_index = int(token_index_per_job[batch_idx])
            env_generator = _get_env_sampling_generator(
                teacher=teacher,
                env_idx=int(job["env_idx"]),
                step_t=int(job["prepared"]["step_t"]),
                worker_rng_state=job["prepared"].get("worker_rng_state"),
            )
            rtg_results, processed_rtg_veh_ids = _decode_rtg_for_job(
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
        return batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job

    rtg_results_by_job, processed_rtg_veh_ids_by_job = rtg_impl.decode_rtg_jobs_batched_impl(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        jobs=jobs,
        token_index_per_job=token_index_per_job,
        rtg_cache=rtg_cache,
        get_env_sampling_generator_fn=_get_env_sampling_generator,
        get_tilt_logits_tensor_fn=_get_tilt_logits_tensor,
        decode_predicted_rtg_fn=decode_predicted_rtg,
        iter_resolved_vehicle_indices_fn=rtg_impl.iter_resolved_vehicle_indices,
    )

    return batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job


@torch.no_grad()
def decode_action_stage_batched(
    teacher: Any,
    batched_data: MotionData,
    batch_meta: Dict[str, Any],
    reserved_rng_states_by_job: Optional[List[Dict[int, torch.Tensor]]] = None,
) -> List[Dict[int, Tuple[float, float]]]:
    return decode_action_stage_batched_impl(
        teacher=teacher,
        batched_data=batched_data,
        batch_meta=batch_meta,
        decode_action_for_job_fn=_decode_action_for_job,
        decode_action_jobs_batched_fn=(
            _decode_action_jobs_batched
            if _decode_action_for_job is _DEFAULT_DECODE_ACTION_FOR_JOB
            else None
        ),
        reserved_rng_states_by_job=reserved_rng_states_by_job,
    )


def forward_chunk_batched(teacher: Any, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return forward_chunk_batched_impl(
        teacher=teacher,
        chunk=chunk,
        chunk_predict_rtgs_mode_fn=_chunk_predict_rtgs_mode,
        elapsed_ms_fn=_elapsed_ms,
        get_env_sampling_generator_fn=_get_env_sampling_generator,
        decode_rtg_for_job_fn=_decode_rtg_for_job,
        reserve_action_rng_states_for_job_fn=_reserve_action_rng_states_for_job,
        decode_rtg_jobs_batched_fn=(
            _decode_rtg_jobs_batched
            if _decode_rtg_for_job is _DEFAULT_DECODE_RTG_FOR_JOB
            else None
        ),
    )


__all__ = [
    "RTGCache",
    "decode_action_stage_batched",
    "decode_rtg_stage_batched",
    "forward_chunk_batched",
    "get_next_worker_rng_state",
    "_decode_action_for_job",
    "_decode_rtg_for_job",
    "_get_env_sampling_generator",
    "_get_tilt_logits_tensor",
    "_get_device_rng_state",
    "decode_predicted_action",
    "decode_predicted_rtg",
]
