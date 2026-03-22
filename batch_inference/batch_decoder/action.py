"""
负责动作 logits 的批量解码与按车辆组织的结果回填。
该模块将 action 头输出转换成连续动作结果，并使用无状态 sampling tickets 保持可复现。
Handles batched action-logit decoding and per-vehicle result assembly.
Converts action-head outputs into continuous actions while using stateless sampling tickets for reproducibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ctrlsim_adapter.opponent_vehicle.discretization import undiscretize_action_index

from .rtg import iter_resolved_vehicle_indices
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
) -> List[Dict[int, Tuple[float, float]]]:
    job_offsets = decode_meta["job_offsets"]
    if decode_meta["job_idx"].size == 0:
        return [{} for _ in range(int(decode_meta["job_count"][0]))]

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

    action_results_by_job: List[Dict[int, Tuple[float, float]]] = []
    for start, end in zip(job_offsets[:-1], job_offsets[1:]):
        per_job_results: Dict[int, Tuple[float, float]] = {}
        for row_idx in range(int(start), int(end)):
            veh_id = int(decode_meta["veh_id"][row_idx])
            per_job_results[veh_id] = (
                float(accel_t[row_idx]),
                float(steer_t[row_idx]),
            )
        action_results_by_job.append(per_job_results)

    if len(action_results_by_job) != int(decode_meta["job_count"][0]):
        raise ValueError("Action decode job/result count mismatch.")
    return action_results_by_job


def decode_action_for_job_impl(
    teacher: Any,
    action_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
) -> Dict[int, Tuple[float, float]]:
    prepared = job["prepared"]
    focal_batch = job["focal_batch"]
    veh_id_to_idx = prepared["veh_id_to_idx"]
    new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
    data_veh_ids = focal_batch["data_veh_ids"]
    sampling = prepared["sampling"]
    sampling_seed = np.asarray([prepared["sampling_seed"]], dtype=np.uint64)
    temperature = torch.as_tensor([float(sampling["action_temperature"])], dtype=action_logits.dtype, device=action_logits.device)
    nucleus_sampling = torch.as_tensor([bool(sampling["nucleus_sampling"])], dtype=torch.bool, device=action_logits.device)
    nucleus_threshold = torch.as_tensor([float(sampling["nucleus_threshold"])], dtype=action_logits.dtype, device=action_logits.device)

    action_results: Dict[int, Tuple[float, float]] = {}
    for veh_id, idx_in_model in iter_resolved_vehicle_indices(
        vehicle_ids=data_veh_ids,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        logits_1d = action_logits[batch_idx, idx_in_model, token_index]
        action_idx = _sample_action_indices(
            flat_logits=logits_1d.unsqueeze(0),
            temperatures=temperature,
            nucleus_sampling=nucleus_sampling,
            nucleus_thresholds=nucleus_threshold,
            sampling_seed=sampling_seed,
            step_t=np.asarray([int(prepared["step_t"])], dtype=np.int64),
            veh_id=np.asarray([int(veh_id)], dtype=np.int64),
        )
        accel, steer = undiscretize_action_index(
            int(action_idx[0]),
            teacher.accel_discretization,
            teacher.steer_discretization,
            teacher.min_accel,
            teacher.max_accel,
            teacher.min_steer,
            teacher.max_steer,
        )
        action_results[veh_id] = (float(accel), float(steer))
    return action_results


@torch.no_grad()
def decode_action_stage_batched_impl(
    teacher: Any,
    batched_data: Any,
    batch_meta: Dict[str, Any],
) -> List[Dict[int, Tuple[float, float]]]:
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    action_logits = preds["action_preds"].float()

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    action_results_by_job = decode_action_jobs_batched_impl(
        teacher=teacher,
        action_logits=action_logits,
        decode_meta=batch_meta["decode_meta"]["action"],
    )
    if len(action_results_by_job) != len(jobs):
        raise ValueError("Action stage job/result count mismatch.")
    return action_results_by_job
