"""ExternalTeacher 的两阶段解码逻辑（RTG -> Action）。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from utils.data import MotionData

from .discretization_utils import (
    decode_predicted_action,
    decode_predicted_rtg,
    get_tilt_logits,
)


def _resolve_idx_in_model(
    veh_id: int,
    veh_id_to_idx: Dict[int, int],
    new_agent_idx_dict: Dict[int, int],
) -> Optional[int]:
    agent_key = veh_id_to_idx.get(veh_id)
    if agent_key is None:
        return None
    idx_in_model = new_agent_idx_dict.get(agent_key)
    if idx_in_model is None:
        return None
    return int(idx_in_model)


def _write_rtg_discrete(
    batched_data: MotionData,
    batch_idx: int,
    idx_in_model: int,
    token_index: int,
    discrete: Tuple[int, int, int],
) -> None:
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 0] = int(discrete[0])
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 1] = int(discrete[1])
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 2] = int(discrete[2])


def _iter_resolved_vehicle_indices(
    vehicle_ids: Iterable[int],
    veh_id_to_idx: Dict[int, int],
    new_agent_idx_dict: Dict[int, int],
) -> Iterator[Tuple[int, int]]:
    for veh_id in vehicle_ids:
        idx_in_model = _resolve_idx_in_model(
            veh_id=veh_id,
            veh_id_to_idx=veh_id_to_idx,
            new_agent_idx_dict=new_agent_idx_dict,
        )
        if idx_in_model is None:
            continue
        yield int(veh_id), idx_in_model


def _decode_rtg_for_job(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    rtg_cache: Dict[Tuple[int, int], Dict[str, Any]],
) -> Tuple[Dict[int, Tuple[float, float, float]], List[int]]:
    env_idx = int(job["env_idx"])
    prepared = job["prepared"]
    focal_batch = job["focal_batch"]
    veh_id_to_idx = prepared["veh_id_to_idx"]
    tilt_by_veh_id = prepared["tilt_by_veh_id"]
    default_tilt = prepared["default_tilt"]

    new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
    data_veh_ids = set(focal_batch["data_veh_ids"])
    veh_ids_in_context = focal_batch["veh_ids_in_context"]
    if not bool(focal_batch["predict_rtgs"]):
        return {}, []

    generator = teacher._get_generator(env_idx)
    rtg_results: Dict[int, Tuple[float, float, float]] = {}
    processed_rtg_veh_ids: List[int] = []

    for veh_id, idx_in_model in _iter_resolved_vehicle_indices(
        vehicle_ids=veh_ids_in_context,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        cache_key = (env_idx, veh_id)
        if cache_key in rtg_cache:
            _write_rtg_discrete(
                batched_data=batched_data,
                batch_idx=batch_idx,
                idx_in_model=idx_in_model,
                token_index=token_index,
                discrete=rtg_cache[cache_key]["discrete"],
            )
            continue

        rtg_logits_3 = rtg_logits[batch_idx, idx_in_model, token_index].reshape(
            teacher.rtg_discretization,
            teacher.num_reward_components,
        )
        if veh_id in data_veh_ids:
            g_tilt, v_tilt, e_tilt = tilt_by_veh_id.get(veh_id, default_tilt)
        else:
            g_tilt, v_tilt, e_tilt = 0, 0, 0

        tilt_logits_np = get_tilt_logits(teacher.rtg_discretization, g_tilt, v_tilt, e_tilt)
        (g_idx_t, v_idx_t, r_idx_t), (g_val, v_val, r_val) = decode_predicted_rtg(
            rtg_logits_3,
            tilt_logits_np,
            teacher.rtg_discretization,
            teacher.min_rtg_pos,
            teacher.max_rtg_pos,
            teacher.min_rtg_veh,
            teacher.max_rtg_veh,
            teacher.min_rtg_road,
            teacher.max_rtg_road,
            device=teacher.device,
            generator=generator,
        )

        g_idx = int(g_idx_t.item())
        v_idx = int(v_idx_t.item())
        r_idx = int(r_idx_t.item())
        _write_rtg_discrete(
            batched_data=batched_data,
            batch_idx=batch_idx,
            idx_in_model=idx_in_model,
            token_index=token_index,
            discrete=(g_idx, v_idx, r_idx),
        )

        continuous_vals = (float(g_val), float(v_val), float(r_val))
        rtg_results[veh_id] = continuous_vals
        processed_rtg_veh_ids.append(veh_id)
        rtg_cache[cache_key] = {
            "discrete": (g_idx, v_idx, r_idx),
            "continuous": continuous_vals,
        }

    return rtg_results, processed_rtg_veh_ids


@torch.no_grad()
def decode_rtg_stage_batched(
    teacher: Any,
    batched_data: MotionData,
    batch_meta: Dict[str, Any],
    rtg_cache: Dict[Tuple[int, int], Dict[str, Any]],
) -> Tuple[MotionData, List[Dict[int, Tuple[float, float, float]]], List[List[int]]]:
    preds = teacher.model(batched_data, eval=True)
    rtg_logits = preds["rtg_preds"]

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
    processed_rtg_veh_ids_by_job: List[List[int]] = []

    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx].item())
        rtg_results, processed_rtg_veh_ids = _decode_rtg_for_job(
            teacher=teacher,
            batched_data=batched_data,
            rtg_logits=rtg_logits,
            batch_idx=batch_idx,
            job=job,
            token_index=token_index,
            rtg_cache=rtg_cache,
        )
        rtg_results_by_job.append(rtg_results)
        processed_rtg_veh_ids_by_job.append(processed_rtg_veh_ids)

    return batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job


def _decode_action_for_job(
    teacher: Any,
    action_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
) -> Dict[int, Tuple[float, float]]:
    env_idx = int(job["env_idx"])
    prepared = job["prepared"]
    focal_batch = job["focal_batch"]
    veh_id_to_idx = prepared["veh_id_to_idx"]
    new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
    data_veh_ids = focal_batch["data_veh_ids"]
    sampling = prepared["sampling"]
    generator = teacher._get_generator(env_idx)

    action_results: Dict[int, Tuple[float, float]] = {}
    for veh_id, idx_in_model in _iter_resolved_vehicle_indices(
        vehicle_ids=data_veh_ids,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        logits_1d = action_logits[batch_idx, idx_in_model, token_index]
        accel, steer = decode_predicted_action(
            logits_1d,
            sampling["action_temperature"],
            sampling["nucleus_sampling"],
            sampling["nucleus_threshold"],
            teacher.accel_discretization,
            teacher.steer_discretization,
            teacher.min_accel,
            teacher.max_accel,
            teacher.min_steer,
            teacher.max_steer,
            generator=generator,
        )
        action_results[veh_id] = (accel, steer)
    return action_results


@torch.no_grad()
def decode_action_stage_batched(
    teacher: Any,
    batched_data: MotionData,
    batch_meta: Dict[str, Any],
) -> List[Dict[int, Tuple[float, float]]]:
    preds = teacher.model(batched_data, eval=True)
    action_logits = preds["action_preds"]

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    action_results_by_job: List[Dict[int, Tuple[float, float]]] = []
    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx].item())
        action_results_by_job.append(
            _decode_action_for_job(
                teacher=teacher,
                action_logits=action_logits,
                batch_idx=batch_idx,
                job=job,
                token_index=token_index,
            )
        )
    return action_results_by_job


def forward_chunk_batched(teacher: Any, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunk:
        return []

    batched_data, batch_meta = teacher._collate_chunk_with_padding(chunk)
    rtg_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job = decode_rtg_stage_batched(
        teacher=teacher,
        batched_data=batched_data,
        batch_meta=batch_meta,
        rtg_cache=rtg_cache,
    )
    action_results_by_job = decode_action_stage_batched(
        teacher=teacher,
        batched_data=batched_data,
        batch_meta=batch_meta,
    )

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
