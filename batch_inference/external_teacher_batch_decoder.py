"""ExternalTeacher 的两阶段解码逻辑（RTG -> Action）。"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from utils.data import MotionData

from .discretization_utils import (
    decode_predicted_action,
    decode_predicted_rtg,
    get_tilt_logits,
)

RTGCache = Dict[Tuple[int, int], Tuple[int, int, int]]


def _elapsed_ms(start_time: float, profile_enabled: bool) -> float:
    if not profile_enabled:
        return 0.0
    return (time.perf_counter() - start_time) * 1000.0


def _get_device_rng_state(device: Any) -> torch.Tensor:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        return torch.cuda.get_rng_state(torch_device).clone()
    return torch.get_rng_state().clone()


def _get_action_reservation_logits(teacher: Any) -> torch.Tensor:
    action_dim = int(teacher.accel_discretization) * int(teacher.steer_discretization)
    cached = getattr(teacher, "_action_reservation_logits", None)
    if cached is None or cached.shape != (action_dim,) or cached.device.type != torch.device(teacher.device).type:
        cached = torch.zeros(action_dim, dtype=torch.float32, device=teacher.device)
        teacher._action_reservation_logits = cached
    return cached


def _get_decode_generator(teacher: Any) -> torch.Generator:
    generator = getattr(teacher, "_decode_generator", None)
    device = torch.device(teacher.device)
    if generator is None or generator.device != device:
        generator = torch.Generator(device=device)
        teacher._decode_generator = generator
    return generator


def _get_tilt_logits_tensor(
    teacher: Any,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
) -> torch.Tensor:
    cache = getattr(teacher, "_tilt_logits_tensor_cache", None)
    if cache is None:
        cache = {}
        teacher._tilt_logits_tensor_cache = cache

    cache_key = (int(goal_tilt), int(veh_tilt), int(road_tilt))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    cached = torch.from_numpy(
        get_tilt_logits(
            teacher.rtg_discretization,
            cache_key[0],
            cache_key[1],
            cache_key[2],
        )
    ).to(teacher.device)
    cache[cache_key] = cached
    return cached


def _reserve_action_rng_states_for_job(
    teacher: Any,
    job: Dict[str, Any],
) -> Dict[int, torch.Tensor]:
    data_veh_ids = job["focal_batch"].get("data_veh_ids", [])
    sampling = job["prepared"].get("sampling")
    if not data_veh_ids or sampling is None:
        return {}
    dummy_logits = _get_action_reservation_logits(teacher)
    reserved_states: Dict[int, torch.Tensor] = {}
    for veh_id in data_veh_ids:
        reserved_states[int(veh_id)] = _get_device_rng_state(teacher.device)
        decode_predicted_action(
            dummy_logits,
            sampling["action_temperature"],
            sampling["nucleus_sampling"],
            sampling["nucleus_threshold"],
            teacher.accel_discretization,
            teacher.steer_discretization,
            teacher.min_accel,
            teacher.max_accel,
            teacher.min_steer,
            teacher.max_steer,
        )
    return reserved_states


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
    rtg_cache: RTGCache,
) -> Tuple[Dict[int, Tuple[float, float, float]], List[int]]:
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

    rtg_results: Dict[int, Tuple[float, float, float]] = {}
    processed_rtg_veh_ids: List[int] = []

    for veh_id, idx_in_model in _iter_resolved_vehicle_indices(
        vehicle_ids=veh_ids_in_context,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        cache_key = (int(job["env_idx"]), veh_id)
        if cache_key in rtg_cache:
            cached_discrete = rtg_cache[cache_key]
            _write_rtg_discrete(
                batched_data=batched_data,
                batch_idx=batch_idx,
                idx_in_model=idx_in_model,
                token_index=token_index,
                discrete=cached_discrete,
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

        tilt_logits_np = _get_tilt_logits_tensor(
            teacher,
            g_tilt,
            v_tilt,
            e_tilt,
        )
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
        )

        g_idx = int(g_idx_t)
        v_idx = int(v_idx_t)
        r_idx = int(r_idx_t)
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
        rtg_cache[cache_key] = (g_idx, v_idx, r_idx)

    return rtg_results, processed_rtg_veh_ids


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
    rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
    processed_rtg_veh_ids_by_job: List[List[int]] = []

    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx])
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
    reserved_rng_states: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, Tuple[float, float]]:
    prepared = job["prepared"]
    focal_batch = job["focal_batch"]
    veh_id_to_idx = prepared["veh_id_to_idx"]
    new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
    data_veh_ids = focal_batch["data_veh_ids"]
    sampling = prepared["sampling"]
    decode_generator = None if reserved_rng_states is None else _get_decode_generator(teacher)

    action_results: Dict[int, Tuple[float, float]] = {}
    for veh_id, idx_in_model in _iter_resolved_vehicle_indices(
        vehicle_ids=data_veh_ids,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        if reserved_rng_states is not None:
            rng_state = reserved_rng_states.get(int(veh_id))
            if rng_state is None:
                raise ValueError(f"Missing reserved RNG state for veh_id={veh_id}.")
            decode_generator.set_state(rng_state)
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
            generator=decode_generator,
        )
        action_results[veh_id] = (accel, steer)
    return action_results


@torch.no_grad()
def decode_action_stage_batched(
    teacher: Any,
    batched_data: MotionData,
    batch_meta: Dict[str, Any],
    reserved_rng_states_by_job: Optional[List[Dict[int, torch.Tensor]]] = None,
) -> List[Dict[int, Tuple[float, float]]]:
    profile_enabled = bool(getattr(teacher, "_profile_enabled", False))
    model_forward_start = time.perf_counter() if profile_enabled else 0.0
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    model_action_ms = _elapsed_ms(model_forward_start, profile_enabled)
    action_logits = preds["action_preds"].float()

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    action_results_by_job: List[Dict[int, Tuple[float, float]]] = []
    action_decode_start = time.perf_counter() if profile_enabled else 0.0
    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx])
        reserved_rng_states = (
            None
            if reserved_rng_states_by_job is None
            else reserved_rng_states_by_job[batch_idx]
        )
        action_results_by_job.append(
            _decode_action_for_job(
                teacher=teacher,
                action_logits=action_logits,
                batch_idx=batch_idx,
                job=job,
                token_index=token_index,
                reserved_rng_states=reserved_rng_states,
            )
        )

    action_decode_ms = _elapsed_ms(action_decode_start, profile_enabled)
    teacher._last_action_stage_profile = {
        "stage_ms": {
            "model_action": model_action_ms,
            "action_decode": action_decode_ms,
        },
        "detail_ms": {
            "restore_rng": 0.0,
        },
    }
    return action_results_by_job


def forward_chunk_batched(teacher: Any, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunk:
        return []

    profile_enabled = bool(getattr(teacher, "_profile_enabled", False))
    total_start = time.perf_counter() if profile_enabled else 0.0

    collate_start = time.perf_counter() if profile_enabled else 0.0
    batched_data, batch_meta = teacher._collate_chunk_with_padding(chunk)
    collate_ms = _elapsed_ms(collate_start, profile_enabled)

    model_rtg_start = time.perf_counter() if profile_enabled else 0.0
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    model_rtg_ms = _elapsed_ms(model_rtg_start, profile_enabled)
    rtg_logits = preds["rtg_preds"].float()

    rtg_cache: RTGCache = {}
    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
    processed_rtg_veh_ids_by_job: List[List[int]] = []
    reserved_action_rng_states_by_job: List[Dict[int, torch.Tensor]] = []
    rtg_decode_ms = 0.0
    rng_reserve_ms = 0.0

    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx])
        rtg_decode_start = time.perf_counter() if profile_enabled else 0.0
        rtg_results, processed_rtg_veh_ids = _decode_rtg_for_job(
            teacher=teacher,
            batched_data=batched_data,
            rtg_logits=rtg_logits,
            batch_idx=batch_idx,
            job=job,
            token_index=token_index,
            rtg_cache=rtg_cache,
        )
        rtg_decode_ms += _elapsed_ms(rtg_decode_start, profile_enabled)
        rtg_results_by_job.append(rtg_results)
        processed_rtg_veh_ids_by_job.append(processed_rtg_veh_ids)
        rng_reserve_start = time.perf_counter() if profile_enabled else 0.0
        reserved_action_rng_states_by_job.append(
            _reserve_action_rng_states_for_job(
                teacher=teacher,
                job=job,
            )
        )
        rng_reserve_ms += _elapsed_ms(rng_reserve_start, profile_enabled)

    final_rng_state_start = time.perf_counter() if profile_enabled else 0.0
    _ = _get_device_rng_state(teacher.device)
    rng_reserve_ms += _elapsed_ms(final_rng_state_start, profile_enabled)

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
            "total": _elapsed_ms(total_start, profile_enabled),
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
