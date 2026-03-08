from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from utils.data import MotionData

from ..discretization_utils import decode_predicted_rtg, get_tilt_logits

RTGCache = Dict[Tuple[int, int], Tuple[int, int, int]]


def get_tilt_logits_tensor_impl(
    teacher: Any,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
    get_tilt_logits_fn: Any,
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
        get_tilt_logits_fn(
            teacher.rtg_discretization,
            cache_key[0],
            cache_key[1],
            cache_key[2],
        )
    ).to(teacher.device)
    cache[cache_key] = cached
    return cached


def get_tilt_logits_tensor(
    teacher: Any,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
) -> torch.Tensor:
    return get_tilt_logits_tensor_impl(
        teacher=teacher,
        goal_tilt=goal_tilt,
        veh_tilt=veh_tilt,
        road_tilt=road_tilt,
        get_tilt_logits_fn=get_tilt_logits,
    )


def resolve_idx_in_model(
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


def write_rtg_discrete(
    batched_data: MotionData,
    batch_idx: int,
    idx_in_model: int,
    token_index: int,
    discrete: Tuple[int, int, int],
) -> None:
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 0] = int(discrete[0])
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 1] = int(discrete[1])
    batched_data["agent"].rtgs[batch_idx, idx_in_model, token_index, 2] = int(discrete[2])


def iter_resolved_vehicle_indices(
    vehicle_ids: Iterable[int],
    veh_id_to_idx: Dict[int, int],
    new_agent_idx_dict: Dict[int, int],
) -> Iterator[Tuple[int, int]]:
    for veh_id in vehicle_ids:
        idx_in_model = resolve_idx_in_model(
            veh_id=veh_id,
            veh_id_to_idx=veh_id_to_idx,
            new_agent_idx_dict=new_agent_idx_dict,
        )
        if idx_in_model is None:
            continue
        yield int(veh_id), idx_in_model


def decode_rtg_for_job_impl(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    rtg_cache: RTGCache,
    get_tilt_logits_tensor_fn: Any,
    decode_predicted_rtg_fn: Any,
    iter_resolved_vehicle_indices_fn: Any,
    write_rtg_discrete_fn: Any,
    env_generator: Optional[torch.Generator] = None,
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
    for veh_id, idx_in_model in iter_resolved_vehicle_indices_fn(
        vehicle_ids=veh_ids_in_context,
        veh_id_to_idx=veh_id_to_idx,
        new_agent_idx_dict=new_agent_idx_dict,
    ):
        cache_key = (int(job["env_idx"]), veh_id)
        if cache_key in rtg_cache:
            write_rtg_discrete_fn(
                batched_data=batched_data,
                batch_idx=batch_idx,
                idx_in_model=idx_in_model,
                token_index=token_index,
                discrete=rtg_cache[cache_key],
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

        tilt_logits_np = get_tilt_logits_tensor_fn(
            teacher,
            g_tilt,
            v_tilt,
            e_tilt,
        )
        (g_idx_t, v_idx_t, r_idx_t), (g_val, v_val, r_val) = decode_predicted_rtg_fn(
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
            generator=env_generator,
        )

        discrete = (int(g_idx_t), int(v_idx_t), int(r_idx_t))
        write_rtg_discrete_fn(
            batched_data=batched_data,
            batch_idx=batch_idx,
            idx_in_model=idx_in_model,
            token_index=token_index,
            discrete=discrete,
        )
        rtg_results[veh_id] = (float(g_val), float(v_val), float(r_val))
        processed_rtg_veh_ids.append(veh_id)
        rtg_cache[cache_key] = discrete

    return rtg_results, processed_rtg_veh_ids


def decode_rtg_for_job(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    rtg_cache: RTGCache,
    env_generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[int, Tuple[float, float, float]], List[int]]:
    return decode_rtg_for_job_impl(
        teacher=teacher,
        batched_data=batched_data,
        rtg_logits=rtg_logits,
        batch_idx=batch_idx,
        job=job,
        token_index=token_index,
        rtg_cache=rtg_cache,
        get_tilt_logits_tensor_fn=get_tilt_logits_tensor,
        decode_predicted_rtg_fn=decode_predicted_rtg,
        iter_resolved_vehicle_indices_fn=iter_resolved_vehicle_indices,
        write_rtg_discrete_fn=write_rtg_discrete,
        env_generator=env_generator,
    )
