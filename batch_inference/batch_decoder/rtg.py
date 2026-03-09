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


def _build_rtg_decode_rows(
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    iter_resolved_vehicle_indices_fn: Any,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    job_offsets = [0]
    for batch_idx, job in enumerate(jobs):
        focal_batch = job["focal_batch"]
        if not bool(focal_batch["predict_rtgs"]):
            job_offsets.append(len(rows))
            continue

        prepared = job["prepared"]
        veh_id_to_idx = prepared["veh_id_to_idx"]
        new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
        data_veh_ids = set(focal_batch["data_veh_ids"])
        veh_ids_in_context = focal_batch["veh_ids_in_context"]
        token_index = int(token_index_per_job[batch_idx])

        for veh_id, idx_in_model in iter_resolved_vehicle_indices_fn(
            vehicle_ids=veh_ids_in_context,
            veh_id_to_idx=veh_id_to_idx,
            new_agent_idx_dict=new_agent_idx_dict,
        ):
            rows.append(
                {
                    "job_idx": batch_idx,
                    "env_idx": int(job["env_idx"]),
                    "veh_id": int(veh_id),
                    "idx_in_model": int(idx_in_model),
                    "token_index": token_index,
                    "step_t": int(prepared["step_t"]),
                    "worker_rng_state": prepared.get("worker_rng_state"),
                    "goal_tilt": int(prepared["tilt_by_veh_id"].get(veh_id, prepared["default_tilt"])[0])
                    if veh_id in data_veh_ids
                    else 0,
                    "veh_tilt": int(prepared["tilt_by_veh_id"].get(veh_id, prepared["default_tilt"])[1])
                    if veh_id in data_veh_ids
                    else 0,
                    "road_tilt": int(prepared["tilt_by_veh_id"].get(veh_id, prepared["default_tilt"])[2])
                    if veh_id in data_veh_ids
                    else 0,
                }
            )
        job_offsets.append(len(rows))
    return rows, job_offsets


def decode_rtg_jobs_batched_impl(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    rtg_cache: RTGCache,
    get_env_sampling_generator_fn: Any,
    get_tilt_logits_tensor_fn: Any,
    decode_predicted_rtg_fn: Any,
    iter_resolved_vehicle_indices_fn: Any,
    write_rtg_discrete_fn: Any = None,
) -> Tuple[List[Dict[int, Tuple[float, float, float]]], List[List[int]]]:
    del write_rtg_discrete_fn

    rows, job_offsets = _build_rtg_decode_rows(
        jobs=jobs,
        token_index_per_job=token_index_per_job,
        iter_resolved_vehicle_indices_fn=iter_resolved_vehicle_indices_fn,
    )
    if not rows:
        return [{} for _ in jobs], [[] for _ in jobs]

    device = rtg_logits.device
    cached_discrete_by_row: List[Optional[Tuple[int, int, int]]] = [None] * len(rows)
    alias_source_by_row: Dict[int, int] = {}
    miss_row_indices: List[int] = []
    miss_rows: List[Dict[str, Any]] = []
    source_row_by_cache_key: Dict[Tuple[int, int], int] = {}
    for row_idx, row in enumerate(rows):
        cache_key = (row["env_idx"], row["veh_id"])
        discrete = rtg_cache.get(cache_key)
        if discrete is not None:
            cached_discrete_by_row[row_idx] = discrete
            continue
        source_row_idx = source_row_by_cache_key.get(cache_key)
        if source_row_idx is not None:
            alias_source_by_row[row_idx] = source_row_idx
            continue
        source_row_by_cache_key[cache_key] = row_idx
        miss_row_indices.append(row_idx)
        miss_rows.append(row)

    sampled_discrete_by_row: List[Optional[Tuple[int, int, int]]] = [None] * len(rows)
    sampled_continuous_by_row: List[Optional[Tuple[float, float, float]]] = [None] * len(rows)
    if miss_rows:
        env_generator_by_job: List[Optional[torch.Generator]] = [None] * len(jobs)
        for batch_idx, job in enumerate(jobs):
            focal_batch = job["focal_batch"]
            if not bool(focal_batch["predict_rtgs"]):
                continue
            env_generator_by_job[batch_idx] = get_env_sampling_generator_fn(
                teacher=teacher,
                env_idx=int(job["env_idx"]),
                step_t=int(job["prepared"]["step_t"]),
                worker_rng_state=job["prepared"].get("worker_rng_state"),
            )
        miss_batch_idx = torch.as_tensor(
            [row["job_idx"] for row in miss_rows],
            dtype=torch.long,
            device=device,
        )
        miss_idx_in_model = torch.as_tensor(
            [row["idx_in_model"] for row in miss_rows],
            dtype=torch.long,
            device=device,
        )
        miss_token_index = torch.as_tensor(
            [row["token_index"] for row in miss_rows],
            dtype=torch.long,
            device=device,
        )
        flat_rtg_logits = rtg_logits[miss_batch_idx, miss_idx_in_model, miss_token_index].reshape(
            -1,
            teacher.rtg_discretization,
            teacher.num_reward_components,
        )
        flat_tilt_logits = torch.stack(
            [
                get_tilt_logits_tensor_fn(
                    teacher,
                    row["goal_tilt"],
                    row["veh_tilt"],
                    row["road_tilt"],
                )
                for row in miss_rows
            ],
            dim=0,
        )

        for miss_idx, row in enumerate(miss_rows):
            env_generator = env_generator_by_job[row["job_idx"]]
            discrete, continuous = decode_predicted_rtg_fn(
                flat_rtg_logits[miss_idx],
                flat_tilt_logits[miss_idx],
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
            row_idx = miss_row_indices[miss_idx]
            discrete = tuple(int(v) for v in discrete)
            continuous = tuple(float(v) for v in continuous)
            sampled_discrete_by_row[row_idx] = discrete
            sampled_continuous_by_row[row_idx] = continuous
            rtg_cache[(rows[row_idx]["env_idx"], rows[row_idx]["veh_id"])] = discrete

    write_batch_idx: List[int] = []
    write_idx_in_model: List[int] = []
    write_token_index: List[int] = []
    write_discrete: List[Tuple[int, int, int]] = []
    for row_idx, row in enumerate(rows):
        discrete = cached_discrete_by_row[row_idx]
        if discrete is None:
            source_row_idx = alias_source_by_row.get(row_idx, row_idx)
            discrete = sampled_discrete_by_row[source_row_idx]
            if discrete is None:
                raise ValueError("Missing sampled RTG discrete result for source row.")
        write_batch_idx.append(row["job_idx"])
        write_idx_in_model.append(row["idx_in_model"])
        write_token_index.append(row["token_index"])
        write_discrete.append(discrete)

    write_batch_idx_t = torch.as_tensor(write_batch_idx, dtype=torch.long, device=device)
    write_idx_in_model_t = torch.as_tensor(write_idx_in_model, dtype=torch.long, device=device)
    write_token_index_t = torch.as_tensor(write_token_index, dtype=torch.long, device=device)
    write_discrete_t = torch.as_tensor(write_discrete, dtype=batched_data["agent"].rtgs.dtype, device=device)
    batched_data["agent"].rtgs[write_batch_idx_t, write_idx_in_model_t, write_token_index_t, 0] = write_discrete_t[:, 0]
    batched_data["agent"].rtgs[write_batch_idx_t, write_idx_in_model_t, write_token_index_t, 1] = write_discrete_t[:, 1]
    batched_data["agent"].rtgs[write_batch_idx_t, write_idx_in_model_t, write_token_index_t, 2] = write_discrete_t[:, 2]

    rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
    processed_rtg_veh_ids_by_job: List[List[int]] = []
    for start, end in zip(job_offsets[:-1], job_offsets[1:]):
        per_job_results: Dict[int, Tuple[float, float, float]] = {}
        per_job_processed: List[int] = []
        for row_idx in range(start, end):
            continuous = sampled_continuous_by_row[row_idx]
            if continuous is None:
                continue
            row = rows[row_idx]
            per_job_results[row["veh_id"]] = continuous
            per_job_processed.append(row["veh_id"])
        rtg_results_by_job.append(per_job_results)
        processed_rtg_veh_ids_by_job.append(per_job_processed)

    if len(rtg_results_by_job) != len(jobs):
        raise ValueError("RTG decode job/result count mismatch.")
    return rtg_results_by_job, processed_rtg_veh_ids_by_job


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
