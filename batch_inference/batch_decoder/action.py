from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .profile import elapsed_ms
from .rng import get_decode_generator
from .rtg import iter_resolved_vehicle_indices


def _build_action_decode_rows(
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    iter_resolved_vehicle_indices_fn: Any,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    job_offsets = [0]
    for batch_idx, job in enumerate(jobs):
        prepared = job["prepared"]
        focal_batch = job["focal_batch"]
        veh_id_to_idx = prepared["veh_id_to_idx"]
        new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
        data_veh_ids = focal_batch["data_veh_ids"]
        sampling = prepared["sampling"]
        token_index = int(token_index_per_job[batch_idx])

        for veh_id, idx_in_model in iter_resolved_vehicle_indices_fn(
            vehicle_ids=data_veh_ids,
            veh_id_to_idx=veh_id_to_idx,
            new_agent_idx_dict=new_agent_idx_dict,
        ):
            rows.append(
                {
                    "job_idx": batch_idx,
                    "veh_id": int(veh_id),
                    "idx_in_model": int(idx_in_model),
                    "token_index": token_index,
                    "action_temperature": float(sampling["action_temperature"]),
                    "nucleus_sampling": bool(sampling["nucleus_sampling"]),
                    "nucleus_threshold": float(sampling["nucleus_threshold"]),
                }
            )
        job_offsets.append(len(rows))
    return rows, job_offsets


def decode_action_jobs_batched_impl(
    teacher: Any,
    action_logits: torch.Tensor,
    jobs: List[Dict[str, Any]],
    token_index_per_job: torch.Tensor,
    decode_predicted_action_fn: Any,
    get_decode_generator_fn: Any,
    iter_resolved_vehicle_indices_fn: Any,
    reserved_rng_states_by_job: Optional[List[Dict[int, torch.Tensor]]] = None,
) -> List[Dict[int, Tuple[float, float]]]:
    rows, job_offsets = _build_action_decode_rows(
        jobs=jobs,
        token_index_per_job=token_index_per_job,
        iter_resolved_vehicle_indices_fn=iter_resolved_vehicle_indices_fn,
    )
    if not rows:
        return [{} for _ in jobs]

    device = action_logits.device
    row_batch_idx = torch.as_tensor([row["job_idx"] for row in rows], dtype=torch.long, device=device)
    row_idx_in_model = torch.as_tensor([row["idx_in_model"] for row in rows], dtype=torch.long, device=device)
    row_token_index = torch.as_tensor([row["token_index"] for row in rows], dtype=torch.long, device=device)
    flat_logits = action_logits[row_batch_idx, row_idx_in_model, row_token_index]

    decode_generator = None
    if reserved_rng_states_by_job is not None:
        decode_generator = get_decode_generator_fn(teacher)

    decoded_actions: List[Tuple[float, float]] = []
    for row_idx, row in enumerate(rows):
        generator = None
        if decode_generator is not None:
            reserved_states = reserved_rng_states_by_job[row["job_idx"]]
            rng_state = reserved_states.get(row["veh_id"])
            if rng_state is None:
                raise ValueError(f"Missing reserved RNG state for veh_id={row['veh_id']}.")
            decode_generator.set_state(rng_state)
            generator = decode_generator
        decoded_actions.append(
            decode_predicted_action_fn(
                flat_logits[row_idx],
                row["action_temperature"],
                row["nucleus_sampling"],
                row["nucleus_threshold"],
                teacher.accel_discretization,
                teacher.steer_discretization,
                teacher.min_accel,
                teacher.max_accel,
                teacher.min_steer,
                teacher.max_steer,
                generator=generator,
            )
        )

    action_results_by_job: List[Dict[int, Tuple[float, float]]] = []
    for start, end in zip(job_offsets[:-1], job_offsets[1:]):
        per_job_results: Dict[int, Tuple[float, float]] = {}
        for row, (accel, steer) in zip(rows[start:end], decoded_actions[start:end]):
            per_job_results[row["veh_id"]] = (float(accel), float(steer))
        action_results_by_job.append(per_job_results)

    if len(action_results_by_job) != len(jobs):
        raise ValueError("Action decode job/result count mismatch.")
    return action_results_by_job


def decode_action_for_job_impl(
    teacher: Any,
    action_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    decode_predicted_action_fn: Any,
    get_decode_generator_fn: Any,
    iter_resolved_vehicle_indices_fn: Any,
    reserved_rng_states: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, Tuple[float, float]]:
    prepared = job["prepared"]
    focal_batch = job["focal_batch"]
    veh_id_to_idx = prepared["veh_id_to_idx"]
    new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
    data_veh_ids = focal_batch["data_veh_ids"]
    sampling = prepared["sampling"]
    decode_generator = None if reserved_rng_states is None else get_decode_generator_fn(teacher)

    action_results: Dict[int, Tuple[float, float]] = {}
    for veh_id, idx_in_model in iter_resolved_vehicle_indices_fn(
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
        accel, steer = decode_predicted_action_fn(
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
def decode_action_stage_batched_impl(
    teacher: Any,
    batched_data: Any,
    batch_meta: Dict[str, Any],
    decode_action_for_job_fn: Any,
    reserved_rng_states_by_job: Optional[List[Dict[int, torch.Tensor]]] = None,
) -> List[Dict[int, Tuple[float, float]]]:
    profile_enabled = bool(getattr(teacher, "_profile_enabled", False))
    model_forward_start = time.perf_counter() if profile_enabled else 0.0
    with teacher.model_forward_context():
        preds = teacher.model(batched_data, eval=True)
    model_action_ms = elapsed_ms(model_forward_start, profile_enabled)
    action_logits = preds["action_preds"].float()

    jobs: List[Dict[str, Any]] = batch_meta["jobs"]
    token_index_per_job: torch.Tensor = batch_meta["token_index_per_job"]
    action_results_by_job: List[Dict[int, Tuple[float, float]]] = []
    action_decode_start = time.perf_counter() if profile_enabled else 0.0
    for batch_idx, job in enumerate(jobs):
        token_index = int(token_index_per_job[batch_idx])
        reserved_rng_states = None if reserved_rng_states_by_job is None else reserved_rng_states_by_job[batch_idx]
        action_results_by_job.append(
            decode_action_for_job_fn(
                teacher=teacher,
                action_logits=action_logits,
                batch_idx=batch_idx,
                job=job,
                token_index=token_index,
                reserved_rng_states=reserved_rng_states,
            )
        )

    action_decode_ms = elapsed_ms(action_decode_start, profile_enabled)
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


def decode_action_for_job(
    teacher: Any,
    action_logits: torch.Tensor,
    batch_idx: int,
    job: Dict[str, Any],
    token_index: int,
    reserved_rng_states: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, Tuple[float, float]]:
    from adapters.ctrlsim_discretization import decode_predicted_action

    return decode_action_for_job_impl(
        teacher=teacher,
        action_logits=action_logits,
        batch_idx=batch_idx,
        job=job,
        token_index=token_index,
        decode_predicted_action_fn=decode_predicted_action,
        get_decode_generator_fn=get_decode_generator,
        iter_resolved_vehicle_indices_fn=iter_resolved_vehicle_indices,
        reserved_rng_states=reserved_rng_states,
    )
