"""
负责将多个 focal job 整理成统一的批量输入，并完成 padding 与缓冲区复用。
该模块将离散的 numpy 字段转换成 MotionData，供 ExternalTeacher 的批量前向阶段直接消费。
Builds uniform batched inputs from focal jobs, including padding and reusable buffer management.
Converts scattered numpy fields into MotionData for ExternalTeacher's batched forward stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.data import MotionData, from_numpy


def get_or_create_collate_buffers(
    collate_numpy_buffers: Dict[Tuple[Any, ...], Dict[str, np.ndarray]],
    cache_key: Tuple[Any, ...],
    specs: Dict[str, Tuple[Tuple[int, ...], np.dtype, Any]],
) -> Dict[str, np.ndarray]:
    buffers = collate_numpy_buffers.get(cache_key)
    if buffers is None:
        buffers = {}
        for name, (shape, dtype, fill_value) in specs.items():
            arr = np.zeros(shape, dtype=dtype)
            if fill_value != 0:
                arr.fill(fill_value)
            buffers[name] = arr
        collate_numpy_buffers[cache_key] = buffers
        return buffers

    for name, (_, _, fill_value) in specs.items():
        buffers[name].fill(fill_value)
    return buffers


def infer_job_batch_layout(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_motion = jobs[0]["focal_batch"]["motion_data_np"]
    max_agents = 1
    max_seq_len = 1
    max_roads = 1
    max_road_pts = 1
    max_timestep_feat_dim = 1
    for job in jobs:
        motion_data_np = job["focal_batch"]["motion_data_np"]
        max_agents = max(max_agents, int(motion_data_np["agent_states"].shape[0]))
        max_seq_len = max(max_seq_len, int(motion_data_np["agent_states"].shape[1]))
        max_roads = max(max_roads, int(motion_data_np["road_points"].shape[0]))
        max_road_pts = max(max_road_pts, int(motion_data_np["road_points"].shape[1]))
        max_timestep_feat_dim = max(max_timestep_feat_dim, int(motion_data_np["timesteps"].shape[2]))

    return {
        "batch_size": len(jobs),
        "max_agents": max_agents,
        "max_seq_len": max_seq_len,
        "max_roads": max_roads,
        "max_road_pts": max_road_pts,
        "max_timestep_feat_dim": max_timestep_feat_dim,
        "agent_feat_dim": int(first_motion["agent_states"].shape[2]),
        "road_pts_dim": int(first_motion["road_points"].shape[2]),
        "road_type_dim": int(first_motion["road_types"].shape[1]),
        "agent_type_dim": int(first_motion["agent_types"].shape[1]),
        "goal_dim": int(first_motion["goals"].shape[1]),
        "rtg_dim": int(first_motion["rtgs"].shape[2]),
        "dtypes": {
            "agent_states": np.dtype(first_motion["agent_states"].dtype),
            "agent_types": np.dtype(first_motion["agent_types"].dtype),
            "goals": np.dtype(first_motion["goals"].dtype),
            "actions": np.dtype(first_motion["actions"].dtype),
            "rtgs": np.dtype(first_motion["rtgs"].dtype),
            "timesteps": np.dtype(first_motion["timesteps"].dtype),
            "moving_agent_mask": np.dtype(first_motion["moving_agent_mask"].dtype),
            "road_points": np.dtype(first_motion["road_points"].dtype),
            "road_types": np.dtype(first_motion["road_types"].dtype),
        },
    }


def build_collate_specs(layout: Dict[str, Any]) -> Dict[str, Tuple[Tuple[int, ...], np.dtype, Any]]:
    batch_size = layout["batch_size"]
    max_agents = layout["max_agents"]
    max_seq_len = layout["max_seq_len"]
    max_roads = layout["max_roads"]
    max_road_pts = layout["max_road_pts"]
    max_timestep_feat_dim = layout["max_timestep_feat_dim"]
    timesteps_shape = (batch_size, max_agents, max_seq_len, max_timestep_feat_dim)
    dtypes = layout["dtypes"]
    return {
        "agent_states_b": (
            (batch_size, max_agents, max_seq_len, layout["agent_feat_dim"]),
            dtypes["agent_states"],
            0,
        ),
        "agent_types_b": (
            (batch_size, max_agents, layout["agent_type_dim"]),
            dtypes["agent_types"],
            -1,
        ),
        "goals_b": (
            (batch_size, max_agents, layout["goal_dim"]),
            dtypes["goals"],
            0,
        ),
        "actions_b": (
            (batch_size, max_agents, max_seq_len),
            dtypes["actions"],
            0,
        ),
        "rtgs_b": (
            (batch_size, max_agents, max_seq_len, layout["rtg_dim"]),
            dtypes["rtgs"],
            0,
        ),
        "timesteps_b": (timesteps_shape, dtypes["timesteps"], 0),
        "moving_agent_mask_b": (
            (batch_size, max_agents),
            dtypes["moving_agent_mask"],
            0,
        ),
        "road_points_b": (
            (batch_size, max_roads, max_road_pts, layout["road_pts_dim"]),
            dtypes["road_points"],
            0,
        ),
        "road_types_b": (
            (batch_size, max_roads, layout["road_type_dim"]),
            dtypes["road_types"],
            -1,
        ),
        "token_index_per_job": ((batch_size,), np.dtype(np.int64), 0),
    }


def build_collate_cache_key(layout: Dict[str, Any]) -> Tuple[Any, ...]:
    dtypes = layout["dtypes"]
    return (
        layout["batch_size"],
        layout["max_agents"],
        layout["max_seq_len"],
        layout["max_roads"],
        layout["max_road_pts"],
        layout["max_timestep_feat_dim"],
        dtypes["agent_states"].str,
        dtypes["agent_types"].str,
        dtypes["goals"].str,
        dtypes["actions"].str,
        dtypes["rtgs"].str,
        dtypes["timesteps"].str,
        dtypes["moving_agent_mask"].str,
        dtypes["road_points"].str,
        dtypes["road_types"].str,
    )


def fill_collate_buffers(jobs: List[Dict[str, Any]], buffers: Dict[str, np.ndarray]) -> None:
    for batch_idx, job in enumerate(jobs):
        prepared = job["prepared"]
        motion_data_np = job["focal_batch"]["motion_data_np"]
        n_agents = int(motion_data_np["agent_states"].shape[0])
        seq_len = int(motion_data_np["agent_states"].shape[1])
        n_roads = int(motion_data_np["road_points"].shape[0])
        n_road_pts = int(motion_data_np["road_points"].shape[1])

        buffers["agent_states_b"][batch_idx, :n_agents, :seq_len] = motion_data_np["agent_states"]
        buffers["agent_types_b"][batch_idx, :n_agents] = motion_data_np["agent_types"]
        buffers["goals_b"][batch_idx, :n_agents] = motion_data_np["goals"]
        buffers["actions_b"][batch_idx, :n_agents, :seq_len] = motion_data_np["actions"]
        buffers["rtgs_b"][batch_idx, :n_agents, :seq_len] = motion_data_np["rtgs"]

        timesteps = motion_data_np["timesteps"]
        t_feat_dim = int(timesteps.shape[2])
        buffers["timesteps_b"][batch_idx, :n_agents, :seq_len, :t_feat_dim] = timesteps
        buffers["moving_agent_mask_b"][batch_idx, :n_agents] = motion_data_np["moving_agent_mask"]
        buffers["road_points_b"][batch_idx, :n_roads, :n_road_pts] = motion_data_np["road_points"]
        buffers["road_types_b"][batch_idx, :n_roads] = motion_data_np["road_types"]

        raw_token_index = int(prepared["token_index"])
        if raw_token_index < 0:
            resolved_token_index = seq_len + raw_token_index
        else:
            resolved_token_index = raw_token_index
        resolved_token_index = max(0, min(resolved_token_index, seq_len - 1))
        buffers["token_index_per_job"][batch_idx] = resolved_token_index


def _resolve_idx_in_model(
    veh_id: int,
    veh_id_to_idx: Dict[int, int],
    new_agent_idx_dict: Dict[int, int],
) -> int | None:
    agent_key = veh_id_to_idx.get(int(veh_id))
    if agent_key is None:
        return None
    idx_in_model = new_agent_idx_dict.get(int(agent_key))
    if idx_in_model is None:
        return None
    return int(idx_in_model)


def _as_int64_array(values: List[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _as_float32_array(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _as_bool_array(values: List[bool]) -> np.ndarray:
    return np.asarray(values, dtype=np.bool_)


def _as_uint64_array(values: List[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def build_decode_metadata(
    jobs: List[Dict[str, Any]],
    token_index_per_job: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    action_job_idx: List[int] = []
    action_idx_in_model: List[int] = []
    action_token_index: List[int] = []
    action_veh_id: List[int] = []
    action_step_t: List[int] = []
    action_sampling_seed: List[int] = []
    action_temperature: List[float] = []
    action_nucleus_sampling: List[bool] = []
    action_nucleus_threshold: List[float] = []
    action_job_offsets = [0]

    rtg_job_idx: List[int] = []
    rtg_env_idx: List[int] = []
    rtg_idx_in_model: List[int] = []
    rtg_token_index: List[int] = []
    rtg_veh_id: List[int] = []
    rtg_step_t: List[int] = []
    rtg_sampling_seed: List[int] = []
    rtg_goal_tilt: List[int] = []
    rtg_veh_tilt: List[int] = []
    rtg_road_tilt: List[int] = []
    rtg_job_offsets = [0]

    for batch_idx, job in enumerate(jobs):
        prepared = job["prepared"]
        focal_batch = job["focal_batch"]
        veh_id_to_idx = prepared["veh_id_to_idx"]
        new_agent_idx_dict = focal_batch["new_agent_idx_dict"]
        token_index = int(token_index_per_job[batch_idx])
        step_t = int(prepared["step_t"])
        sampling_seed = int(prepared["sampling_seed"])

        sampling = prepared["sampling"]
        for veh_id in focal_batch["data_veh_ids"]:
            idx_in_model = _resolve_idx_in_model(
                veh_id=int(veh_id),
                veh_id_to_idx=veh_id_to_idx,
                new_agent_idx_dict=new_agent_idx_dict,
            )
            if idx_in_model is None:
                continue
            action_job_idx.append(batch_idx)
            action_idx_in_model.append(idx_in_model)
            action_token_index.append(token_index)
            action_veh_id.append(int(veh_id))
            action_step_t.append(step_t)
            action_sampling_seed.append(sampling_seed)
            action_temperature.append(float(sampling["action_temperature"]))
            action_nucleus_sampling.append(bool(sampling["nucleus_sampling"]))
            action_nucleus_threshold.append(float(sampling["nucleus_threshold"]))
        action_job_offsets.append(len(action_job_idx))

        if not bool(focal_batch["predict_rtgs"]):
            rtg_job_offsets.append(len(rtg_job_idx))
            continue

        data_veh_id_set = set(int(veh_id) for veh_id in focal_batch["data_veh_ids"])
        default_tilt = prepared["default_tilt"]
        tilt_by_veh_id = prepared["tilt_by_veh_id"]
        for veh_id in focal_batch["veh_ids_in_context"]:
            idx_in_model = _resolve_idx_in_model(
                veh_id=int(veh_id),
                veh_id_to_idx=veh_id_to_idx,
                new_agent_idx_dict=new_agent_idx_dict,
            )
            if idx_in_model is None:
                continue
            veh_id_int = int(veh_id)
            if veh_id_int in data_veh_id_set:
                goal_tilt, veh_tilt, road_tilt = tilt_by_veh_id.get(veh_id_int, default_tilt)
            else:
                goal_tilt, veh_tilt, road_tilt = (0, 0, 0)
            rtg_job_idx.append(batch_idx)
            rtg_env_idx.append(int(job["env_idx"]))
            rtg_idx_in_model.append(idx_in_model)
            rtg_token_index.append(token_index)
            rtg_veh_id.append(veh_id_int)
            rtg_step_t.append(step_t)
            rtg_sampling_seed.append(sampling_seed)
            rtg_goal_tilt.append(int(goal_tilt))
            rtg_veh_tilt.append(int(veh_tilt))
            rtg_road_tilt.append(int(road_tilt))
        rtg_job_offsets.append(len(rtg_job_idx))

    return {
        "action": {
            "job_idx": _as_int64_array(action_job_idx),
            "idx_in_model": _as_int64_array(action_idx_in_model),
            "token_index": _as_int64_array(action_token_index),
            "veh_id": _as_int64_array(action_veh_id),
            "step_t": _as_int64_array(action_step_t),
            "sampling_seed": _as_uint64_array(action_sampling_seed),
            "temperature": _as_float32_array(action_temperature),
            "nucleus_sampling": _as_bool_array(action_nucleus_sampling),
            "nucleus_threshold": _as_float32_array(action_nucleus_threshold),
            "job_offsets": _as_int64_array(action_job_offsets),
            "job_count": np.asarray([len(jobs)], dtype=np.int64),
        },
        "rtg": {
            "job_idx": _as_int64_array(rtg_job_idx),
            "env_idx": _as_int64_array(rtg_env_idx),
            "idx_in_model": _as_int64_array(rtg_idx_in_model),
            "token_index": _as_int64_array(rtg_token_index),
            "veh_id": _as_int64_array(rtg_veh_id),
            "step_t": _as_int64_array(rtg_step_t),
            "sampling_seed": _as_uint64_array(rtg_sampling_seed),
            "goal_tilt": _as_int64_array(rtg_goal_tilt),
            "veh_tilt": _as_int64_array(rtg_veh_tilt),
            "road_tilt": _as_int64_array(rtg_road_tilt),
            "job_offsets": _as_int64_array(rtg_job_offsets),
            "job_count": np.asarray([len(jobs)], dtype=np.int64),
        },
    }


def build_motion_data_from_buffers(
    buffers: Dict[str, np.ndarray],
    device: str,
) -> MotionData:
    batched_np = {
        "agent": {
            "agent_states": buffers["agent_states_b"],
            "agent_types": buffers["agent_types_b"],
            "goals": buffers["goals_b"],
            "actions": buffers["actions_b"],
            "rtgs": buffers["rtgs_b"],
            "timesteps": buffers["timesteps_b"],
            "moving_agent_mask": buffers["moving_agent_mask_b"],
        },
        "map": {
            "road_points": buffers["road_points_b"],
            "road_types": buffers["road_types_b"],
        },
    }
    batched_cpu = MotionData(from_numpy(batched_np))
    return batched_cpu.to(device)


def attach_decode_meta_tensors(
    decode_meta: Dict[str, Dict[str, np.ndarray]],
    device: str,
) -> None:
    action_meta = decode_meta["action"]
    action_meta["job_idx_t"] = torch.as_tensor(action_meta["job_idx"], dtype=torch.long, device=device)
    action_meta["idx_in_model_t"] = torch.as_tensor(action_meta["idx_in_model"], dtype=torch.long, device=device)
    action_meta["token_index_t"] = torch.as_tensor(action_meta["token_index"], dtype=torch.long, device=device)
    action_meta["temperature_t"] = torch.as_tensor(action_meta["temperature"], dtype=torch.float32, device=device)
    action_meta["nucleus_sampling_t"] = torch.as_tensor(action_meta["nucleus_sampling"], dtype=torch.bool, device=device)
    action_meta["nucleus_threshold_t"] = torch.as_tensor(action_meta["nucleus_threshold"], dtype=torch.float32, device=device)

    rtg_meta = decode_meta["rtg"]
    rtg_meta["job_idx_t"] = torch.as_tensor(rtg_meta["job_idx"], dtype=torch.long, device=device)
    rtg_meta["idx_in_model_t"] = torch.as_tensor(rtg_meta["idx_in_model"], dtype=torch.long, device=device)
    rtg_meta["token_index_t"] = torch.as_tensor(rtg_meta["token_index"], dtype=torch.long, device=device)
    rtg_meta["goal_tilt_t"] = torch.as_tensor(rtg_meta["goal_tilt"], dtype=torch.float32, device=device)
    rtg_meta["veh_tilt_t"] = torch.as_tensor(rtg_meta["veh_tilt"], dtype=torch.float32, device=device)
    rtg_meta["road_tilt_t"] = torch.as_tensor(rtg_meta["road_tilt"], dtype=torch.float32, device=device)


def collate_jobs_with_padding(
    jobs: List[Dict[str, Any]],
    device: str,
    collate_numpy_buffers: Dict[Tuple[Any, ...], Dict[str, np.ndarray]],
) -> Tuple[MotionData, Dict[str, Any]]:
    if not jobs:
        raise ValueError("jobs must not be empty")

    layout = infer_job_batch_layout(jobs)
    specs = build_collate_specs(layout)
    cache_key = build_collate_cache_key(layout)
    buffers = get_or_create_collate_buffers(
        collate_numpy_buffers=collate_numpy_buffers,
        cache_key=cache_key,
        specs=specs,
    )

    fill_collate_buffers(jobs, buffers)

    batched_data = build_motion_data_from_buffers(
        buffers,
        device=device,
    )

    token_index_per_job = torch.from_numpy(buffers["token_index_per_job"])

    decode_meta = build_decode_metadata(
        jobs=jobs,
        token_index_per_job=buffers["token_index_per_job"],
    )
    attach_decode_meta_tensors(decode_meta=decode_meta, device=device)

    batch_meta = {
        "jobs": jobs,
        "token_index_per_job": token_index_per_job,
        "decode_meta": decode_meta,
    }
    return batched_data, batch_meta
