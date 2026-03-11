"""
负责将多个 focal job 整理成统一的批量输入，并完成 padding 与缓冲区复用。
该模块将离散的 numpy 字段转换成 MotionData，供 ExternalTeacher 的批量前向阶段直接消费。
Builds uniform batched inputs from focal jobs, including padding and reusable buffer management.
Converts scattered numpy fields into MotionData for ExternalTeacher's batched forward stage.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.data import MotionData, from_numpy


def _elapsed_ms(start_time: float, profile_enabled: bool) -> float:
    if not profile_enabled:
        return 0.0
    return (time.perf_counter() - start_time) * 1000.0


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


def infer_chunk_layout(chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_motion = chunk[0]["focal_batch"]["motion_data_np"]
    max_agents = 1
    max_seq_len = 1
    max_roads = 1
    max_road_pts = 1
    max_timestep_feat_dim = 1
    for job in chunk:
        motion_data_np = job["focal_batch"]["motion_data_np"]
        max_agents = max(max_agents, int(motion_data_np["agent_states"].shape[0]))
        max_seq_len = max(max_seq_len, int(motion_data_np["agent_states"].shape[1]))
        max_roads = max(max_roads, int(motion_data_np["road_points"].shape[0]))
        max_road_pts = max(max_road_pts, int(motion_data_np["road_points"].shape[1]))
        max_timestep_feat_dim = max(max_timestep_feat_dim, int(motion_data_np["timesteps"].shape[2]))

    return {
        "batch_size": len(chunk),
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


def fill_collate_buffers(chunk: List[Dict[str, Any]], buffers: Dict[str, np.ndarray]) -> None:
    for batch_idx, job in enumerate(chunk):
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


def build_motion_data_from_buffers(
    buffers: Dict[str, np.ndarray],
    device: str,
    profile_enabled: bool = False,
) -> Tuple[MotionData, Dict[str, float]]:
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
    from_numpy_start = time.perf_counter() if profile_enabled else 0.0
    batched_cpu = MotionData(from_numpy(batched_np))
    from_numpy_ms = _elapsed_ms(from_numpy_start, profile_enabled)

    to_device_start = time.perf_counter() if profile_enabled else 0.0
    batched_data = batched_cpu.to(device)
    to_device_ms = _elapsed_ms(to_device_start, profile_enabled)
    return batched_data, {
        "from_numpy": from_numpy_ms,
        "to_device": to_device_ms,
        "total": from_numpy_ms + to_device_ms,
    }


def collate_chunk_with_padding(
    chunk: List[Dict[str, Any]],
    device: str,
    collate_numpy_buffers: Dict[Tuple[Any, ...], Dict[str, np.ndarray]],
    profile_enabled: bool = False,
) -> Tuple[MotionData, Dict[str, Any]]:
    if not chunk:
        raise ValueError("chunk must not be empty")

    total_start = time.perf_counter() if profile_enabled else 0.0

    infer_layout_start = time.perf_counter() if profile_enabled else 0.0
    layout = infer_chunk_layout(chunk)
    infer_layout_ms = _elapsed_ms(infer_layout_start, profile_enabled)

    get_buffers_start = time.perf_counter() if profile_enabled else 0.0
    specs = build_collate_specs(layout)
    cache_key = build_collate_cache_key(layout)
    buffers = get_or_create_collate_buffers(
        collate_numpy_buffers=collate_numpy_buffers,
        cache_key=cache_key,
        specs=specs,
    )
    get_buffers_ms = _elapsed_ms(get_buffers_start, profile_enabled)

    fill_buffers_start = time.perf_counter() if profile_enabled else 0.0
    fill_collate_buffers(chunk, buffers)
    fill_buffers_ms = _elapsed_ms(fill_buffers_start, profile_enabled)

    build_motion_data_start = time.perf_counter() if profile_enabled else 0.0
    batched_data, motion_data_profile = build_motion_data_from_buffers(
        buffers,
        device=device,
        profile_enabled=profile_enabled,
    )
    build_motion_data_ms = _elapsed_ms(build_motion_data_start, profile_enabled)

    token_index_to_device_start = time.perf_counter() if profile_enabled else 0.0
    token_index_per_job = torch.from_numpy(buffers["token_index_per_job"])
    token_index_to_device_ms = _elapsed_ms(token_index_to_device_start, profile_enabled)

    batch_meta = {
        "jobs": chunk,
        "token_index_per_job": token_index_per_job,
        "collate_profile": {
            "infer_layout": infer_layout_ms,
            "get_buffers": get_buffers_ms,
            "fill_buffers": fill_buffers_ms,
            "build_motion_data": build_motion_data_ms,
            "from_numpy": motion_data_profile["from_numpy"],
            "to_device": motion_data_profile["to_device"],
            "token_index_to_device": token_index_to_device_ms,
            "total": _elapsed_ms(total_start, profile_enabled),
        },
    }
    return batched_data, batch_meta
