"""
负责将多个 focal job 整理成统一的批量输入，并完成轻量级缓冲区复用。
该模块直接消费 flat prepared layout，只做按 focal 索引切片、必要 copy 和解码 metadata 组装。
Builds the batched model input for multiple focal jobs while reusing lightweight buffers.
It consumes the flat prepared layout directly and only performs focal-row slicing, necessary copies,
and decode-metadata assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from batch_inference.batch_ipc.prepared import (
    get_prepared_focal_context_model_indices,
    get_prepared_focal_context_veh_ids,
    get_prepared_focal_data_model_indices,
    get_prepared_focal_data_veh_ids,
    get_prepared_focal_motion_data,
    get_prepared_focal_predict_rtgs,
)
from .external_teacher_helper import (
    _allocate_host_collate_buffer,
    _build_rtg_row_metadata,
    _concat_or_empty,
    _numpy_dtype_to_torch,
    _resolve_token_index,
)
from utils.data import MotionData


@dataclass
class CollateBufferSet:
    """聚合一组可复用的 collate 缓冲区。 / Aggregate one reusable collate-buffer bundle.

    该结构把 host tensor 与 numpy 视图显式收敛到一起，避免使用带有魔法字符串键的
    嵌套字典，同时把可复用范围控制在最必要的 host 侧缓冲区。

    This structure keeps host tensors and numpy views together explicitly. It replaces
    nested dictionaries keyed by magic strings while limiting reuse to the minimum
    host-side buffers that the collator always needs.
    """

    numpy_views: Dict[str, np.ndarray] = field(default_factory=dict)
    host_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)



def _build_motion_data_from_tensor_buffers(
    tensor_buffers: Dict[str, torch.Tensor],
) -> MotionData:
    """从 tensor 缓冲区构建 MotionData。 / Build a MotionData instance from reusable tensor buffers.

    这里不再经过 `from_numpy()` 中转，而是直接把已经缓存好的 tensor 组织成 MotionData，
    以避免每个 active inference step 重复创建新的中间 tensor 对象。

    This path skips `from_numpy()` and directly organizes the cached tensors into a
    MotionData object so active inference steps do not keep recreating intermediate tensors.
    """
    return MotionData(
        {
            "agent": {
                "agent_states": tensor_buffers["agent_states_b"],
                "agent_types": tensor_buffers["agent_types_b"],
                "goals": tensor_buffers["goals_b"],
                "actions": tensor_buffers["actions_b"],
                "rtgs": tensor_buffers["rtgs_b"],
                "timesteps": tensor_buffers["timesteps_b"],
                "moving_agent_mask": tensor_buffers["moving_agent_mask_b"],
            },
            "map": {
                "road_points": tensor_buffers["road_points_b"],
                "road_types": tensor_buffers["road_types_b"],
            },
        }
    )


def get_or_create_collate_buffers(
    collate_numpy_buffers: Dict[Tuple[Any, ...], CollateBufferSet],
    cache_key: Tuple[Any, ...],
    specs: Dict[str, Tuple[Tuple[int, ...], np.dtype, Any]],
) -> CollateBufferSet:
    """按固定 layout 规格获取可复用的 collate 缓冲区。

    Get reusable collate buffers for the fixed layout described by the given specs.
    """
    buffers = collate_numpy_buffers.get(cache_key)
    if buffers is None:
        buffers = CollateBufferSet()
        collate_numpy_buffers[cache_key] = buffers

    host_tensors = buffers.host_tensors
    for name, (shape, dtype, fill_value) in specs.items():
        host_tensor = host_tensors.get(name)
        if (
            host_tensor is None
            or tuple(host_tensor.shape) != tuple(shape)
            or host_tensor.dtype != _numpy_dtype_to_torch(dtype)
        ):
            host_tensor, np_view = _allocate_host_collate_buffer(shape, dtype)
            host_tensors[name] = host_tensor
            buffers.numpy_views[name] = np_view
        if fill_value != 0:
            buffers.numpy_views[name].fill(fill_value)
        else:
            buffers.numpy_views[name].fill(0)

    return buffers


def infer_job_batch_layout(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从第一个 job 的 batched motion row 推断固定 collate layout。

    Infer the fixed collate layout from the first job's batched motion row.
    """
    first_job = jobs[0]
    first_prepared = first_job["prepared"]
    first_motion = get_prepared_focal_motion_data(
        first_prepared,
        int(first_job["focal_idx"]),
    )
    shared_timesteps = np.asarray(first_prepared["shared_timesteps"])
    return {
        "batch_size": len(jobs),
        "agent_states_shape": tuple(int(dim) for dim in first_motion["agent_states"].shape),
        "agent_types_shape": tuple(int(dim) for dim in first_motion["agent_types"].shape),
        "goals_shape": tuple(int(dim) for dim in first_motion["goals"].shape),
        "actions_shape": tuple(int(dim) for dim in first_motion["actions"].shape),
        "rtgs_shape": tuple(int(dim) for dim in first_motion["rtgs"].shape),
        "moving_mask_shape": tuple(int(dim) for dim in first_motion["moving_agent_mask"].shape),
        "road_points_shape": tuple(int(dim) for dim in first_motion["road_points"].shape),
        "road_types_shape": tuple(int(dim) for dim in first_motion["road_types"].shape),
        "timesteps_shape": tuple(int(dim) for dim in shared_timesteps.shape),
        "dtypes": {
            "agent_states": np.dtype(first_motion["agent_states"].dtype),
            "agent_types": np.dtype(first_motion["agent_types"].dtype),
            "goals": np.dtype(first_motion["goals"].dtype),
            "actions": np.dtype(first_motion["actions"].dtype),
            "rtgs": np.dtype(first_motion["rtgs"].dtype),
            "timesteps": np.dtype(shared_timesteps.dtype),
            "moving_agent_mask": np.dtype(first_motion["moving_agent_mask"].dtype),
            "road_points": np.dtype(first_motion["road_points"].dtype),
            "road_types": np.dtype(first_motion["road_types"].dtype),
        },
    }


def build_collate_specs(
    layout: Dict[str, Any],
) -> Dict[str, Tuple[Tuple[int, ...], np.dtype, Any]]:
    """根据固定 layout 生成 collate 缓冲区规格。

    Build the collate buffer specs from the fixed batch layout.
    """
    batch_size = int(layout["batch_size"])
    dtypes = layout["dtypes"]
    return {
        "agent_states_b": (
            (batch_size, *layout["agent_states_shape"]),
            dtypes["agent_states"],
            0,
        ),
        "agent_types_b": (
            (batch_size, *layout["agent_types_shape"]),
            dtypes["agent_types"],
            -1,
        ),
        "goals_b": ((batch_size, *layout["goals_shape"]), dtypes["goals"], 0),
        "actions_b": ((batch_size, *layout["actions_shape"]), dtypes["actions"], 0),
        "rtgs_b": ((batch_size, *layout["rtgs_shape"]), dtypes["rtgs"], 0),
        "timesteps_b": ((batch_size, *layout["timesteps_shape"]), dtypes["timesteps"], 0),
        "moving_agent_mask_b": (
            (batch_size, *layout["moving_mask_shape"]),
            dtypes["moving_agent_mask"],
            0,
        ),
        "road_points_b": (
            (batch_size, *layout["road_points_shape"]),
            dtypes["road_points"],
            0,
        ),
        "road_types_b": (
            (batch_size, *layout["road_types_shape"]),
            dtypes["road_types"],
            -1,
        ),
        "token_index_per_job": ((batch_size,), np.dtype(np.int64), 0),
    }


def build_collate_cache_key(layout: Dict[str, Any]) -> Tuple[Any, ...]:
    """将固定 layout 编码成可复用缓冲区的缓存键。

    Encode the fixed layout into the cache key used for reusable collate buffers.
    """
    dtypes = layout["dtypes"]
    return (
        int(layout["batch_size"]),
        *layout["agent_states_shape"],
        *layout["road_points_shape"],
        *layout["timesteps_shape"],
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


def fill_collate_buffers(jobs: List[Dict[str, Any]], buffers: CollateBufferSet) -> None:
    """把每个 job 的 batched row copy 进 collate 缓冲区。

    Copy each job's batched focal row into the collate buffers.
    """
    numpy_views = buffers.numpy_views
    for batch_idx, job in enumerate(jobs):
        prepared = job["prepared"]
        focal_idx = int(job["focal_idx"])
        motion = get_prepared_focal_motion_data(prepared, focal_idx)
        numpy_views["agent_states_b"][batch_idx] = motion["agent_states"]
        numpy_views["agent_types_b"][batch_idx] = motion["agent_types"]
        numpy_views["goals_b"][batch_idx] = motion["goals"]
        numpy_views["actions_b"][batch_idx] = motion["actions"]
        numpy_views["rtgs_b"][batch_idx] = motion["rtgs"]
        numpy_views["timesteps_b"][batch_idx] = np.asarray(prepared["shared_timesteps"])
        numpy_views["moving_agent_mask_b"][batch_idx] = motion["moving_agent_mask"]
        numpy_views["road_points_b"][batch_idx] = motion["road_points"]
        numpy_views["road_types_b"][batch_idx] = motion["road_types"]
        seq_len = int(motion["agent_states"].shape[1])
        numpy_views["token_index_per_job"][batch_idx] = _resolve_token_index(
            int(prepared["token_index"]),
            seq_len,
        )


def build_decode_metadata(
    jobs: List[Dict[str, Any]],
    token_index_per_job: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """从 flat prepared layout 构建 action/RTG 解码元数据。

    Build action/RTG decode metadata directly from the flat prepared layout.
    """
    action_job_idx_parts: List[np.ndarray] = []
    action_idx_in_model_parts: List[np.ndarray] = []
    action_token_index_parts: List[np.ndarray] = []
    action_veh_id_parts: List[np.ndarray] = []
    action_env_idx_parts: List[np.ndarray] = []
    action_step_t_parts: List[np.ndarray] = []
    action_sampling_seed_parts: List[np.ndarray] = []
    action_temperature_parts: List[np.ndarray] = []
    action_nucleus_sampling_parts: List[np.ndarray] = []
    action_nucleus_threshold_parts: List[np.ndarray] = []
    action_effective_scale_parts: List[np.ndarray] = []
    action_job_offsets = [0]

    rtg_job_idx_parts: List[np.ndarray] = []
    rtg_env_idx_parts: List[np.ndarray] = []
    rtg_idx_in_model_parts: List[np.ndarray] = []
    rtg_token_index_parts: List[np.ndarray] = []
    rtg_veh_id_parts: List[np.ndarray] = []
    rtg_step_t_parts: List[np.ndarray] = []
    rtg_sampling_seed_parts: List[np.ndarray] = []
    rtg_goal_tilt_parts: List[np.ndarray] = []
    rtg_veh_tilt_parts: List[np.ndarray] = []
    rtg_road_tilt_parts: List[np.ndarray] = []
    rtg_effective_scale_parts: List[np.ndarray] = []
    rtg_job_offsets = [0]
    for batch_idx, job in enumerate(jobs):
        prepared = job["prepared"]
        focal_idx = int(job["focal_idx"])
        token_index = int(token_index_per_job[batch_idx])
        step_t = int(prepared["step_t"])
        sampling_seed = int(prepared["sampling_seed"])
        sampling = prepared["sampling"]
        delayed_scale = float(prepared.get("delayed_ego_action_scale", 1.0))
        is_opponent_job = str(job.get("job_type", "opponent")) == "opponent"

        data_veh_ids = get_prepared_focal_data_veh_ids(prepared, focal_idx)
        data_model_indices = get_prepared_focal_data_model_indices(prepared, focal_idx)
        valid_action_mask = data_model_indices >= 0
        valid_action_count = int(np.count_nonzero(valid_action_mask))
        action_job_offsets.append(action_job_offsets[-1] + valid_action_count)
        if valid_action_count > 0:
            action_job_idx_parts.append(
                np.full((valid_action_count,), batch_idx, dtype=np.int64)
            )
            action_idx_in_model_parts.append(
                data_model_indices[valid_action_mask].astype(np.int64, copy=False)
            )
            action_token_index_parts.append(
                np.full((valid_action_count,), token_index, dtype=np.int64)
            )
            action_veh_id_parts.append(
                data_veh_ids[valid_action_mask].astype(np.int64, copy=False)
            )
            action_env_idx_parts.append(
                np.full((valid_action_count,), int(job["env_idx"]), dtype=np.int64)
            )
            action_step_t_parts.append(
                np.full((valid_action_count,), step_t, dtype=np.int64)
            )
            action_sampling_seed_parts.append(
                np.full((valid_action_count,), sampling_seed, dtype=np.uint64)
            )
            action_temperature_parts.append(
                np.full(
                    (valid_action_count,),
                    float(sampling["action_temperature"]),
                    dtype=np.float32,
                )
            )
            action_nucleus_sampling_parts.append(
                np.full(
                    (valid_action_count,),
                    bool(sampling["nucleus_sampling"]),
                    dtype=np.bool_,
                )
            )
            action_nucleus_threshold_parts.append(
                np.full(
                    (valid_action_count,),
                    float(sampling["nucleus_threshold"]),
                    dtype=np.float32,
                )
            )
            action_effective_scale_parts.append(
                np.full(
                    (valid_action_count,),
                    delayed_scale if is_opponent_job else 1.0,
                    dtype=np.float32,
                )
            )

        if not get_prepared_focal_predict_rtgs(prepared, focal_idx):
            rtg_job_offsets.append(rtg_job_offsets[-1])
            continue

        context_veh_ids = get_prepared_focal_context_veh_ids(prepared, focal_idx)
        context_model_indices = get_prepared_focal_context_model_indices(prepared, focal_idx)
        valid_rtg_mask = context_model_indices >= 0
        valid_context_veh_ids = context_veh_ids[valid_rtg_mask].astype(np.int64, copy=False)
        valid_context_indices = context_model_indices[valid_rtg_mask].astype(
            np.int64,
            copy=False,
        )
        valid_rtg_count = int(valid_context_veh_ids.shape[0])
        rtg_job_offsets.append(rtg_job_offsets[-1] + valid_rtg_count)
        if valid_rtg_count <= 0:
            continue

        goal_tilt, veh_tilt, road_tilt, effective_scale_per_row = _build_rtg_row_metadata(
            valid_context_veh_ids=valid_context_veh_ids,
            data_veh_ids=data_veh_ids,
            default_tilt=tuple(int(v) for v in prepared["default_tilt"]),
            tilt_by_veh_id=prepared["tilt_by_veh_id"],
            delayed_scale=delayed_scale,
            is_opponent_job=is_opponent_job,
        )

        rtg_job_idx_parts.append(np.full((valid_rtg_count,), batch_idx, dtype=np.int64))
        rtg_env_idx_parts.append(
            np.full((valid_rtg_count,), int(job["env_idx"]), dtype=np.int64)
        )
        rtg_idx_in_model_parts.append(valid_context_indices)
        rtg_token_index_parts.append(
            np.full((valid_rtg_count,), token_index, dtype=np.int64)
        )
        rtg_veh_id_parts.append(valid_context_veh_ids)
        rtg_step_t_parts.append(np.full((valid_rtg_count,), step_t, dtype=np.int64))
        rtg_sampling_seed_parts.append(
            np.full((valid_rtg_count,), sampling_seed, dtype=np.uint64)
        )
        rtg_goal_tilt_parts.append(goal_tilt)
        rtg_veh_tilt_parts.append(veh_tilt)
        rtg_road_tilt_parts.append(road_tilt)
        rtg_effective_scale_parts.append(effective_scale_per_row)

    return {
        "action": {
            "job_idx": _concat_or_empty(action_job_idx_parts, dtype=np.int64),
            "env_idx": _concat_or_empty(action_env_idx_parts, dtype=np.int64),
            "idx_in_model": _concat_or_empty(action_idx_in_model_parts, dtype=np.int64),
            "token_index": _concat_or_empty(action_token_index_parts, dtype=np.int64),
            "veh_id": _concat_or_empty(action_veh_id_parts, dtype=np.int64),
            "step_t": _concat_or_empty(action_step_t_parts, dtype=np.int64),
            "sampling_seed": _concat_or_empty(action_sampling_seed_parts, dtype=np.uint64),
            "temperature": _concat_or_empty(action_temperature_parts, dtype=np.float32),
            "nucleus_sampling": _concat_or_empty(
                action_nucleus_sampling_parts,
                dtype=np.bool_,
            ),
            "nucleus_threshold": _concat_or_empty(
                action_nucleus_threshold_parts,
                dtype=np.float32,
            ),
            "effective_scale": _concat_or_empty(
                action_effective_scale_parts,
                dtype=np.float32,
            ),
            "job_offsets": np.asarray(action_job_offsets, dtype=np.int64),
            "job_count": np.asarray([len(jobs)], dtype=np.int64),
        },
        "rtg": {
            "job_idx": _concat_or_empty(rtg_job_idx_parts, dtype=np.int64),
            "env_idx": _concat_or_empty(rtg_env_idx_parts, dtype=np.int64),
            "idx_in_model": _concat_or_empty(rtg_idx_in_model_parts, dtype=np.int64),
            "token_index": _concat_or_empty(rtg_token_index_parts, dtype=np.int64),
            "veh_id": _concat_or_empty(rtg_veh_id_parts, dtype=np.int64),
            "step_t": _concat_or_empty(rtg_step_t_parts, dtype=np.int64),
            "sampling_seed": _concat_or_empty(rtg_sampling_seed_parts, dtype=np.uint64),
            "goal_tilt": _concat_or_empty(rtg_goal_tilt_parts, dtype=np.int64),
            "veh_tilt": _concat_or_empty(rtg_veh_tilt_parts, dtype=np.int64),
            "road_tilt": _concat_or_empty(rtg_road_tilt_parts, dtype=np.int64),
            "effective_scale": _concat_or_empty(
                rtg_effective_scale_parts,
                dtype=np.float32,
            ),
            "job_offsets": np.asarray(rtg_job_offsets, dtype=np.int64),
            "job_count": np.asarray([len(jobs)], dtype=np.int64),
        },
    }


def build_motion_data_from_buffers(
    buffers: CollateBufferSet,
    device: str,
) -> MotionData:
    """将填充后的 numpy 缓冲区组装成 `MotionData` 并搬运到目标设备。

    Assemble the populated numpy buffers into `MotionData` and move it to the target device.
    """
    host_tensors = buffers.host_tensors
    if str(device) == "cpu":
        return _build_motion_data_from_tensor_buffers(host_tensors)

    device_tensor_map = {
        name: host_tensor.to(device=device)
        for name, host_tensor in host_tensors.items()
    }
    return _build_motion_data_from_tensor_buffers(device_tensor_map)


def attach_decode_meta_tensors(
    decode_meta: Dict[str, Dict[str, np.ndarray]],
    device: str,
) -> None:
    """将解码元数据中的关键数组搬运到目标设备上，并提供 torch-only decode 所需缓存。

    Move key decode metadata arrays onto the target device and expose the
    tensor caches required by the torch-only decode path.
    """
    action_meta = decode_meta["action"]
    action_meta["job_idx_t"] = torch.as_tensor(
        np.asarray(action_meta["job_idx"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    action_meta["idx_in_model_t"] = torch.as_tensor(
        np.asarray(action_meta["idx_in_model"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    action_meta["token_index_t"] = torch.as_tensor(
        np.asarray(action_meta["token_index"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    action_meta["temperature_t"] = torch.as_tensor(
        np.asarray(action_meta["temperature"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    action_meta["nucleus_sampling_t"] = torch.as_tensor(
        np.asarray(action_meta["nucleus_sampling"], dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    )
    action_meta["nucleus_threshold_t"] = torch.as_tensor(
        np.asarray(action_meta["nucleus_threshold"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    action_meta["effective_scale_t"] = torch.as_tensor(
        np.asarray(action_meta["effective_scale"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    rtg_meta = decode_meta["rtg"]
    rtg_meta["job_idx_t"] = torch.as_tensor(
        np.asarray(rtg_meta["job_idx"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["idx_in_model_t"] = torch.as_tensor(
        np.asarray(rtg_meta["idx_in_model"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["token_index_t"] = torch.as_tensor(
        np.asarray(rtg_meta["token_index"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["goal_tilt_t"] = torch.as_tensor(
        np.asarray(rtg_meta["goal_tilt"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    rtg_meta["veh_tilt_t"] = torch.as_tensor(
        np.asarray(rtg_meta["veh_tilt"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    rtg_meta["road_tilt_t"] = torch.as_tensor(
        np.asarray(rtg_meta["road_tilt"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    rtg_meta["env_idx_t"] = torch.as_tensor(
        np.asarray(rtg_meta["env_idx"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["veh_id_t"] = torch.as_tensor(
        np.asarray(rtg_meta["veh_id"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["step_t_t"] = torch.as_tensor(
        np.asarray(rtg_meta["step_t"], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    rtg_meta["effective_scale_t"] = torch.as_tensor(
        np.asarray(rtg_meta["effective_scale"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )


def collate_jobs_with_padding(
    jobs: List[Dict[str, Any]],
    device: str,
    collate_numpy_buffers: Dict[Tuple[Any, ...], CollateBufferSet],
) -> Tuple[MotionData, Dict[str, Any]]:
    """将多个 flat focal job 整理成统一 batch，并构建解码元数据。

    Collate multiple flat focal jobs into one batch and build the decode metadata.
    """
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
    batched_data = build_motion_data_from_buffers(buffers, device=device)
    token_index_per_job = buffers.host_tensors["token_index_per_job"]
    decode_meta = build_decode_metadata(
        jobs=jobs,
        token_index_per_job=buffers.numpy_views["token_index_per_job"],
    )
    attach_decode_meta_tensors(
        decode_meta=decode_meta,
        device=device,
    )
    return batched_data, {
        "jobs": jobs,
        "token_index_per_job": token_index_per_job,
        "decode_meta": decode_meta,
    }
