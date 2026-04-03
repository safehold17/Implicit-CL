"""
负责 prepared 推理负载的序列化、反序列化与共享内存资源释放。
该模块将适配器侧构造的 flat focal batch 输入转换成可跨进程传递的 IPC payload，
并提供按 focal 索引读取 batched motion / ragged metadata 的辅助函数。
Handles serialization, deserialization, and shared-memory cleanup for prepared inference payloads.
It turns adapter-side flat focal-batch inputs into IPC-safe payloads and provides helpers for
reading batched motion tensors and ragged metadata by focal index.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from .arrays import (
    as_float32_array,
    as_int32_array,
    as_int_list,
    pack_motion_array,
    packed_motion_nbytes,
    unpack_motion_array,
)
from .schema import (
    INLINE_MOTION_STORAGE,
    MOTION_FIELD_NAMES,
    PREPARED_IPC_FORMAT,
    SHM_MOTION_STORAGE,
    PreparedPayload,
)
from .shared_memory import (
    close_and_unlink_shared_memory,
    pack_motion_array_to_shared_memory,
    should_use_shared_memory,
    unpack_motion_array_from_shared_memory,
)
from .validate import require_keys, require_valid_status, validate_prepared_payload


def _pack_tilt_mapping(
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """将按车辆组织的 tilt 映射压成两个定长数组。

    Pack the per-vehicle tilt mapping into two dense arrays for IPC transport.
    """
    if not tilt_by_veh_id:
        return as_int32_array([]), np.zeros((0, 3), dtype=np.int32)
    ordered_ids = [int(veh_id) for veh_id in tilt_by_veh_id.keys()]
    tilt_values = np.asarray(
        [tilt_by_veh_id[veh_id] for veh_id in ordered_ids],
        dtype=np.int32,
    ).reshape((-1, 3))
    return as_int32_array(ordered_ids), tilt_values


def _unpack_tilt_mapping(
    tilt_ids_payload: Any,
    tilt_values_payload: Any,
) -> Dict[int, Tuple[int, int, int]]:
    """将 IPC 中的 tilt 数组还原为按车辆索引的映射。

    Reconstruct the per-vehicle tilt mapping from its IPC array representation.
    """
    tilt_ids = np.asarray(tilt_ids_payload, dtype=np.int32).reshape((-1,))
    tilt_values = np.asarray(tilt_values_payload, dtype=np.int32).reshape((-1, 3))
    if tilt_ids.shape[0] != tilt_values.shape[0]:
        raise ValueError("packed prepared payload tilt arrays length mismatch.")
    return {
        int(veh_id): (int(tilt[0]), int(tilt[1]), int(tilt[2]))
        for veh_id, tilt in zip(tilt_ids.tolist(), tilt_values.tolist())
    }


def _pack_motion_batches(
    motion_data_np: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """打包 batched motion 张量，并在大负载时切换到共享内存。

    Pack batched motion tensors and switch to shared memory when the payload is large.
    """
    total_motion_bytes = sum(
        packed_motion_nbytes(field_name, motion_data_np[field_name])
        for field_name in MOTION_FIELD_NAMES
    )
    motion_storage = (
        SHM_MOTION_STORAGE
        if should_use_shared_memory(total_motion_bytes)
        else INLINE_MOTION_STORAGE
    )
    packed_motion: Dict[str, Any] = {}
    for field_name in MOTION_FIELD_NAMES:
        packed_array = pack_motion_array(field_name, motion_data_np[field_name])
        if motion_storage == SHM_MOTION_STORAGE:
            packed_motion[field_name] = pack_motion_array_to_shared_memory(packed_array)
        else:
            packed_motion[field_name] = packed_array
    return motion_storage, packed_motion


def _unpack_motion_batches(
    packed: Dict[str, Any],
    motion_storage: str,
) -> Tuple[Dict[str, np.ndarray], list[shared_memory.SharedMemory]]:
    """恢复 batched motion 张量，并返回需要在消费后释放的 SHM 句柄。

    Restore batched motion tensors and return the SHM handles that must be released afterwards.
    """
    motion_data_np: Dict[str, np.ndarray] = {}
    shm_handles: list[shared_memory.SharedMemory] = []
    try:
        for field_name in MOTION_FIELD_NAMES:
            payload_key = f"motion_{field_name}"
            if motion_storage == SHM_MOTION_STORAGE:
                restored_array, shm_handle = unpack_motion_array_from_shared_memory(
                    packed[payload_key]
                )
                shm_handles.append(shm_handle)
            else:
                restored_array = np.asarray(packed[payload_key])
            motion_data_np[field_name] = unpack_motion_array(field_name, restored_array)
        return motion_data_np, shm_handles
    except Exception:
        for shm_handle in shm_handles:
            close_and_unlink_shared_memory(shm_handle)
        raise


def _get_ragged_row(
    flat_values: Any,
    offsets: Any,
    focal_idx: int,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """按 focal 索引读取一行 ragged metadata，并返回对应的一维视图。

    Read one ragged metadata row for a focal index and return the corresponding 1D view.
    """
    flat_array = np.asarray(flat_values, dtype=dtype).reshape((-1,))
    offsets_array = np.asarray(offsets, dtype=np.int64).reshape((-1,))
    left = int(offsets_array[focal_idx])
    right = int(offsets_array[focal_idx + 1])
    return flat_array[left:right]


def get_prepared_focal_count(prepared: Dict[str, Any]) -> int:
    """返回当前 prepared 中 focal batch 的数量。

    Return the number of focal batches stored in the prepared payload.
    """
    return int(np.asarray(prepared["focal_ids"], dtype=np.int64).shape[0])


def get_prepared_focal_id(prepared: Dict[str, Any], focal_idx: int) -> int:
    """读取指定 focal 索引对应的 focal_id。

    Read the focal_id for the given focal index.
    """
    return int(np.asarray(prepared["focal_ids"], dtype=np.int64)[focal_idx])


def get_prepared_focal_predict_rtgs(prepared: Dict[str, Any], focal_idx: int) -> bool:
    """读取指定 focal 是否启用 RTG 预测。

    Read whether the given focal index should run RTG prediction.
    """
    return bool(np.asarray(prepared["predict_rtgs"], dtype=np.bool_)[focal_idx])


def get_prepared_focal_motion_data(prepared: Dict[str, Any], focal_idx: int) -> Dict[str, np.ndarray]:
    """按 focal 索引返回一组 motion 张量视图。

    Return the motion tensor views for one focal index.
    """
    motion_batches = prepared["motion_data_np"]
    return {
        field_name: np.asarray(motion_batches[field_name])[focal_idx]
        for field_name in MOTION_FIELD_NAMES
    }


def get_prepared_focal_data_veh_ids(prepared: Dict[str, Any], focal_idx: int) -> np.ndarray:
    """读取指定 focal 的受控车辆 ID 列表。

    Read the controlled vehicle ids for the given focal index.
    """
    return _get_ragged_row(
        prepared["data_veh_ids_flat"],
        prepared["data_veh_ids_offsets"],
        focal_idx,
        dtype=np.int64,
    )


def get_prepared_focal_context_veh_ids(prepared: Dict[str, Any], focal_idx: int) -> np.ndarray:
    """读取指定 focal 的上下文车辆 ID 列表。

    Read the context vehicle ids for the given focal index.
    """
    return _get_ragged_row(
        prepared["veh_ids_in_context_flat"],
        prepared["veh_ids_in_context_offsets"],
        focal_idx,
        dtype=np.int64,
    )


def get_prepared_focal_data_model_indices(prepared: Dict[str, Any], focal_idx: int) -> np.ndarray:
    """读取指定 focal 的 action 目标车辆在模型张量中的索引。

    Read the model indices for action-target vehicles at the given focal index.
    """
    return _get_ragged_row(
        prepared["data_veh_model_indices_flat"],
        prepared["data_veh_model_indices_offsets"],
        focal_idx,
        dtype=np.int64,
    )


def get_prepared_focal_context_model_indices(prepared: Dict[str, Any], focal_idx: int) -> np.ndarray:
    """读取指定 focal 的 RTG 上下文车辆在模型张量中的索引。

    Read the model indices for RTG-context vehicles at the given focal index.
    """
    return _get_ragged_row(
        prepared["context_veh_model_indices_flat"],
        prepared["context_veh_model_indices_offsets"],
        focal_idx,
        dtype=np.int64,
    )


def pack_prepared(prepared: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 flat prepared 结构打包为 IPC payload。

    Pack the flat prepared structure into its IPC payload representation.
    """
    if prepared is None:
        return None

    validate_prepared_payload(prepared)
    prepared_typed = cast(PreparedPayload, prepared)
    status = require_valid_status(prepared_typed["status"], "prepared")

    packed: Dict[str, Any] = {
        "ipc_format": PREPARED_IPC_FORMAT,
        "status": status,
        "step_t": np.int32(int(prepared_typed["step_t"])),
        "token_index": np.int32(int(prepared_typed["token_index"])),
        "dead_ids": as_int32_array(prepared_typed["dead_ids"]),
        "ego_id": np.int32(int(prepared_typed.get("ego_id", -1) or -1)),
        "ego_context_owner_focal_id": np.int32(
            int(prepared_typed.get("ego_context_owner_focal_id", -1) or -1)
        ),
        "ego_reweight_tilt": as_int32_array(
            prepared_typed.get("ego_reweight_tilt", (0, 0, 0))
        ),
        "delayed_ego_action_scale": np.float32(
            float(prepared_typed.get("delayed_ego_action_scale", 1.0))
        ),
    }
    sampling_seed = prepared_typed.get("sampling_seed")
    if sampling_seed is not None:
        packed["sampling_seed"] = np.uint64(int(sampling_seed))
    if status == "skip":
        return packed

    sampling = prepared_typed["sampling"]
    tilt_ids, tilt_values = _pack_tilt_mapping(prepared_typed["tilt_by_veh_id"])
    motion_storage, packed_motion = _pack_motion_batches(prepared_typed["motion_data_np"])

    packed.update(
        {
            "shared_timesteps": pack_motion_array(
                "timesteps",
                prepared_typed["shared_timesteps"],
            ),
            "sampling_values": as_float32_array(
                [
                    float(sampling["action_temperature"]),
                    float(sampling["nucleus_threshold"]),
                ]
            ),
            "sampling_flags": as_int32_array(
                [1 if bool(sampling["nucleus_sampling"]) else 0]
            ),
            "default_tilt": as_int32_array(prepared_typed["default_tilt"]),
            "tilt_veh_ids": tilt_ids,
            "tilt_values": tilt_values,
            "focal_ids": as_int32_array(prepared_typed["focal_ids"]),
            "predict_rtgs": np.asarray(prepared_typed["predict_rtgs"], dtype=np.bool_),
            "data_veh_ids_flat": as_int32_array(prepared_typed["data_veh_ids_flat"]),
            "data_veh_ids_offsets": as_int32_array(prepared_typed["data_veh_ids_offsets"]),
            "veh_ids_in_context_flat": as_int32_array(prepared_typed["veh_ids_in_context_flat"]),
            "veh_ids_in_context_offsets": as_int32_array(
                prepared_typed["veh_ids_in_context_offsets"]
            ),
            "data_veh_model_indices_flat": as_int32_array(
                prepared_typed["data_veh_model_indices_flat"]
            ),
            "data_veh_model_indices_offsets": as_int32_array(
                prepared_typed["data_veh_model_indices_offsets"]
            ),
            "context_veh_model_indices_flat": as_int32_array(
                prepared_typed["context_veh_model_indices_flat"]
            ),
            "context_veh_model_indices_offsets": as_int32_array(
                prepared_typed["context_veh_model_indices_offsets"]
            ),
            "motion_storage": motion_storage,
        }
    )
    for field_name, payload in packed_motion.items():
        packed[f"motion_{field_name}"] = payload
    return packed


def unpack_prepared(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 IPC payload 还原为 flat prepared 结构。

    Unpack the IPC payload back into the flat prepared structure.
    """
    if packed is None:
        return None
    if packed.get("ipc_format") != PREPARED_IPC_FORMAT:
        raise ValueError("Unexpected prepared IPC payload format.")

    require_keys(
        packed,
        ("status", "step_t", "token_index", "dead_ids"),
        "packed prepared payload",
    )
    ego_id_value = int(np.int32(packed.get("ego_id", -1)))
    owner_focal_id_value = int(np.int32(packed.get("ego_context_owner_focal_id", -1)))
    ego_reweight_tilt_arr = np.asarray(
        packed.get("ego_reweight_tilt", np.asarray([0, 0, 0], dtype=np.int32)),
        dtype=np.int32,
    ).reshape((-1,))
    if ego_reweight_tilt_arr.shape[0] != 3:
        raise ValueError("packed prepared payload has invalid ego_reweight_tilt shape.")
    status = require_valid_status(packed["status"], "packed prepared payload")

    prepared: Dict[str, Any] = {
        "status": status,
        "step_t": int(packed["step_t"]),
        "token_index": int(packed["token_index"]),
        "dead_ids": as_int_list(packed["dead_ids"]),
        "ego_id": None if ego_id_value < 0 else ego_id_value,
        "ego_context_owner_focal_id": None
        if owner_focal_id_value < 0
        else owner_focal_id_value,
        "ego_reweight_tilt": (
            int(ego_reweight_tilt_arr[0]),
            int(ego_reweight_tilt_arr[1]),
            int(ego_reweight_tilt_arr[2]),
        ),
        "delayed_ego_action_scale": float(
            np.float32(packed.get("delayed_ego_action_scale", 1.0))
        ),
    }
    if "sampling_seed" in packed:
        prepared["sampling_seed"] = int(packed["sampling_seed"])
    if status == "skip":
        return prepared

    require_keys(
        packed,
        (
            "shared_timesteps",
            "sampling_values",
            "sampling_flags",
            "default_tilt",
            "tilt_veh_ids",
            "tilt_values",
            "focal_ids",
            "predict_rtgs",
            "data_veh_ids_flat",
            "data_veh_ids_offsets",
            "veh_ids_in_context_flat",
            "veh_ids_in_context_offsets",
            "data_veh_model_indices_flat",
            "data_veh_model_indices_offsets",
            "context_veh_model_indices_flat",
            "context_veh_model_indices_offsets",
            "motion_storage",
        )
        + tuple(f"motion_{field_name}" for field_name in MOTION_FIELD_NAMES),
        "packed prepared payload",
    )

    sampling_values = np.asarray(packed["sampling_values"], dtype=np.float32)
    sampling_flags = np.asarray(packed["sampling_flags"], dtype=np.int32)
    if sampling_values.shape[0] != 2 or sampling_flags.shape[0] != 1:
        raise ValueError("packed prepared payload has invalid sampling array shapes.")
    default_tilt = np.asarray(packed["default_tilt"], dtype=np.int32).reshape((-1,))
    if default_tilt.shape[0] != 3:
        raise ValueError("packed prepared payload has invalid default_tilt shape.")

    motion_storage = str(packed.get("motion_storage", INLINE_MOTION_STORAGE))
    motion_data_np, shm_handles = _unpack_motion_batches(packed, motion_storage)

    prepared.update(
        {
            "shared_timesteps": unpack_motion_array(
                "timesteps",
                np.asarray(packed["shared_timesteps"]),
            ),
            "sampling": {
                "action_temperature": float(sampling_values[0]),
                "nucleus_sampling": bool(int(sampling_flags[0])),
                "nucleus_threshold": float(sampling_values[1]),
            },
            "default_tilt": (
                int(default_tilt[0]),
                int(default_tilt[1]),
                int(default_tilt[2]),
            ),
            "tilt_by_veh_id": _unpack_tilt_mapping(
                packed["tilt_veh_ids"],
                packed["tilt_values"],
            ),
            "focal_ids": np.asarray(packed["focal_ids"], dtype=np.int64),
            "predict_rtgs": np.asarray(packed["predict_rtgs"], dtype=np.bool_),
            "data_veh_ids_flat": np.asarray(packed["data_veh_ids_flat"], dtype=np.int64),
            "data_veh_ids_offsets": np.asarray(
                packed["data_veh_ids_offsets"],
                dtype=np.int64,
            ),
            "veh_ids_in_context_flat": np.asarray(
                packed["veh_ids_in_context_flat"],
                dtype=np.int64,
            ),
            "veh_ids_in_context_offsets": np.asarray(
                packed["veh_ids_in_context_offsets"],
                dtype=np.int64,
            ),
            "data_veh_model_indices_flat": np.asarray(
                packed["data_veh_model_indices_flat"],
                dtype=np.int64,
            ),
            "data_veh_model_indices_offsets": np.asarray(
                packed["data_veh_model_indices_offsets"],
                dtype=np.int64,
            ),
            "context_veh_model_indices_flat": np.asarray(
                packed["context_veh_model_indices_flat"],
                dtype=np.int64,
            ),
            "context_veh_model_indices_offsets": np.asarray(
                packed["context_veh_model_indices_offsets"],
                dtype=np.int64,
            ),
            "motion_data_np": motion_data_np,
            "_ipc_shm_handles": shm_handles,
        }
    )
    return prepared


def release_prepared_payload(prepared: Optional[Dict[str, Any]]) -> None:
    """释放 `unpack_prepared()` 恢复出的共享内存句柄。

    Release the shared-memory handles restored by `unpack_prepared()`.
    """
    if prepared is None:
        return
    shm_handles = prepared.pop("_ipc_shm_handles", [])
    for shm_handle in shm_handles:
        close_and_unlink_shared_memory(shm_handle)
