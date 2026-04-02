"""
负责 prepared 推理负载的序列化、反序列化与共享内存资源释放。
该模块将适配器侧构造的批量输入转换成可跨进程传递的 IPC payload，并在消费后回收资源。
Handles serialization, deserialization, and shared-memory cleanup for prepared inference payloads.
Turns adapter-side batched inputs into IPC-safe payloads and releases resources after consumption.
"""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from .arrays import (
    as_float32_array,
    as_int32_array,
    as_int_list,
    pack_dict_int32,
    pack_motion_array,
    pack_motion_array_list,
    pack_ragged_int_lists,
    packed_motion_nbytes,
    unpack_dict_int32,
    unpack_motion_array,
    unpack_motion_array_list,
    unpack_ragged_int_lists,
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


def pack_prepared(prepared: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    packed["shared_timesteps"] = pack_motion_array("timesteps", prepared_typed["shared_timesteps"])
    packed["sampling_values"] = as_float32_array(
        [
            float(sampling["action_temperature"]),
            float(sampling["nucleus_threshold"]),
        ]
    )
    packed["sampling_flags"] = as_int32_array(
        [1 if bool(sampling["nucleus_sampling"]) else 0]
    )
    packed["default_tilt"] = as_int32_array(prepared_typed["default_tilt"])

    tilt_by_veh_id = prepared_typed["tilt_by_veh_id"]
    tilt_ids: List[int] = []
    tilt_vals: List[Tuple[int, int, int]] = []
    for veh_id, tilt in tilt_by_veh_id.items():
        tilt_ids.append(int(veh_id))
        tilt_vals.append((int(tilt[0]), int(tilt[1]), int(tilt[2])))
    packed["tilt_veh_ids"] = as_int32_array(tilt_ids)
    packed["tilt_values"] = (
        as_int32_array(tilt_vals).reshape((-1, 3))
        if tilt_vals
        else np.zeros((0, 3), dtype=np.int32)
    )

    veh_keys, veh_vals = pack_dict_int32(prepared_typed["veh_id_to_idx"])
    packed["veh_id_to_idx_keys"] = veh_keys
    packed["veh_id_to_idx_vals"] = veh_vals

    focal_batches = prepared_typed["focal_batches"]
    packed["focal_ids"] = as_int32_array([int(fb["focal_id"]) for fb in focal_batches])
    packed["predict_rtgs"] = np.asarray(
        [bool(fb["predict_rtgs"]) for fb in focal_batches],
        dtype=np.bool_,
    )

    map_keys_rows: List[List[int]] = []
    map_vals_rows: List[List[int]] = []
    data_veh_rows: List[List[int]] = []
    context_rows: List[List[int]] = []
    data_model_index_rows: List[List[int]] = []
    context_model_index_rows: List[List[int]] = []
    total_motion_bytes = 0
    for fb in focal_batches:
        map_items = list(fb["new_agent_idx_dict"].items())
        map_keys_rows.append([int(k) for k, _ in map_items])
        map_vals_rows.append([int(v) for _, v in map_items])
        data_veh_rows.append([int(v) for v in fb["data_veh_ids"]])
        context_rows.append([int(v) for v in fb["veh_ids_in_context"]])
        data_model_index_rows.append(
            [
                int(v)
                for v in np.asarray(
                    fb.get("data_veh_model_indices", []),
                    dtype=np.int32,
                ).tolist()
            ]
        )
        context_model_index_rows.append(
            [
                int(v)
                for v in np.asarray(
                    fb.get("context_veh_model_indices", []),
                    dtype=np.int32,
                ).tolist()
            ]
        )
        motion_data_np = fb["motion_data_np"]
        for field_name in MOTION_FIELD_NAMES:
            total_motion_bytes += packed_motion_nbytes(field_name, motion_data_np[field_name])

    motion_storage = SHM_MOTION_STORAGE if should_use_shared_memory(total_motion_bytes) else INLINE_MOTION_STORAGE
    packed["motion_storage"] = motion_storage

    map_keys_flat, map_offsets = pack_ragged_int_lists(map_keys_rows)
    map_vals_flat, _ = pack_ragged_int_lists(map_vals_rows)
    packed["new_agent_idx_keys"] = map_keys_flat
    packed["new_agent_idx_vals"] = map_vals_flat
    packed["new_agent_idx_offsets"] = map_offsets

    data_flat, data_offsets = pack_ragged_int_lists(data_veh_rows)
    packed["data_veh_ids_flat"] = data_flat
    packed["data_veh_ids_offsets"] = data_offsets

    context_flat, context_offsets = pack_ragged_int_lists(context_rows)
    packed["veh_ids_in_context_flat"] = context_flat
    packed["veh_ids_in_context_offsets"] = context_offsets

    data_model_flat, data_model_offsets = pack_ragged_int_lists(data_model_index_rows)
    packed["data_veh_model_indices_flat"] = data_model_flat
    packed["data_veh_model_indices_offsets"] = data_model_offsets

    context_model_flat, context_model_offsets = pack_ragged_int_lists(context_model_index_rows)
    packed["context_veh_model_indices_flat"] = context_model_flat
    packed["context_veh_model_indices_offsets"] = context_model_offsets

    for field_name in MOTION_FIELD_NAMES:
        packed_values = [
            pack_motion_array(field_name, fb["motion_data_np"][field_name])
            for fb in focal_batches
        ]
        flat_values, flat_offsets, flat_shapes = pack_motion_array_list(field_name, packed_values)
        packed[f"motion_{field_name}_offsets"] = flat_offsets
        packed[f"motion_{field_name}_shapes"] = flat_shapes
        if motion_storage == SHM_MOTION_STORAGE:
            packed[f"motion_{field_name}_flat"] = pack_motion_array_to_shared_memory(flat_values)
            continue
        packed[f"motion_{field_name}_flat"] = flat_values
    return packed


def unpack_prepared(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if packed is None:
        return None
    if packed.get("ipc_format") != PREPARED_IPC_FORMAT:
        raise ValueError("Unexpected prepared IPC payload format.")

    require_keys(packed, ("status", "step_t", "token_index", "dead_ids"), "packed prepared payload")
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
        "ego_context_owner_focal_id": None if owner_focal_id_value < 0 else owner_focal_id_value,
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
            "veh_id_to_idx_keys",
            "veh_id_to_idx_vals",
            "focal_ids",
            "predict_rtgs",
            "new_agent_idx_keys",
            "new_agent_idx_vals",
            "new_agent_idx_offsets",
            "data_veh_ids_flat",
            "data_veh_ids_offsets",
            "veh_ids_in_context_flat",
            "veh_ids_in_context_offsets",
            "data_veh_model_indices_flat",
            "data_veh_model_indices_offsets",
            "context_veh_model_indices_flat",
            "context_veh_model_indices_offsets",
        ),
        "packed prepared payload",
    )

    sampling_values = np.asarray(packed["sampling_values"], dtype=np.float32)
    sampling_flags = np.asarray(packed["sampling_flags"], dtype=np.int32)
    if sampling_values.shape[0] != 2 or sampling_flags.shape[0] != 1:
        raise ValueError("packed prepared payload has invalid sampling array shapes.")
    prepared["shared_timesteps"] = unpack_motion_array(
        "timesteps",
        np.asarray(packed["shared_timesteps"]),
    )
    prepared["sampling"] = {
        "action_temperature": float(sampling_values[0]),
        "nucleus_sampling": bool(int(sampling_flags[0])),
        "nucleus_threshold": float(sampling_values[1]),
    }

    default_tilt = np.asarray(packed["default_tilt"], dtype=np.int32).reshape((-1,))
    if default_tilt.shape[0] != 3:
        raise ValueError("packed prepared payload has invalid default_tilt shape.")
    prepared["default_tilt"] = (
        int(default_tilt[0]),
        int(default_tilt[1]),
        int(default_tilt[2]),
    )

    tilt_ids = np.asarray(packed["tilt_veh_ids"], dtype=np.int32)
    tilt_values = np.asarray(packed["tilt_values"], dtype=np.int32).reshape((-1, 3))
    if tilt_ids.shape[0] != tilt_values.shape[0]:
        raise ValueError("packed prepared payload tilt arrays length mismatch.")
    prepared["tilt_by_veh_id"] = {
        int(veh_id): (int(tilt[0]), int(tilt[1]), int(tilt[2]))
        for veh_id, tilt in zip(tilt_ids.tolist(), tilt_values.tolist())
    }

    prepared["veh_id_to_idx"] = unpack_dict_int32(
        np.asarray(packed["veh_id_to_idx_keys"], dtype=np.int32),
        np.asarray(packed["veh_id_to_idx_vals"], dtype=np.int32),
    )

    focal_ids = np.asarray(packed["focal_ids"], dtype=np.int32)
    predict_rtgs_arr = np.asarray(packed["predict_rtgs"], dtype=np.bool_)

    new_agent_offsets = np.asarray(packed["new_agent_idx_offsets"], dtype=np.int32)
    new_agent_keys_rows = unpack_ragged_int_lists(
        np.asarray(packed["new_agent_idx_keys"], dtype=np.int32),
        new_agent_offsets,
    )
    new_agent_vals_rows = unpack_ragged_int_lists(
        np.asarray(packed["new_agent_idx_vals"], dtype=np.int32),
        new_agent_offsets,
    )
    data_veh_rows = unpack_ragged_int_lists(
        np.asarray(packed["data_veh_ids_flat"], dtype=np.int32),
        np.asarray(packed["data_veh_ids_offsets"], dtype=np.int32),
    )
    context_rows = unpack_ragged_int_lists(
        np.asarray(packed["veh_ids_in_context_flat"], dtype=np.int32),
        np.asarray(packed["veh_ids_in_context_offsets"], dtype=np.int32),
    )
    data_model_index_rows = unpack_ragged_int_lists(
        np.asarray(packed["data_veh_model_indices_flat"], dtype=np.int32),
        np.asarray(packed["data_veh_model_indices_offsets"], dtype=np.int32),
    )
    context_model_index_rows = unpack_ragged_int_lists(
        np.asarray(packed["context_veh_model_indices_flat"], dtype=np.int32),
        np.asarray(packed["context_veh_model_indices_offsets"], dtype=np.int32),
    )

    row_count = int(focal_ids.shape[0])
    if not (
        row_count
        == predict_rtgs_arr.shape[0]
        == len(new_agent_keys_rows)
        == len(new_agent_vals_rows)
        == len(data_veh_rows)
        == len(context_rows)
        == len(data_model_index_rows)
        == len(context_model_index_rows)
    ):
        raise ValueError("packed prepared payload ragged row counts mismatch.")

    motion_storage = str(packed.get("motion_storage", INLINE_MOTION_STORAGE))
    motion_arrays: Dict[str, List[np.ndarray]] = {}
    for field_name in MOTION_FIELD_NAMES:
        flat_key = f"motion_{field_name}_flat"
        offsets_key = f"motion_{field_name}_offsets"
        shapes_key = f"motion_{field_name}_shapes"
        require_keys(
            packed,
            (flat_key, offsets_key, shapes_key),
            "packed prepared payload",
        )

    shm_handles: List[shared_memory.SharedMemory] = []
    try:
        for field_name in MOTION_FIELD_NAMES:
            flat_key = f"motion_{field_name}_flat"
            if motion_storage == SHM_MOTION_STORAGE:
                flat_array, shm_handle = unpack_motion_array_from_shared_memory(packed[flat_key])
                shm_handles.append(shm_handle)
            else:
                flat_array = np.asarray(packed[flat_key])
            motion_arrays[field_name] = unpack_motion_array_list(
                field_name,
                flat=flat_array,
                offsets=np.asarray(packed[f"motion_{field_name}_offsets"], dtype=np.int32),
                shapes=np.asarray(packed[f"motion_{field_name}_shapes"], dtype=np.int32),
            )
            if len(motion_arrays[field_name]) != row_count:
                raise ValueError(
                    f"packed prepared payload 'motion_{field_name}' row count mismatch."
                )

        focal_batches: List[Dict[str, Any]] = []
        for i, focal_id in enumerate(focal_ids.tolist()):
            keys_row = new_agent_keys_rows[i]
            vals_row = new_agent_vals_rows[i]
            new_agent_idx_dict = {int(k): int(v) for k, v in zip(keys_row, vals_row)}

            motion_data_np: Dict[str, np.ndarray] = {}
            for field_name in MOTION_FIELD_NAMES:
                motion_data_np[field_name] = motion_arrays[field_name][i]

            focal_batches.append(
                {
                    "focal_id": int(focal_id),
                    "motion_data_np": motion_data_np,
                    "new_agent_idx_dict": new_agent_idx_dict,
                    "data_veh_ids": data_veh_rows[i],
                    "veh_ids_in_context": context_rows[i],
                    "data_veh_model_indices": np.asarray(
                        data_model_index_rows[i],
                        dtype=np.int64,
                    ),
                    "context_veh_model_indices": np.asarray(
                        context_model_index_rows[i],
                        dtype=np.int64,
                    ),
                    "predict_rtgs": bool(predict_rtgs_arr[i]),
                }
            )
    except Exception:
        for shm_handle in shm_handles:
            close_and_unlink_shared_memory(shm_handle)
        raise

    prepared["focal_batches"] = focal_batches
    prepared["_ipc_shm_handles"] = shm_handles
    return prepared


def release_prepared_payload(prepared: Optional[Dict[str, Any]]) -> None:
    if prepared is None:
        return
    shm_handles = prepared.pop("_ipc_shm_handles", [])
    for shm_handle in shm_handles:
        close_and_unlink_shared_memory(shm_handle)
