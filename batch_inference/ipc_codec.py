"""IPC payload 编解码（batch_inference 专用）。

目标：
1. 减少跨进程传输时的嵌套 dict/list 层级。
2. 将索引/ID/step/token 统一为 int32，连续值统一为 float32。
3. 在 ExternalTeacher / adapter 侧恢复为现有业务结构，保持语义不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PREPARED_IPC_FORMAT = "prepared_v1"
MODEL_OUTPUTS_IPC_FORMAT = "model_outputs_v1"


def _as_int32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _as_float32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _pack_dict_int32(mapping: Dict[Any, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if not mapping:
        return _as_int32_array([]), _as_int32_array([])
    items = list(mapping.items())
    keys = _as_int32_array([int(k) for k, _ in items])
    vals = _as_int32_array([int(v) for _, v in items])
    return keys, vals


def _unpack_dict_int32(keys: np.ndarray, vals: np.ndarray) -> Dict[int, int]:
    if keys.size == 0:
        return {}
    return {int(k): int(v) for k, v in zip(keys.tolist(), vals.tolist())}


def _pack_ragged_int_lists(lists: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(lists) + 1,), dtype=np.int32)
    flat_values: List[int] = []
    cursor = 0
    for i, row in enumerate(lists):
        row_vals = [int(v) for v in row]
        flat_values.extend(row_vals)
        cursor += len(row_vals)
        offsets[i + 1] = cursor
    return _as_int32_array(flat_values), offsets


def _unpack_ragged_int_lists(flat: np.ndarray, offsets: np.ndarray) -> List[List[int]]:
    out: List[List[int]] = []
    for i in range(int(offsets.shape[0]) - 1):
        left = int(offsets[i])
        right = int(offsets[i + 1])
        out.append([int(v) for v in flat[left:right].tolist()])
    return out


def _pack_motion_array(name: str, array: Any) -> np.ndarray:
    np_arr = np.asarray(array)
    if name in ("actions", "rtgs", "timesteps"):
        return np_arr.astype(np.int32, copy=False)
    if name in ("moving_agent_mask",):
        return np_arr.astype(np.bool_, copy=False)
    if np.issubdtype(np_arr.dtype, np.floating):
        return np_arr.astype(np.float32, copy=False)
    if np.issubdtype(np_arr.dtype, np.integer):
        return np_arr.astype(np.int32, copy=False)
    return np_arr.astype(np.float32, copy=False)


def _unpack_motion_array(name: str, array: np.ndarray) -> np.ndarray:
    if name in ("actions", "rtgs", "timesteps"):
        return np.asarray(array, dtype=np.int64)
    return array


def pack_prepared(prepared: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 prepared_dict 打包为紧凑 IPC payload。"""
    if prepared is None:
        return None

    status = str(prepared.get("status", "ok"))
    packed: Dict[str, Any] = {
        "ipc_format": PREPARED_IPC_FORMAT,
        "status": status,
        "step_t": np.int32(int(prepared.get("step_t", 0))),
        "token_index": np.int32(int(prepared.get("token_index", 0))),
        "dead_ids": _as_int32_array(prepared.get("dead_ids", [])),
    }
    if status != "ok":
        return packed

    sampling = prepared.get("sampling", {})
    packed["sampling_values"] = _as_float32_array(
        [
            float(sampling.get("action_temperature", 1.0)),
            float(sampling.get("nucleus_threshold", 1.0)),
        ]
    )
    packed["sampling_flags"] = _as_int32_array(
        [1 if bool(sampling.get("nucleus_sampling", False)) else 0]
    )
    packed["default_tilt"] = _as_int32_array(prepared.get("default_tilt", (0, 0, 0)))

    tilt_by_veh_id = prepared.get("tilt_by_veh_id", {})
    tilt_ids: List[int] = []
    tilt_vals: List[Tuple[int, int, int]] = []
    for veh_id, tilt in tilt_by_veh_id.items():
        tilt_ids.append(int(veh_id))
        tilt_vals.append((int(tilt[0]), int(tilt[1]), int(tilt[2])))
    packed["tilt_veh_ids"] = _as_int32_array(tilt_ids)
    packed["tilt_values"] = _as_int32_array(tilt_vals).reshape((-1, 3)) if tilt_vals else np.zeros((0, 3), dtype=np.int32)

    veh_keys, veh_vals = _pack_dict_int32(prepared.get("veh_id_to_idx", {}))
    packed["veh_id_to_idx_keys"] = veh_keys
    packed["veh_id_to_idx_vals"] = veh_vals

    focal_batches = prepared.get("focal_batches", [])
    packed["focal_ids"] = _as_int32_array([int(fb.get("focal_id", -1)) for fb in focal_batches])
    packed["seq_len"] = _as_int32_array([int(fb.get("seq_len", 0)) for fb in focal_batches])
    packed["valid_agent_count"] = _as_int32_array(
        [int(fb.get("valid_agent_count", 0)) for fb in focal_batches]
    )
    packed["valid_road_count"] = _as_int32_array(
        [int(fb.get("valid_road_count", 0)) for fb in focal_batches]
    )
    packed["predict_rtgs"] = np.asarray(
        [bool(fb.get("predict_rtgs", True)) for fb in focal_batches],
        dtype=np.bool_,
    )

    map_keys_rows: List[List[int]] = []
    map_vals_rows: List[List[int]] = []
    data_veh_rows: List[List[int]] = []
    context_rows: List[List[int]] = []

    motion_agent_states: List[np.ndarray] = []
    motion_agent_types: List[np.ndarray] = []
    motion_goals: List[np.ndarray] = []
    motion_actions: List[np.ndarray] = []
    motion_rtgs: List[np.ndarray] = []
    motion_timesteps: List[np.ndarray] = []
    motion_moving_agent_mask: List[np.ndarray] = []
    motion_road_points: List[np.ndarray] = []
    motion_road_types: List[np.ndarray] = []

    for fb in focal_batches:
        map_dict = fb.get("new_agent_idx_dict", {})
        map_items = list(map_dict.items())
        map_keys_rows.append([int(k) for k, _ in map_items])
        map_vals_rows.append([int(v) for _, v in map_items])
        data_veh_rows.append([int(v) for v in fb.get("data_veh_ids", [])])
        context_rows.append([int(v) for v in fb.get("veh_ids_in_context", [])])

        motion_data_np = fb.get("motion_data_np", {})
        motion_agent_states.append(_pack_motion_array("agent_states", motion_data_np.get("agent_states")))
        motion_agent_types.append(_pack_motion_array("agent_types", motion_data_np.get("agent_types")))
        motion_goals.append(_pack_motion_array("goals", motion_data_np.get("goals")))
        motion_actions.append(_pack_motion_array("actions", motion_data_np.get("actions")))
        motion_rtgs.append(_pack_motion_array("rtgs", motion_data_np.get("rtgs")))
        motion_timesteps.append(_pack_motion_array("timesteps", motion_data_np.get("timesteps")))
        motion_moving_agent_mask.append(
            _pack_motion_array("moving_agent_mask", motion_data_np.get("moving_agent_mask"))
        )
        motion_road_points.append(_pack_motion_array("road_points", motion_data_np.get("road_points")))
        motion_road_types.append(_pack_motion_array("road_types", motion_data_np.get("road_types")))

    map_keys_flat, map_offsets = _pack_ragged_int_lists(map_keys_rows)
    map_vals_flat, _ = _pack_ragged_int_lists(map_vals_rows)
    packed["new_agent_idx_keys"] = map_keys_flat
    packed["new_agent_idx_vals"] = map_vals_flat
    packed["new_agent_idx_offsets"] = map_offsets

    data_flat, data_offsets = _pack_ragged_int_lists(data_veh_rows)
    packed["data_veh_ids_flat"] = data_flat
    packed["data_veh_ids_offsets"] = data_offsets

    context_flat, context_offsets = _pack_ragged_int_lists(context_rows)
    packed["veh_ids_in_context_flat"] = context_flat
    packed["veh_ids_in_context_offsets"] = context_offsets

    packed["motion_agent_states"] = motion_agent_states
    packed["motion_agent_types"] = motion_agent_types
    packed["motion_goals"] = motion_goals
    packed["motion_actions"] = motion_actions
    packed["motion_rtgs"] = motion_rtgs
    packed["motion_timesteps"] = motion_timesteps
    packed["motion_moving_agent_mask"] = motion_moving_agent_mask
    packed["motion_road_points"] = motion_road_points
    packed["motion_road_types"] = motion_road_types
    return packed


def unpack_prepared(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将紧凑 prepared payload 恢复为 ExternalTeacher 现有消费结构。"""
    if packed is None:
        return None
    if packed.get("ipc_format") != PREPARED_IPC_FORMAT:
        raise ValueError("Unexpected prepared IPC payload format.")

    status = str(packed.get("status", "ok"))
    prepared: Dict[str, Any] = {
        "status": status,
        "step_t": int(packed.get("step_t", 0)),
        "token_index": int(packed.get("token_index", 0)),
        "dead_ids": [int(v) for v in np.asarray(packed.get("dead_ids", []), dtype=np.int32).tolist()],
    }
    if status != "ok":
        return prepared

    sampling_values = np.asarray(packed.get("sampling_values", [1.0, 1.0]), dtype=np.float32)
    sampling_flags = np.asarray(packed.get("sampling_flags", [0]), dtype=np.int32)
    prepared["sampling"] = {
        "action_temperature": float(sampling_values[0]) if sampling_values.size >= 1 else 1.0,
        "nucleus_sampling": bool(int(sampling_flags[0])) if sampling_flags.size >= 1 else False,
        "nucleus_threshold": float(sampling_values[1]) if sampling_values.size >= 2 else 1.0,
    }
    default_tilt = np.asarray(packed.get("default_tilt", [0, 0, 0]), dtype=np.int32)
    prepared["default_tilt"] = (
        int(default_tilt[0]) if default_tilt.size >= 1 else 0,
        int(default_tilt[1]) if default_tilt.size >= 2 else 0,
        int(default_tilt[2]) if default_tilt.size >= 3 else 0,
    )

    tilt_ids = np.asarray(packed.get("tilt_veh_ids", []), dtype=np.int32)
    tilt_values = np.asarray(packed.get("tilt_values", []), dtype=np.int32).reshape((-1, 3))
    prepared["tilt_by_veh_id"] = {
        int(veh_id): (int(tilt[0]), int(tilt[1]), int(tilt[2]))
        for veh_id, tilt in zip(tilt_ids.tolist(), tilt_values.tolist())
    }

    prepared["veh_id_to_idx"] = _unpack_dict_int32(
        np.asarray(packed.get("veh_id_to_idx_keys", []), dtype=np.int32),
        np.asarray(packed.get("veh_id_to_idx_vals", []), dtype=np.int32),
    )

    focal_ids = np.asarray(packed.get("focal_ids", []), dtype=np.int32)
    seq_len_arr = np.asarray(packed.get("seq_len", []), dtype=np.int32)
    valid_agent_arr = np.asarray(packed.get("valid_agent_count", []), dtype=np.int32)
    valid_road_arr = np.asarray(packed.get("valid_road_count", []), dtype=np.int32)
    predict_rtgs_arr = np.asarray(packed.get("predict_rtgs", []), dtype=np.bool_)

    new_agent_keys_rows = _unpack_ragged_int_lists(
        np.asarray(packed.get("new_agent_idx_keys", []), dtype=np.int32),
        np.asarray(packed.get("new_agent_idx_offsets", [0]), dtype=np.int32),
    )
    new_agent_vals_rows = _unpack_ragged_int_lists(
        np.asarray(packed.get("new_agent_idx_vals", []), dtype=np.int32),
        np.asarray(packed.get("new_agent_idx_offsets", [0]), dtype=np.int32),
    )
    data_veh_rows = _unpack_ragged_int_lists(
        np.asarray(packed.get("data_veh_ids_flat", []), dtype=np.int32),
        np.asarray(packed.get("data_veh_ids_offsets", [0]), dtype=np.int32),
    )
    context_rows = _unpack_ragged_int_lists(
        np.asarray(packed.get("veh_ids_in_context_flat", []), dtype=np.int32),
        np.asarray(packed.get("veh_ids_in_context_offsets", [0]), dtype=np.int32),
    )

    motion_agent_states = list(packed.get("motion_agent_states", []))
    motion_agent_types = list(packed.get("motion_agent_types", []))
    motion_goals = list(packed.get("motion_goals", []))
    motion_actions = list(packed.get("motion_actions", []))
    motion_rtgs = list(packed.get("motion_rtgs", []))
    motion_timesteps = list(packed.get("motion_timesteps", []))
    motion_moving_agent_mask = list(packed.get("motion_moving_agent_mask", []))
    motion_road_points = list(packed.get("motion_road_points", []))
    motion_road_types = list(packed.get("motion_road_types", []))

    focal_batches: List[Dict[str, Any]] = []
    for i, focal_id in enumerate(focal_ids.tolist()):
        keys_row = new_agent_keys_rows[i] if i < len(new_agent_keys_rows) else []
        vals_row = new_agent_vals_rows[i] if i < len(new_agent_vals_rows) else []
        new_agent_idx_dict = {int(k): int(v) for k, v in zip(keys_row, vals_row)}

        motion_data_np = {
            "agent_states": _unpack_motion_array("agent_states", np.asarray(motion_agent_states[i])),
            "agent_types": _unpack_motion_array("agent_types", np.asarray(motion_agent_types[i])),
            "goals": _unpack_motion_array("goals", np.asarray(motion_goals[i])),
            "actions": _unpack_motion_array("actions", np.asarray(motion_actions[i])),
            "rtgs": _unpack_motion_array("rtgs", np.asarray(motion_rtgs[i])),
            "timesteps": _unpack_motion_array("timesteps", np.asarray(motion_timesteps[i])),
            "moving_agent_mask": _unpack_motion_array(
                "moving_agent_mask", np.asarray(motion_moving_agent_mask[i])
            ),
            "road_points": _unpack_motion_array("road_points", np.asarray(motion_road_points[i])),
            "road_types": _unpack_motion_array("road_types", np.asarray(motion_road_types[i])),
        }

        focal_batches.append(
            {
                "focal_id": int(focal_id),
                "motion_data_np": motion_data_np,
                "new_agent_idx_dict": new_agent_idx_dict,
                "data_veh_ids": data_veh_rows[i] if i < len(data_veh_rows) else [],
                "veh_ids_in_context": context_rows[i] if i < len(context_rows) else [],
                "seq_len": int(seq_len_arr[i]) if i < seq_len_arr.size else 0,
                "valid_agent_count": int(valid_agent_arr[i]) if i < valid_agent_arr.size else 0,
                "valid_road_count": int(valid_road_arr[i]) if i < valid_road_arr.size else 0,
                "predict_rtgs": bool(predict_rtgs_arr[i]) if i < predict_rtgs_arr.size else True,
            }
        )

    prepared["focal_batches"] = focal_batches
    return prepared


def pack_model_outputs(model_outputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 ExternalTeacher 输出打包为紧凑 IPC payload。"""
    if model_outputs is None:
        return None

    status = str(model_outputs.get("status", "ok"))
    action_results = model_outputs.get("action_results", {})
    action_veh_ids = _as_int32_array(list(action_results.keys()))
    if action_veh_ids.size > 0:
        action_values = _as_float32_array([action_results[int(k)] for k in action_veh_ids.tolist()])
    else:
        action_values = np.zeros((0, 2), dtype=np.float32)

    rtg_results = model_outputs.get("rtg_results", {})
    rtg_veh_ids = _as_int32_array(list(rtg_results.keys()))
    if rtg_veh_ids.size > 0:
        rtg_values = _as_float32_array([rtg_results[int(k)] for k in rtg_veh_ids.tolist()])
    else:
        rtg_values = np.zeros((0, 3), dtype=np.float32)

    packed = {
        "ipc_format": MODEL_OUTPUTS_IPC_FORMAT,
        "status": status,
        "env_idx": np.int32(int(model_outputs.get("env_idx", -1))),
        "step_t": np.int32(int(model_outputs.get("step_t", 0))),
        "token_index": np.int32(int(model_outputs.get("token_index", 0))),
        "action_veh_ids": action_veh_ids,
        "action_values": action_values,
        "rtg_veh_ids": rtg_veh_ids,
        "rtg_values": rtg_values,
        "processed_rtg_veh_ids": _as_int32_array(model_outputs.get("processed_rtg_veh_ids", [])),
        "dead_ids": _as_int32_array(model_outputs.get("dead_ids", [])),
        "error": model_outputs.get("error"),
    }
    return packed


def unpack_model_outputs(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将紧凑 model_outputs payload 恢复为 adapter 现有消费结构。"""
    if packed is None:
        return None
    if packed.get("ipc_format") != MODEL_OUTPUTS_IPC_FORMAT:
        raise ValueError("Unexpected model_outputs IPC payload format.")

    action_veh_ids = np.asarray(packed.get("action_veh_ids", []), dtype=np.int32)
    action_values = np.asarray(packed.get("action_values", []), dtype=np.float32).reshape((-1, 2))
    action_results = {
        int(veh_id): (float(val[0]), float(val[1]))
        for veh_id, val in zip(action_veh_ids.tolist(), action_values.tolist())
    }

    rtg_veh_ids = np.asarray(packed.get("rtg_veh_ids", []), dtype=np.int32)
    rtg_values = np.asarray(packed.get("rtg_values", []), dtype=np.float32).reshape((-1, 3))
    rtg_results = {
        int(veh_id): (float(val[0]), float(val[1]), float(val[2]))
        for veh_id, val in zip(rtg_veh_ids.tolist(), rtg_values.tolist())
    }

    return {
        "status": str(packed.get("status", "ok")),
        "env_idx": int(packed.get("env_idx", -1)),
        "step_t": int(packed.get("step_t", 0)),
        "token_index": int(packed.get("token_index", 0)),
        "action_results": action_results,
        "rtg_results": rtg_results,
        "processed_rtg_veh_ids": [
            int(v)
            for v in np.asarray(packed.get("processed_rtg_veh_ids", []), dtype=np.int32).tolist()
        ],
        "dead_ids": [int(v) for v in np.asarray(packed.get("dead_ids", []), dtype=np.int32).tolist()],
        "error": packed.get("error"),
    }

