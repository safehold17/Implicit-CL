"""IPC payload 编解码（batch_inference 专用）。

目标：
1. 减少跨进程传输时的嵌套 dict/list 层级。
2. 将索引/ID/step/token 统一为 int32，连续值统一为 float32。
3. 在 ExternalTeacher / adapter 侧恢复为现有业务结构，保持语义不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np

PREPARED_IPC_FORMAT = "prepared_v1"
MODEL_OUTPUTS_IPC_FORMAT = "model_outputs_v1"
VALID_STATUS_VALUES = {"ok", "skip"}
MOTION_FIELD_NAMES = (
    "agent_states",
    "agent_types",
    "goals",
    "actions",
    "rtgs",
    "timesteps",
    "moving_agent_mask",
    "road_points",
    "road_types",
)


class SamplingPayload(TypedDict):
    action_temperature: float
    nucleus_sampling: bool
    nucleus_threshold: float


class FocalBatchPayload(TypedDict):
    focal_id: int
    motion_data_np: Dict[str, Any]
    new_agent_idx_dict: Dict[int, int]
    data_veh_ids: List[int]
    veh_ids_in_context: List[int]
    predict_rtgs: bool


class PreparedPayload(TypedDict):
    status: str
    step_t: int
    token_index: int
    dead_ids: List[int]
    sampling: SamplingPayload
    default_tilt: Tuple[int, int, int]
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]]
    veh_id_to_idx: Dict[int, int]
    focal_batches: List[FocalBatchPayload]


class ModelOutputsPayload(TypedDict):
    status: str
    env_idx: int
    step_t: int
    token_index: int
    action_results: Dict[int, Tuple[float, float]]
    rtg_results: Dict[int, Tuple[float, float, float]]
    processed_rtg_veh_ids: List[int]
    dead_ids: List[int]


def _as_int32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _as_float32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _as_int_list(values: Any) -> List[int]:
    return [int(v) for v in np.asarray(values, dtype=np.int32).tolist()]


def _pack_result_map(
    result_map: Dict[int, Tuple[float, ...]],
    value_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    veh_ids = _as_int32_array(list(result_map.keys()))
    if veh_ids.size == 0:
        return veh_ids, np.zeros((0, value_width), dtype=np.float32)
    values = _as_float32_array([result_map[int(k)] for k in veh_ids.tolist()]).reshape((-1, value_width))
    return veh_ids, values


def _unpack_result_map(
    veh_ids_payload: Any,
    values_payload: Any,
    value_width: int,
    field_name: str,
) -> Dict[int, Tuple[float, ...]]:
    veh_ids = np.asarray(veh_ids_payload, dtype=np.int32)
    values = np.asarray(values_payload, dtype=np.float32).reshape((-1, value_width))
    if veh_ids.shape[0] != values.shape[0]:
        raise ValueError(f"packed model_outputs payload {field_name} arrays length mismatch.")
    return {
        int(veh_id): tuple(float(component) for component in val)
        for veh_id, val in zip(veh_ids.tolist(), values.tolist())
    }


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


def _require_keys(payload: Dict[str, Any], required_keys: Sequence[str], payload_name: str) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def _require_valid_status(status: Any, payload_name: str) -> str:
    status_str = str(status)
    if status_str not in VALID_STATUS_VALUES:
        raise ValueError(
            f"{payload_name} has invalid status='{status_str}', expected one of {sorted(VALID_STATUS_VALUES)}"
        )
    return status_str


def _validate_focal_batch_payload(focal_batch: Dict[str, Any], index: int) -> None:
    _require_keys(
        focal_batch,
        (
            "focal_id",
            "motion_data_np",
            "new_agent_idx_dict",
            "data_veh_ids",
            "veh_ids_in_context",
            "predict_rtgs",
        ),
        f"prepared.focal_batches[{index}]",
    )
    motion_data_np = focal_batch["motion_data_np"]
    if not isinstance(motion_data_np, dict):
        raise ValueError(f"prepared.focal_batches[{index}].motion_data_np must be a dict.")
    _require_keys(
        motion_data_np,
        MOTION_FIELD_NAMES,
        f"prepared.focal_batches[{index}].motion_data_np",
    )


def validate_prepared_payload(prepared: Dict[str, Any]) -> None:
    _require_keys(prepared, ("status", "step_t", "token_index", "dead_ids"), "prepared")
    status = _require_valid_status(prepared["status"], "prepared")
    if status == "skip":
        return

    _require_keys(
        prepared,
        (
            "sampling",
            "default_tilt",
            "tilt_by_veh_id",
            "veh_id_to_idx",
            "focal_batches",
        ),
        "prepared",
    )
    sampling = prepared["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("prepared['sampling'] must be a dict.")
    _require_keys(
        sampling,
        ("action_temperature", "nucleus_sampling", "nucleus_threshold"),
        "prepared['sampling']",
    )

    focal_batches = prepared["focal_batches"]
    if not isinstance(focal_batches, list):
        raise ValueError("prepared['focal_batches'] must be a list.")
    for i, focal_batch in enumerate(focal_batches):
        if not isinstance(focal_batch, dict):
            raise ValueError(f"prepared.focal_batches[{i}] must be a dict.")
        _validate_focal_batch_payload(focal_batch, i)


def validate_model_outputs_payload(model_outputs: Dict[str, Any]) -> None:
    _require_keys(
        model_outputs,
        (
            "status",
            "env_idx",
            "step_t",
            "token_index",
            "action_results",
            "rtg_results",
            "processed_rtg_veh_ids",
            "dead_ids",
        ),
        "model_outputs",
    )
    _require_valid_status(model_outputs["status"], "model_outputs")


def pack_prepared(prepared: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 prepared_dict 打包为紧凑 IPC payload。"""
    if prepared is None:
        return None

    validate_prepared_payload(prepared)
    prepared_typed = cast(PreparedPayload, prepared)
    status = _require_valid_status(prepared_typed["status"], "prepared")

    packed: Dict[str, Any] = {
        "ipc_format": PREPARED_IPC_FORMAT,
        "status": status,
        "step_t": np.int32(int(prepared_typed["step_t"])),
        "token_index": np.int32(int(prepared_typed["token_index"])),
        "dead_ids": _as_int32_array(prepared_typed["dead_ids"]),
    }
    if status == "skip":
        return packed

    sampling = prepared_typed["sampling"]
    packed["sampling_values"] = _as_float32_array(
        [
            float(sampling["action_temperature"]),
            float(sampling["nucleus_threshold"]),
        ]
    )
    packed["sampling_flags"] = _as_int32_array(
        [1 if bool(sampling["nucleus_sampling"]) else 0]
    )
    packed["default_tilt"] = _as_int32_array(prepared_typed["default_tilt"])

    tilt_by_veh_id = prepared_typed["tilt_by_veh_id"]
    tilt_ids: List[int] = []
    tilt_vals: List[Tuple[int, int, int]] = []
    for veh_id, tilt in tilt_by_veh_id.items():
        tilt_ids.append(int(veh_id))
        tilt_vals.append((int(tilt[0]), int(tilt[1]), int(tilt[2])))
    packed["tilt_veh_ids"] = _as_int32_array(tilt_ids)
    packed["tilt_values"] = (
        _as_int32_array(tilt_vals).reshape((-1, 3))
        if tilt_vals
        else np.zeros((0, 3), dtype=np.int32)
    )

    veh_keys, veh_vals = _pack_dict_int32(prepared_typed["veh_id_to_idx"])
    packed["veh_id_to_idx_keys"] = veh_keys
    packed["veh_id_to_idx_vals"] = veh_vals

    focal_batches = prepared_typed["focal_batches"]
    packed["focal_ids"] = _as_int32_array([int(fb["focal_id"]) for fb in focal_batches])
    packed["predict_rtgs"] = np.asarray(
        [bool(fb["predict_rtgs"]) for fb in focal_batches],
        dtype=np.bool_,
    )

    map_keys_rows: List[List[int]] = []
    map_vals_rows: List[List[int]] = []
    data_veh_rows: List[List[int]] = []
    context_rows: List[List[int]] = []
    motion_fields = {name: [] for name in MOTION_FIELD_NAMES}

    for fb in focal_batches:
        map_items = list(fb["new_agent_idx_dict"].items())
        map_keys_rows.append([int(k) for k, _ in map_items])
        map_vals_rows.append([int(v) for _, v in map_items])
        data_veh_rows.append([int(v) for v in fb["data_veh_ids"]])
        context_rows.append([int(v) for v in fb["veh_ids_in_context"]])

        motion_data_np = fb["motion_data_np"]
        for field_name in MOTION_FIELD_NAMES:
            motion_fields[field_name].append(
                _pack_motion_array(field_name, motion_data_np[field_name])
            )

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

    for field_name, values in motion_fields.items():
        packed[f"motion_{field_name}"] = values
    return packed


def unpack_prepared(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将紧凑 prepared payload 恢复为 ExternalTeacher 现有消费结构。"""
    if packed is None:
        return None
    if packed.get("ipc_format") != PREPARED_IPC_FORMAT:
        raise ValueError("Unexpected prepared IPC payload format.")

    _require_keys(packed, ("status", "step_t", "token_index", "dead_ids"), "packed prepared payload")
    status = _require_valid_status(packed["status"], "packed prepared payload")
    prepared: Dict[str, Any] = {
        "status": status,
        "step_t": int(packed["step_t"]),
        "token_index": int(packed["token_index"]),
        "dead_ids": _as_int_list(packed["dead_ids"]),
    }
    if status == "skip":
        return prepared

    _require_keys(
        packed,
        (
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
        ),
        "packed prepared payload",
    )

    sampling_values = np.asarray(packed["sampling_values"], dtype=np.float32)
    sampling_flags = np.asarray(packed["sampling_flags"], dtype=np.int32)
    if sampling_values.shape[0] != 2 or sampling_flags.shape[0] != 1:
        raise ValueError("packed prepared payload has invalid sampling array shapes.")
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

    prepared["veh_id_to_idx"] = _unpack_dict_int32(
        np.asarray(packed["veh_id_to_idx_keys"], dtype=np.int32),
        np.asarray(packed["veh_id_to_idx_vals"], dtype=np.int32),
    )

    focal_ids = np.asarray(packed["focal_ids"], dtype=np.int32)
    predict_rtgs_arr = np.asarray(packed["predict_rtgs"], dtype=np.bool_)

    new_agent_offsets = np.asarray(packed["new_agent_idx_offsets"], dtype=np.int32)
    new_agent_keys_rows = _unpack_ragged_int_lists(
        np.asarray(packed["new_agent_idx_keys"], dtype=np.int32),
        new_agent_offsets,
    )
    new_agent_vals_rows = _unpack_ragged_int_lists(
        np.asarray(packed["new_agent_idx_vals"], dtype=np.int32),
        new_agent_offsets,
    )
    data_veh_rows = _unpack_ragged_int_lists(
        np.asarray(packed["data_veh_ids_flat"], dtype=np.int32),
        np.asarray(packed["data_veh_ids_offsets"], dtype=np.int32),
    )
    context_rows = _unpack_ragged_int_lists(
        np.asarray(packed["veh_ids_in_context_flat"], dtype=np.int32),
        np.asarray(packed["veh_ids_in_context_offsets"], dtype=np.int32),
    )

    row_count = len(focal_ids.tolist())
    if not (
        row_count
        == predict_rtgs_arr.shape[0]
        == len(new_agent_keys_rows)
        == len(new_agent_vals_rows)
        == len(data_veh_rows)
        == len(context_rows)
    ):
        raise ValueError("packed prepared payload ragged row counts mismatch.")

    motion_arrays: Dict[str, List[np.ndarray]] = {}
    for field_name in MOTION_FIELD_NAMES:
        key = f"motion_{field_name}"
        if key not in packed:
            raise ValueError(f"packed prepared payload missing required key '{key}'.")
        values = list(packed[key])
        if len(values) != row_count:
            raise ValueError(f"packed prepared payload '{key}' row count mismatch.")
        motion_arrays[field_name] = values

    focal_batches: List[Dict[str, Any]] = []
    for i, focal_id in enumerate(focal_ids.tolist()):
        keys_row = new_agent_keys_rows[i]
        vals_row = new_agent_vals_rows[i]
        new_agent_idx_dict = {int(k): int(v) for k, v in zip(keys_row, vals_row)}

        motion_data_np = {
            field_name: _unpack_motion_array(
                field_name,
                np.asarray(motion_arrays[field_name][i]),
            )
            for field_name in MOTION_FIELD_NAMES
        }

        focal_batches.append(
            {
                "focal_id": int(focal_id),
                "motion_data_np": motion_data_np,
                "new_agent_idx_dict": new_agent_idx_dict,
                "data_veh_ids": data_veh_rows[i],
                "veh_ids_in_context": context_rows[i],
                "predict_rtgs": bool(predict_rtgs_arr[i]),
            }
        )

    prepared["focal_batches"] = focal_batches
    return prepared


def pack_model_outputs(model_outputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 ExternalTeacher 输出打包为紧凑 IPC payload。"""
    if model_outputs is None:
        return None

    validate_model_outputs_payload(model_outputs)
    model_outputs_typed = cast(ModelOutputsPayload, model_outputs)
    status = _require_valid_status(model_outputs_typed["status"], "model_outputs")

    action_veh_ids, action_values = _pack_result_map(
        model_outputs_typed["action_results"],
        value_width=2,
    )
    rtg_veh_ids, rtg_values = _pack_result_map(
        model_outputs_typed["rtg_results"],
        value_width=3,
    )

    return {
        "ipc_format": MODEL_OUTPUTS_IPC_FORMAT,
        "status": status,
        "env_idx": np.int32(int(model_outputs_typed["env_idx"])),
        "step_t": np.int32(int(model_outputs_typed["step_t"])),
        "token_index": np.int32(int(model_outputs_typed["token_index"])),
        "action_veh_ids": action_veh_ids,
        "action_values": action_values,
        "rtg_veh_ids": rtg_veh_ids,
        "rtg_values": rtg_values,
        "processed_rtg_veh_ids": _as_int32_array(model_outputs_typed["processed_rtg_veh_ids"]),
        "dead_ids": _as_int32_array(model_outputs_typed["dead_ids"]),
    }


def unpack_model_outputs(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将紧凑 model_outputs payload 恢复为 adapter 现有消费结构。"""
    if packed is None:
        return None
    if packed.get("ipc_format") != MODEL_OUTPUTS_IPC_FORMAT:
        raise ValueError("Unexpected model_outputs IPC payload format.")

    _require_keys(
        packed,
        (
            "status",
            "env_idx",
            "step_t",
            "token_index",
            "action_veh_ids",
            "action_values",
            "rtg_veh_ids",
            "rtg_values",
            "processed_rtg_veh_ids",
            "dead_ids",
        ),
        "packed model_outputs payload",
    )
    status = _require_valid_status(packed["status"], "packed model_outputs payload")

    action_results = _unpack_result_map(
        packed["action_veh_ids"],
        packed["action_values"],
        value_width=2,
        field_name="action",
    )
    rtg_results = _unpack_result_map(
        packed["rtg_veh_ids"],
        packed["rtg_values"],
        value_width=3,
        field_name="rtg",
    )

    return {
        "status": status,
        "env_idx": int(packed["env_idx"]),
        "step_t": int(packed["step_t"]),
        "token_index": int(packed["token_index"]),
        "action_results": action_results,
        "rtg_results": rtg_results,
        "processed_rtg_veh_ids": _as_int_list(packed["processed_rtg_veh_ids"]),
        "dead_ids": _as_int_list(packed["dead_ids"]),
    }
