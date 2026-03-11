"""
负责 batch inference 协议中常见数组结构的类型转换、压平与重建。
该模块服务于结果映射、ragged 列表和 motion 字段打包，是 prepared/model_outputs 的基础工具层。
Provides type conversion, flattening, and reconstruction helpers for common array structures in the IPC protocol.
Underpins prepared/model_outputs packing for result maps, ragged lists, and motion-data fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def as_int32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def as_float32_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def as_int_list(values: Any) -> List[int]:
    return [int(v) for v in np.asarray(values, dtype=np.int32).tolist()]


def pack_result_map(
    result_map: Dict[int, Tuple[float, ...]],
    value_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    veh_ids = as_int32_array(list(result_map.keys()))
    if veh_ids.size == 0:
        return veh_ids, np.zeros((0, value_width), dtype=np.float32)
    values = as_float32_array([result_map[int(k)] for k in veh_ids.tolist()]).reshape((-1, value_width))
    return veh_ids, values


def unpack_result_map(
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


def pack_dict_int32(mapping: Dict[Any, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if not mapping:
        return as_int32_array([]), as_int32_array([])
    items = list(mapping.items())
    keys = as_int32_array([int(k) for k, _ in items])
    vals = as_int32_array([int(v) for _, v in items])
    return keys, vals


def unpack_dict_int32(keys: np.ndarray, vals: np.ndarray) -> Dict[int, int]:
    if keys.size == 0:
        return {}
    return {int(k): int(v) for k, v in zip(keys.tolist(), vals.tolist())}


def pack_ragged_int_lists(lists: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(lists) + 1,), dtype=np.int32)
    flat_values: List[int] = []
    cursor = 0
    for i, row in enumerate(lists):
        row_vals = [int(v) for v in row]
        flat_values.extend(row_vals)
        cursor += len(row_vals)
        offsets[i + 1] = cursor
    return as_int32_array(flat_values), offsets


def unpack_ragged_int_lists(flat: np.ndarray, offsets: np.ndarray) -> List[List[int]]:
    out: List[List[int]] = []
    for i in range(int(offsets.shape[0]) - 1):
        left = int(offsets[i])
        right = int(offsets[i + 1])
        out.append([int(v) for v in flat[left:right].tolist()])
    return out


def pack_motion_array(name: str, array: Any) -> np.ndarray:
    np_arr = np.asarray(array)
    if name in ("actions", "rtgs", "timesteps"):
        return np_arr.astype(np.int32, copy=False)
    if name == "moving_agent_mask":
        return np_arr.astype(np.bool_, copy=False)
    if np.issubdtype(np_arr.dtype, np.floating):
        return np_arr.astype(np.float32, copy=False)
    if np.issubdtype(np_arr.dtype, np.integer):
        return np_arr.astype(np.int32, copy=False)
    return np_arr.astype(np.float32, copy=False)


def packed_motion_dtype_for_array(name: str, np_arr: np.ndarray) -> np.dtype:
    if name in ("actions", "rtgs", "timesteps"):
        return np.dtype(np.int32)
    if name == "moving_agent_mask":
        return np.dtype(np.bool_)
    if np.issubdtype(np_arr.dtype, np.integer):
        return np.dtype(np.int32)
    return np.dtype(np.float32)


def packed_motion_nbytes(name: str, array: Any) -> int:
    np_arr = np.asarray(array)
    return int(np_arr.size) * packed_motion_dtype_for_array(name, np_arr).itemsize


def unpack_motion_array(name: str, array: np.ndarray) -> np.ndarray:
    if name in ("actions", "rtgs", "timesteps"):
        return np.asarray(array, dtype=np.int64)
    return array
