"""
负责模型输出负载在内存结构与 IPC 表示之间的转换。
该模块使用 flat 结果数组表达 action/RTG/dead_ids，避免在 worker 与 adapter 之间往返构造结果字典。
Converts model outputs between in-memory structures and their IPC representation.
It uses flat result arrays for action/RTG/dead-ids to avoid rebuilding result maps between worker and adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import numpy as np

from .arrays import as_float32_array, as_int32_array
from .schema import MODEL_OUTPUTS_IPC_FORMAT, ModelOutputsPayload
from .validate import require_keys, require_valid_status, validate_model_outputs_payload


def pack_model_outputs(model_outputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 flat model_outputs 结构打包为 IPC payload。

    Pack the flat model_outputs structure into its IPC payload representation.
    """
    if model_outputs is None:
        return None

    validate_model_outputs_payload(model_outputs)
    model_outputs_typed = cast(ModelOutputsPayload, model_outputs)
    status = require_valid_status(model_outputs_typed["status"], "model_outputs")
    return {
        "ipc_format": MODEL_OUTPUTS_IPC_FORMAT,
        "status": status,
        "env_idx": np.int32(int(model_outputs_typed["env_idx"])),
        "step_t": np.int32(int(model_outputs_typed["step_t"])),
        "token_index": np.int32(int(model_outputs_typed["token_index"])),
        "ego_action_scale": np.float32(
            float(model_outputs_typed.get("ego_action_scale", 1.0))
        ),
        "action_veh_ids": as_int32_array(model_outputs_typed["action_veh_ids"]),
        "action_values": as_float32_array(model_outputs_typed["action_values"]).reshape((-1, 2)),
        "rtg_veh_ids": as_int32_array(model_outputs_typed["rtg_veh_ids"]),
        "rtg_values": as_float32_array(model_outputs_typed["rtg_values"]).reshape((-1, 3)),
        "processed_rtg_veh_ids": as_int32_array(model_outputs_typed["processed_rtg_veh_ids"]),
        "dead_ids": as_int32_array(model_outputs_typed["dead_ids"]),
    }


def unpack_model_outputs(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """将 IPC payload 还原为 flat model_outputs 结构。

    Unpack the IPC payload back into the flat model_outputs structure.
    """
    if packed is None:
        return None
    if packed.get("ipc_format") != MODEL_OUTPUTS_IPC_FORMAT:
        raise ValueError("Unexpected model_outputs IPC payload format.")

    require_keys(
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
    status = require_valid_status(packed["status"], "packed model_outputs payload")
    return {
        "status": status,
        "env_idx": int(packed["env_idx"]),
        "step_t": int(packed["step_t"]),
        "token_index": int(packed["token_index"]),
        "ego_action_scale": float(np.float32(packed.get("ego_action_scale", 1.0))),
        "action_veh_ids": np.asarray(packed["action_veh_ids"], dtype=np.int64),
        "action_values": np.asarray(packed["action_values"], dtype=np.float32).reshape((-1, 2)),
        "rtg_veh_ids": np.asarray(packed["rtg_veh_ids"], dtype=np.int64),
        "rtg_values": np.asarray(packed["rtg_values"], dtype=np.float32).reshape((-1, 3)),
        "processed_rtg_veh_ids": np.asarray(
            packed["processed_rtg_veh_ids"],
            dtype=np.int64,
        ),
        "dead_ids": np.asarray(packed["dead_ids"], dtype=np.int64),
    }
