from __future__ import annotations

from typing import Any, Dict, Optional, cast

import numpy as np

from .arrays import as_int32_array, as_int_list, pack_result_map, unpack_result_map
from .schema import MODEL_OUTPUTS_IPC_FORMAT, ModelOutputsPayload
from .validate import require_keys, require_valid_status, validate_model_outputs_payload


def pack_model_outputs(model_outputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if model_outputs is None:
        return None

    validate_model_outputs_payload(model_outputs)
    model_outputs_typed = cast(ModelOutputsPayload, model_outputs)
    status = require_valid_status(model_outputs_typed["status"], "model_outputs")

    action_veh_ids, action_values = pack_result_map(
        model_outputs_typed["action_results"],
        value_width=2,
    )
    rtg_veh_ids, rtg_values = pack_result_map(
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
        "processed_rtg_veh_ids": as_int32_array(model_outputs_typed["processed_rtg_veh_ids"]),
        "dead_ids": as_int32_array(model_outputs_typed["dead_ids"]),
        "next_worker_rng_state": np.asarray(
            model_outputs_typed["next_worker_rng_state"],
            dtype=np.uint8,
        ),
    }


def unpack_model_outputs(packed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
            "next_worker_rng_state",
        ),
        "packed model_outputs payload",
    )
    status = require_valid_status(packed["status"], "packed model_outputs payload")

    action_results = unpack_result_map(
        packed["action_veh_ids"],
        packed["action_values"],
        value_width=2,
        field_name="action",
    )
    rtg_results = unpack_result_map(
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
        "processed_rtg_veh_ids": as_int_list(packed["processed_rtg_veh_ids"]),
        "dead_ids": as_int_list(packed["dead_ids"]),
        "next_worker_rng_state": np.asarray(
            packed["next_worker_rng_state"],
            dtype=np.uint8,
        ),
    }

