"""
负责校验 batch inference IPC 负载的字段完整性、状态值与基础结构约束。
该模块在 prepared/model_outputs 等协议边界上尽早暴露格式错误，避免下游解码阶段出现隐式失败。
Validates required fields, status values, and structural constraints of batch inference IPC payloads.
Surfaces malformed prepared/model_outputs payloads at the protocol boundary before downstream decoding.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .schema import MOTION_FIELD_NAMES, VALID_STATUS_VALUES


def require_keys(payload: Dict[str, Any], required_keys: Sequence[str], payload_name: str) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def require_valid_status(status: Any, payload_name: str) -> str:
    status_str = str(status)
    if status_str not in VALID_STATUS_VALUES:
        raise ValueError(
            f"{payload_name} has invalid status='{status_str}', expected one of {sorted(VALID_STATUS_VALUES)}"
        )
    return status_str


def validate_focal_batch_payload(focal_batch: Dict[str, Any], index: int) -> None:
    require_keys(
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
    require_keys(
        motion_data_np,
        MOTION_FIELD_NAMES,
        f"prepared.focal_batches[{index}].motion_data_np",
    )


def validate_prepared_payload(prepared: Dict[str, Any]) -> None:
    require_keys(prepared, ("status", "step_t", "token_index", "dead_ids"), "prepared")
    status = require_valid_status(prepared["status"], "prepared")
    if status == "skip":
        return

    require_keys(
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
    require_keys(
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
        validate_focal_batch_payload(focal_batch, i)


def validate_model_outputs_payload(model_outputs: Dict[str, Any]) -> None:
    require_keys(
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
    require_valid_status(model_outputs["status"], "model_outputs")
