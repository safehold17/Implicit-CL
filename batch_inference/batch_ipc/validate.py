"""
负责校验 batch inference IPC 负载的字段完整性、状态值与基础结构约束。
该模块面向新的 flat prepared/model_outputs 结构，在协议边界尽早暴露格式错误。
Validates required fields, status values, and structural constraints of batch inference IPC payloads.
It targets the new flat prepared/model_outputs structures and surfaces malformed payloads early.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .schema import MOTION_FIELD_NAMES, VALID_STATUS_VALUES


def require_keys(payload: Dict[str, Any], required_keys: Sequence[str], payload_name: str) -> None:
    """检查字典负载是否包含给定字段集合。

    Check whether the dictionary payload contains the required keys.
    """
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def require_valid_status(status: Any, payload_name: str) -> str:
    """校验协议状态值是否合法，并返回标准化字符串。

    Validate the protocol status value and return the normalized string.
    """
    status_str = str(status)
    if status_str not in VALID_STATUS_VALUES:
        raise ValueError(
            f"{payload_name} has invalid status='{status_str}', expected one of {sorted(VALID_STATUS_VALUES)}"
        )
    return status_str


def _validate_motion_batches(motion_data_np: Dict[str, Any], focal_count: int) -> None:
    """校验 flat prepared 中 batched motion 张量的字段齐全性与 batch 维度。

    Validate the field completeness and batch dimension of batched motion tensors in a flat prepared payload.
    """
    if not isinstance(motion_data_np, dict):
        raise ValueError("prepared['motion_data_np'] must be a dict.")
    require_keys(motion_data_np, MOTION_FIELD_NAMES, "prepared['motion_data_np']")
    for field_name in MOTION_FIELD_NAMES:
        field_value = np.asarray(motion_data_np[field_name])
        if field_value.ndim == 0:
            raise ValueError(f"prepared['motion_data_np']['{field_name}'] must be at least 1D.")
        if int(field_value.shape[0]) != focal_count:
            raise ValueError(
                f"prepared['motion_data_np']['{field_name}'] batch dimension mismatch."
            )


def validate_prepared_payload(prepared: Dict[str, Any]) -> None:
    """校验 flat prepared 负载的结构完整性。

    Validate the structural integrity of the flat prepared payload.
    """
    require_keys(
        prepared,
        (
            "status",
            "step_t",
            "token_index",
            "dead_ids",
            "target_rtg",
            "target_rtg_valid",
            "query_gap",
        ),
        "prepared",
    )
    target_rtg = np.asarray(prepared["target_rtg"])
    if target_rtg.shape != (3,):
        raise ValueError("prepared['target_rtg'] must have shape [3].")
    if target_rtg.dtype != np.float32:
        raise ValueError("prepared['target_rtg'] must have dtype float32.")
    target_rtg_valid = prepared["target_rtg_valid"]
    if not isinstance(target_rtg_valid, (bool, np.bool_)):
        raise ValueError("prepared['target_rtg_valid'] must be a bool.")
    query_gap = prepared["query_gap"]
    if isinstance(query_gap, (bool, np.bool_)) or not isinstance(
        query_gap,
        (int, np.integer),
    ):
        raise ValueError("prepared['query_gap'] must be an integer.")
    if int(query_gap) < 0:
        raise ValueError("prepared['query_gap'] must be non-negative.")
    if not bool(target_rtg_valid):
        if np.any(target_rtg != 0):
            raise ValueError("prepared invalid target_rtg metadata must use zeros.")
        if int(query_gap) != 0:
            raise ValueError("prepared invalid target_rtg metadata must use query_gap=0.")
    status = require_valid_status(prepared["status"], "prepared")
    if status == "skip":
        return

    require_keys(
        prepared,
        (
            "sampling",
            "default_tilt",
            "tilt_by_veh_id",
            "shared_timesteps",
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
            "motion_data_np",
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

    focal_ids = np.asarray(prepared["focal_ids"])
    predict_rtgs = np.asarray(prepared["predict_rtgs"])
    if focal_ids.ndim != 1:
        raise ValueError("prepared['focal_ids'] must be a 1D array.")
    if predict_rtgs.ndim != 1:
        raise ValueError("prepared['predict_rtgs'] must be a 1D array.")
    focal_count = int(focal_ids.shape[0])
    if int(predict_rtgs.shape[0]) != focal_count:
        raise ValueError("prepared focal arrays length mismatch.")

    for flat_key, offsets_key in (
        ("data_veh_ids_flat", "data_veh_ids_offsets"),
        ("veh_ids_in_context_flat", "veh_ids_in_context_offsets"),
        ("data_veh_model_indices_flat", "data_veh_model_indices_offsets"),
        ("context_veh_model_indices_flat", "context_veh_model_indices_offsets"),
    ):
        flat_values = np.asarray(prepared[flat_key])
        offsets = np.asarray(prepared[offsets_key])
        if flat_values.ndim != 1:
            raise ValueError(f"prepared['{flat_key}'] must be a 1D array.")
        if offsets.ndim != 1:
            raise ValueError(f"prepared['{offsets_key}'] must be a 1D array.")
        if int(offsets.shape[0]) != focal_count + 1:
            raise ValueError(f"prepared['{offsets_key}'] row count mismatch.")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"prepared['{offsets_key}'] must be non-decreasing.")
        if int(offsets[-1]) != int(flat_values.shape[0]):
            raise ValueError(f"prepared['{flat_key}'] length mismatch with offsets.")

    _validate_motion_batches(prepared["motion_data_np"], focal_count)


def validate_model_outputs_payload(model_outputs: Dict[str, Any]) -> None:
    """校验 flat model_outputs 负载的结构完整性。

    Validate the structural integrity of the flat model_outputs payload.
    """
    require_keys(
        model_outputs,
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
        "model_outputs",
    )
    require_valid_status(model_outputs["status"], "model_outputs")
    action_veh_ids = np.asarray(model_outputs["action_veh_ids"])
    action_values = np.asarray(model_outputs["action_values"])
    rtg_veh_ids = np.asarray(model_outputs["rtg_veh_ids"])
    rtg_values = np.asarray(model_outputs["rtg_values"])
    processed_rtg_veh_ids = np.asarray(model_outputs["processed_rtg_veh_ids"])
    dead_ids = np.asarray(model_outputs["dead_ids"])
    if action_veh_ids.ndim != 1:
        raise ValueError("model_outputs['action_veh_ids'] must be a 1D array.")
    if action_values.shape != (int(action_veh_ids.shape[0]), 2):
        raise ValueError("model_outputs['action_values'] must have shape [N, 2].")
    if rtg_veh_ids.ndim != 1:
        raise ValueError("model_outputs['rtg_veh_ids'] must be a 1D array.")
    if rtg_values.shape != (int(rtg_veh_ids.shape[0]), 3):
        raise ValueError("model_outputs['rtg_values'] must have shape [M, 3].")
    if processed_rtg_veh_ids.ndim != 1:
        raise ValueError("model_outputs['processed_rtg_veh_ids'] must be a 1D array.")
    if dead_ids.ndim != 1:
        raise ValueError("model_outputs['dead_ids'] must be a 1D array.")
