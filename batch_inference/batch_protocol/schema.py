"""
负责定义 batch inference IPC 协议的常量、状态值与 TypedDict 结构。
该模块是 prepared/model_outputs 两类负载的共享契约，供打包、校验和解包模块共同依赖。
Defines constants, status values, and TypedDict schemas for the batch inference IPC protocol.
Acts as the shared contract for prepared/model_outputs payloads across packing, validation, and unpacking code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np

PREPARED_IPC_FORMAT = "prepared_v3"
MODEL_OUTPUTS_IPC_FORMAT = "model_outputs_v1"
VALID_STATUS_VALUES = {"ok", "skip"}
INLINE_MOTION_STORAGE = "inline"
SHM_MOTION_STORAGE = "shm"
MOTION_FIELD_NAMES = (
    "agent_states",
    "agent_types",
    "goals",
    "actions",
    "rtgs",
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
    sampling_seed: int
    sampling: SamplingPayload
    default_tilt: Tuple[int, int, int]
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]]
    veh_id_to_idx: Dict[int, int]
    shared_timesteps: Any
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
