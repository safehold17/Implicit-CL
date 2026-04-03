"""
负责定义 batch inference IPC 协议的常量、状态值与 TypedDict 结构。
该模块是 prepared/model_outputs 两类负载的共享契约，供打包、校验和解包模块共同依赖。
Defines constants, status values, and TypedDict schemas for the batch inference IPC protocol.
Acts as the shared contract for prepared/model_outputs payloads across packing, validation, and unpacking code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypedDict

PREPARED_IPC_FORMAT = "prepared_v4"
MODEL_OUTPUTS_IPC_FORMAT = "model_outputs_v2"
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


class PreparedPayload(TypedDict):
    status: str
    step_t: int
    token_index: int
    dead_ids: List[int]
    sampling_seed: int
    ego_id: int | None
    ego_context_owner_focal_id: int | None
    ego_reweight_tilt: Tuple[int, int, int]
    delayed_ego_action_scale: float
    sampling: SamplingPayload
    default_tilt: Tuple[int, int, int]
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]]
    shared_timesteps: Any
    focal_ids: Any
    predict_rtgs: Any
    data_veh_ids_flat: Any
    data_veh_ids_offsets: Any
    veh_ids_in_context_flat: Any
    veh_ids_in_context_offsets: Any
    data_veh_model_indices_flat: Any
    data_veh_model_indices_offsets: Any
    context_veh_model_indices_flat: Any
    context_veh_model_indices_offsets: Any
    motion_data_np: Dict[str, Any]


class ModelOutputsPayload(TypedDict):
    status: str
    env_idx: int
    step_t: int
    token_index: int
    ego_action_scale: float
    action_veh_ids: Any
    action_values: Any
    rtg_veh_ids: Any
    rtg_values: Any
    processed_rtg_veh_ids: Any
    dead_ids: Any
