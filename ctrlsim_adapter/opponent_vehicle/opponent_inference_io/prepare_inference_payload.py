"""
负责在当前仿真步收集控制车辆、构造 focal batches，并打包 prepared payload。
该模块还处理稀疏推理动作缓存与下一步 RTG 字段，是 adapter 到 worker 的输入边界。
Collects controlled vehicles, builds focal batches, and packs the prepared payload for the current step.
Also manages sparse-action caches and next-step RTG fields as the adapter-to-worker input boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from batch_inference.batch_ipc import pack_prepared
from ctrlsim_adapter.regret_enhancement_helper import (
    should_collect_ego_ctrlsim_rtg_step,
)
from ctrlsim_adapter.policy_reweighting_helpers import (
    should_trigger_policy_reweighting_step,
)

from .sampling_rng import resolve_sampling_seed

NEXT_RTG_KEYS = ("next_rtg_goal", "next_rtg_veh", "next_rtg_road")


def get_or_create_prepare_buffer(
    adapter: Any,
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    cache = getattr(adapter, "_batch_prepare_cache", None)
    if cache is None:
        cache = {}
        adapter._batch_prepare_cache = cache

    arr = cache.get(name)
    if arr is None or arr.shape != shape or arr.dtype != dtype:
        arr = np.empty(shape, dtype=dtype)
        cache[name] = arr
    return arr


def get_control_vehicle_queue(adapter: Any) -> List[int]:
    return list(adapter._vehicles_to_control_sorted or adapter._vehicles_to_control)


def require_vehicle_data(
    vehicle_data_dict: Dict[int, Dict[str, Any]],
    veh_id: int,
    source_name: str,
    step_t: int,
) -> Dict[str, Any]:
    veh_data = vehicle_data_dict.get(veh_id)
    if veh_data is None:
        raise ValueError(f"Unknown veh_id={veh_id} in {source_name} at step_t={step_t}")
    return veh_data


def get_step_controlled_ids(adapter: Any) -> List[int]:
    step_ids = list(getattr(adapter, "_controlled_vehicle_ids_step", []))
    if step_ids:
        return step_ids
    return list(adapter._vehicles_to_control)


def build_sparse_repeat_actions(
    adapter: Any,
    step_t: int,
) -> Dict[int, Tuple[float, float]]:
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in get_step_controlled_ids(adapter):
        veh_data = require_vehicle_data(
            adapter._vehicle_data_dict,
            veh_id,
            "sparse_repeat",
            step_t,
        )
        if not veh_data["existence"][-1]:
            action = (0.0, 0.0)
        else:
            accel_hist = veh_data["acceleration"]
            steer_hist = veh_data["steering"]
            if accel_hist and steer_hist:
                action = (float(accel_hist[-1]), float(steer_hist[-1]))
            else:
                action = (0.0, 0.0)
        veh_data["next_acceleration"] = action[0]
        veh_data["next_steering"] = action[1]
        actions[veh_id] = action
    return actions


def build_warmup_gt_actions(
    adapter: Any,
    step_t: int,
) -> Dict[int, Tuple[float, float]]:
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in get_step_controlled_ids(adapter):
        veh_data = require_vehicle_data(
            adapter._vehicle_data_dict,
            veh_id,
            "warmup_gt",
            step_t,
        )
        veh = adapter._last_vehicle_by_id.get(veh_id)
        action = adapter._get_gt_action(veh_id, step_t, veh)
        if action is None:
            action = (0.0, 0.0)
        accel = float(action[0])
        steer = float(action[1])
        veh_data["next_acceleration"] = accel
        veh_data["next_steering"] = steer
        actions[veh_id] = (accel, steer)
    return actions


def clear_pending_sparse_actions(adapter: Any) -> None:
    adapter._pending_sparse_actions_step_t = None
    adapter._pending_sparse_actions = {}


def set_pending_sparse_actions(
    adapter: Any,
    step_t: int,
    actions: Dict[int, Tuple[float, float]],
) -> None:
    adapter._pending_sparse_actions_step_t = int(step_t)
    adapter._pending_sparse_actions = dict(actions)


def consume_pending_sparse_actions(
    adapter: Any,
    step_t: Optional[int] = None,
) -> Optional[Dict[int, Tuple[float, float]]]:
    pending_step = getattr(adapter, "_pending_sparse_actions_step_t", None)
    pending_actions = dict(getattr(adapter, "_pending_sparse_actions", {}))
    if pending_step is None:
        return None
    if step_t is not None and int(pending_step) != int(step_t):
        return None
    clear_pending_sparse_actions(adapter)
    return pending_actions


def prepare_step(
    adapter: Any,
    t: int,
    vehicles: List[Any],
) -> Optional[Dict[str, Any]]:
    prepared_pack = prepare_step_pack(adapter, t, vehicles, ego_id=None)
    return prepared_pack["opponent_prepared"]


def _prepare_step_state(
    adapter: Any,
    t: int,
    vehicles: List[Any],
) -> bool:
    """同步当前步的 adapter 状态。 / Refresh adapter state for the current step."""
    if adapter._policy is None or len(vehicles) == 0:
        clear_pending_sparse_actions(adapter)
        return False

    adapter._last_vehicles = vehicles
    adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}

    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(
        t,
        vehicles,
        adapter._vehicle_data_dict,
    )
    adapter.update_policy_state(t)
    return True


def _should_skip_opponent_inference(
    adapter: Any,
    t: int,
) -> bool:
    """判断当前步是否跳过 opponent 推理。 / Decide whether to skip opponent inference this step."""
    if t < adapter.history_steps - 1:
        warmup_actions = build_warmup_gt_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=warmup_actions)
        return True

    is_sparse_step = adapter.sparse_inference.is_sparse_step(
        t=t,
        history_steps=adapter.history_steps,
    )
    if is_sparse_step and adapter.sparse_inference_action_repeat:
        actions = build_sparse_repeat_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=actions)
        return True

    clear_pending_sparse_actions(adapter)
    return False


def should_collect_ego_ctrlsim_kl_step(
    t: int,
    history_steps: int,
    kl_loss_computation_frequency: int,
) -> bool:
    """判断当前步是否需要采样 ego KL teacher。 / Decide whether this step should collect ego KL teacher logits."""
    if kl_loss_computation_frequency < 1:
        raise ValueError(
            "kl_loss_computation_frequency must be >= 1, "
            f"got {kl_loss_computation_frequency}"
        )

    anchor_t = int(history_steps) - 1
    if t < anchor_t:
        return False
    phase = (int(t) - anchor_t) % int(kl_loss_computation_frequency)
    return phase == int(kl_loss_computation_frequency) - 1


def _build_prepared_payload(
    adapter: Any,
    t: int,
    focal_vehicle_ids: List[int],
    predict_rtgs: bool,
    ego_id: Optional[int],
    owner_focal_id: Optional[int],
    ego_reweight_tilt: Tuple[int, int, int],
    delayed_ego_action_scale: float,
    default_tilt: Tuple[int, int, int],
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]],
    shared_context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """为给定 focal 车辆列表打包 prepared payload。 / Pack a prepared payload for the given focal vehicles."""
    if not focal_vehicle_ids:
        return None

    from .focal_input import build_focal_batches_from_shared_context

    focal_layout, dead_ids, shared_timesteps = (
        build_focal_batches_from_shared_context(
            adapter,
            t,
            shared_context,
            focal_vehicle_ids=focal_vehicle_ids,
            predict_rtgs=predict_rtgs,
        )
    )
    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1
    sampling_seed = resolve_sampling_seed(adapter)
    if int(np.asarray(focal_layout["focal_ids"], dtype=np.int64).shape[0]) == 0 and not dead_ids:
        return pack_prepared(
            {
                "status": "skip",
                "step_t": t,
                "token_index": token_index,
                "dead_ids": [],
                "sampling_seed": sampling_seed,
            }
        )

    prepared_dict = {
        "status": "ok",
        "step_t": t,
        "token_index": token_index,
        "dead_ids": dead_ids,
        "sampling_seed": sampling_seed,
        "ego_id": None if ego_id is None else int(ego_id),
        "ego_context_owner_focal_id": (
            None if owner_focal_id is None else int(owner_focal_id)
        ),
        "ego_reweight_tilt": tuple(int(v) for v in ego_reweight_tilt),
        "delayed_ego_action_scale": float(delayed_ego_action_scale),
        "sampling": {
            "action_temperature": adapter.action_temperature,
            "nucleus_sampling": adapter.nucleus_sampling,
            "nucleus_threshold": adapter.nucleus_threshold,
        },
        "default_tilt": default_tilt,
        "tilt_by_veh_id": tilt_by_veh_id,
        "shared_timesteps": shared_timesteps,
        **focal_layout,
    }
    return pack_prepared(prepared_dict)


def prepare_step_pack(
    adapter: Any,
    t: int,
    vehicles: List[Any],
    ego_id: Optional[int],
    include_ego_ctrlsim_prepared: bool = True,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    为当前仿真步构建推理输入 pack，按需分别产出对手车辆和 ego 的 prepared payload。
    Build the inference-input pack for the current simulation step, optionally
    producing separate prepared payloads for opponent vehicles and the ego vehicle.

    The function first refreshes runtime state for the current step. If opponent
    inference should be skipped, such as during warmup or a sparse-repeat step,
    `opponent_prepared` is set to None. When a valid `ego_id` is provided, it
    also builds `ego_ctrlsim_prepared` for ego ctrl-sim inference.

    Returns:
        包含 `opponent_prepared` 和 `ego_ctrlsim_prepared` 的字典。
        A dictionary containing `opponent_prepared` and `ego_ctrlsim_prepared`.
    """
    if not _prepare_step_state(adapter, t, vehicles):
        return {
            "opponent_prepared": None,
            "ego_ctrlsim_prepared": None,
        }

    opponent_needs_prepare = not _should_skip_opponent_inference(adapter, t)
    ego_policy_ready = (
        include_ego_ctrlsim_prepared
        and ego_id is not None
        and ego_id in adapter._policy.veh_id_to_idx
    )
    ego_needs_kl_prepare = (
        ego_policy_ready
        and bool(getattr(adapter, "use_ego_ctrlsim_kl_loss", True))
        and should_collect_ego_ctrlsim_kl_step(
            t=t,
            history_steps=adapter.history_steps,
            kl_loss_computation_frequency=adapter.kl_loss_computation_frequency,
        )
    )
    ego_needs_rtg_prepare = (
        ego_policy_ready
        and bool(getattr(adapter, "enhanced_regret", False))
        and should_collect_ego_ctrlsim_rtg_step(
            t=t,
            history_steps=adapter.history_steps,
            sparse_inference_action_repeat=bool(
                getattr(adapter, "sparse_inference_action_repeat", False)
            ),
            action_repeat_frequency=int(
                getattr(adapter, "action_repeat_frequency", 1)
            ),
        )
    )
    ego_needs_prepare = ego_needs_kl_prepare or ego_needs_rtg_prepare

    shared_context = None
    if opponent_needs_prepare or ego_needs_prepare:
        from .focal_input import build_step_shared_context

        shared_context = build_step_shared_context(adapter, t)

    owner_focal_id = None
    if ego_id is not None and adapter._policy is not None:
        ego_id_int = int(ego_id)
        if ego_id_int in adapter._policy.veh_id_to_idx:
            owner_focal_id = ego_id_int

    delayed_ego_action_scale = 1.0
    if owner_focal_id is not None and bool(getattr(adapter, "use_policy_reweighting", False)):
        should_trigger = should_trigger_policy_reweighting_step(
            t=t,
            history_steps=int(adapter.history_steps),
            reweighting_frequency=int(adapter.reweighting_frequency),
        )
        if should_trigger:
            delayed_ego_action_scale = float(getattr(adapter, "_ego_action_scale", 1.0))
    ego_reweight_tilt = tuple(
        int(v) for v in getattr(adapter, "_ego_reweight_tilt", (0, 0, 0))
    )

    opponent_prepared = None
    if opponent_needs_prepare:
        opponent_prepared = _build_prepared_payload(
            adapter,
            t,
            focal_vehicle_ids=get_control_vehicle_queue(adapter),
            predict_rtgs=bool(adapter._policy.predict_rtgs),
            ego_id=ego_id,
            owner_focal_id=owner_focal_id,
            ego_reweight_tilt=ego_reweight_tilt,
            delayed_ego_action_scale=delayed_ego_action_scale,
            default_tilt=(
                adapter.current_tilt.goal_tilt,
                adapter.current_tilt.veh_veh_tilt,
                adapter.current_tilt.veh_edge_tilt,
            ),
            tilt_by_veh_id=(
                dict(adapter.per_vehicle_tilting)
                if adapter.per_vehicle_tilting
                else {}
            ),
            shared_context=shared_context,
        )

    ego_ctrlsim_prepared = None
    if ego_needs_prepare:
        ego_ctrlsim_prepared = _build_prepared_payload(
            adapter,
            t,
            focal_vehicle_ids=[int(ego_id)],
            predict_rtgs=True,
            ego_id=ego_id,
            owner_focal_id=owner_focal_id,
            ego_reweight_tilt=ego_reweight_tilt,
            delayed_ego_action_scale=delayed_ego_action_scale,
            default_tilt=(0, 0, 0),
            tilt_by_veh_id={},
            shared_context=shared_context,
        )

    return {
        "opponent_prepared": opponent_prepared,
        "ego_ctrlsim_prepared": ego_ctrlsim_prepared,
    }
