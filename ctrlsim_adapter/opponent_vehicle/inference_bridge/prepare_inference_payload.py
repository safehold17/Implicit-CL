"""
负责在当前仿真步收集控制车辆、构造 focal batches，并打包 prepared payload。
该模块还处理稀疏推理动作缓存与下一步 RTG 字段，是 adapter 到 worker 的输入边界。
Collects controlled vehicles, builds focal batches, and packs the prepared payload for the current step.
Also manages sparse-action caches and next-step RTG fields as the adapter-to-worker input boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from batch_inference.batch_protocol import pack_prepared

from .sampling_rng import resolve_sampling_rng_state

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
    worker_rng_state: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if adapter._policy is None or len(vehicles) == 0:
        clear_pending_sparse_actions(adapter)
        return None

    adapter._last_vehicles = vehicles
    adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}

    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(
        t,
        vehicles,
        adapter._vehicle_data_dict,
    )
    adapter.update_policy_state(t)

    if t < adapter.history_steps - 1:
        warmup_actions = build_warmup_gt_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=warmup_actions)

    is_sparse_step = adapter.sparse_inference.is_sparse_step(
        t=t,
        history_steps=adapter.history_steps,
    )
    if is_sparse_step and adapter.sparse_inference_action_repeat:
        actions = build_sparse_repeat_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=actions)
        return None

    clear_pending_sparse_actions(adapter)
    from .focal_input import build_focal_batches

    focal_batches, dead_ids = build_focal_batches(adapter, t)
    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1
    sampling_rng_state = resolve_sampling_rng_state(adapter, worker_rng_state)
    if not focal_batches and not dead_ids:
        return pack_prepared(
            {
                "status": "skip",
                "step_t": t,
                "token_index": token_index,
                "dead_ids": [],
                "worker_rng_state": sampling_rng_state,
            }
        )

    tilt_by_veh_id: Dict[int, tuple[int, int, int]] = (
        dict(adapter.per_vehicle_tilting) if adapter.per_vehicle_tilting else {}
    )
    prepared_dict = {
        "status": "ok",
        "step_t": t,
        "token_index": token_index,
        "dead_ids": dead_ids,
        "worker_rng_state": sampling_rng_state,
        "sampling": {
            "action_temperature": adapter.action_temperature,
            "nucleus_sampling": adapter.nucleus_sampling,
            "nucleus_threshold": adapter.nucleus_threshold,
        },
        "default_tilt": (
            adapter.current_tilt.goal_tilt,
            adapter.current_tilt.veh_veh_tilt,
            adapter.current_tilt.veh_edge_tilt,
        ),
        "tilt_by_veh_id": tilt_by_veh_id,
        "veh_id_to_idx": dict(adapter._policy.veh_id_to_idx),
        "focal_batches": focal_batches,
    }
    return pack_prepared(prepared_dict)
