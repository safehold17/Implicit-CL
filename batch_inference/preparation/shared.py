from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


def set_predict_rtgs_override(adapter: Any, step_t: int, predict_rtgs: bool) -> None:
    policy = adapter._policy
    if policy is None:
        return
    adapter._predict_rtgs_override_prev = bool(policy.predict_rtgs)
    adapter._predict_rtgs_override_step_t = int(step_t)
    policy.predict_rtgs = bool(predict_rtgs)


def clear_predict_rtgs_override(adapter: Any, step_t: Optional[int] = None) -> None:
    override_step = getattr(adapter, "_predict_rtgs_override_step_t", None)
    if override_step is None:
        return
    if step_t is not None and int(step_t) != int(override_step):
        return

    policy = adapter._policy
    prev_predict_rtgs = getattr(adapter, "_predict_rtgs_override_prev", None)
    if policy is not None and prev_predict_rtgs is not None:
        policy.predict_rtgs = bool(prev_predict_rtgs)
    adapter._predict_rtgs_override_step_t = None
    adapter._predict_rtgs_override_prev = None

