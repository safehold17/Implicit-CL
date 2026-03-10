from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..opponent_state_helpers import store_next_action


def build_warmup_gt_actions(service: Any, t: int) -> Dict[int, Tuple[float, float]]:
    adapter = service.adapter
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in adapter._controlled_vehicle_ids_step:
        veh_data = adapter._vehicle_data_dict.get(veh_id)
        veh = adapter._step_vehicle_by_id.get(veh_id)
        if veh_data is None or veh is None:
            continue
        action = service._get_gt_action(veh_id, t, veh)
        store_next_action(veh_data, veh_id, action, actions)
    return actions


def build_sparse_repeat_actions(service: Any) -> Dict[int, Tuple[float, float]]:
    adapter = service.adapter
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in adapter._controlled_vehicle_ids_step:
        veh_data = adapter._vehicle_data_dict.get(veh_id)
        if veh_data is None:
            continue
        if not veh_data["existence"][-1]:
            action = (0.0, 0.0)
        else:
            accel_hist = veh_data["acceleration"]
            steer_hist = veh_data["steering"]
            action = (
                (float(accel_hist[-1]), float(steer_hist[-1]))
                if accel_hist and steer_hist
                else (0.0, 0.0)
            )
        store_next_action(veh_data, veh_id, action, actions)
    return actions


def collect_predicted_actions(service: Any) -> Dict[int, Tuple[float, float]]:
    adapter = service.adapter
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in adapter._controlled_vehicle_ids_step:
        veh_data = adapter._vehicle_data_dict.get(veh_id)
        if veh_data is None:
            continue
        if not veh_data["existence"][-1]:
            action = (0.0, 0.0)
        else:
            action = (
                float(veh_data["next_acceleration"]),
                float(veh_data["next_steering"]),
            )
        store_next_action(veh_data, veh_id, action, actions)
    return actions


def predict_actions_with_policy(
    service: Any,
    t: int,
) -> Dict[int, Tuple[float, float]]:
    adapter = service.adapter
    policy = adapter._policy
    if policy is None:
        return {}

    for veh_id in adapter._controlled_vehicle_ids_present:
        policy.relevant_agent_idxs.setdefault(veh_id, [])
    adapter._vehicle_data_dict = policy.predict(
        adapter._vehicle_data_dict,
        adapter._gt_data_dict,
        adapter._preproc_data,
        adapter.dataset,
        adapter._vehicles_to_control,
        t,
    )
    return collect_predicted_actions(service)


def step(service: Any, t: int, vehicles: List[Any]) -> Dict[int, Tuple[float, float]]:
    adapter = service.adapter
    if len(vehicles) == 0 or adapter._policy is None:
        return {}

    adapter._vehicle_data_dict = service._update_vehicle_data_dict(
        t,
        vehicles,
        adapter._vehicle_data_dict,
    )
    service.update_policy_state(t)

    if t < adapter.history_steps - 1:
        predict_actions_with_policy(
            service,
            t=t,
        )
        return build_warmup_gt_actions(service, t)

    is_sparse_step = adapter.sparse_inference.is_sparse_step(
        t=t,
        history_steps=adapter.history_steps,
    )
    if not is_sparse_step:
        return predict_actions_with_policy(
            service,
            t=t,
        )
    if adapter.sparse_inference_action_repeat:
        return build_sparse_repeat_actions(service)
    return predict_actions_with_policy(service, t=t)


def apply_action(service: Any, veh: Any, action: Tuple[float, float]) -> None:
    acceleration, steering = action
    if acceleration > 0.0:
        veh.acceleration = acceleration
    else:
        veh.brake(np.abs(acceleration))
    veh.steering = steering


def record_action(service: Any, veh_id: int, action: Tuple[float, float]) -> None:
    adapter = service.adapter
    if veh_id in adapter._vehicle_data_dict:
        adapter._vehicle_data_dict[veh_id]["acceleration"].append(action[0])
        adapter._vehicle_data_dict[veh_id]["steering"].append(action[1])


def record_all_actions(
    service: Any,
    t: int,
    vehicles: List[Any],
    controlled_actions: Dict[int, Tuple[float, float]],
) -> None:
    for veh in vehicles:
        veh_id = veh.getID()
        if veh_id in controlled_actions:
            action = controlled_actions[veh_id]
        else:
            action = service._get_gt_action(veh_id, t, veh)
        record_action(service, veh_id, action)


def finalize(service: Any, vehicles: List[Any]) -> Dict:
    adapter = service.adapter
    for veh in vehicles:
        veh_id = veh.getID()
        if veh_id in adapter._vehicle_data_dict:
            adapter._vehicle_data_dict[veh_id]["acceleration"].append(0)
            adapter._vehicle_data_dict[veh_id]["steering"].append(0)
    return adapter._vehicle_data_dict


def is_initialized(service: Any) -> bool:
    return service.adapter._policy is not None


def get_vehicle_data(service: Any, veh_id: int) -> Optional[Dict]:
    return service.adapter._vehicle_data_dict.get(veh_id)
