"""
负责消费 batch inference 返回的 model_outputs，并生成当前步的对手动作结果。
该模块同时处理 skip 状态、稀疏动作复用和 RNG 恢复，是 worker 输出回接 adapter 的出口。
Consumes `model_outputs` from batch inference and produces opponent actions for the current step.
Handles skip states, sparse action reuse, and RNG restoration as the adapter-side output bridge.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from batch_inference.batch_ipc import unpack_model_outputs, validate_model_outputs_payload

from .prepare_inference_payload import (
    NEXT_RTG_KEYS,
    consume_pending_sparse_actions,
    get_step_controlled_ids,
    require_vehicle_data,
)


def _get_rtg_history_vehicle_ids(
    adapter: Any,
    processed_rtg_veh_ids: set[int],
    dead_ids: set[int],
) -> list[int]:
    """返回当前步需要写回 RTG 历史的车辆列表。

    该列表只覆盖当前步真正相关的车辆：受控车辆优先，其次补上本步额外返回 RTG 的车辆和死亡车辆，
    以避免在 `predict_rtgs=True` 时扫描整个 `vehicle_data_dict`。

    Return the vehicle ids whose RTG history should be written for the current step.
    The list is scoped to vehicles that are actually relevant to this step: controlled vehicles first,
    followed by any additional vehicles appearing in RTG outputs or dead-id results, avoiding a full
    scan over `vehicle_data_dict` when `predict_rtgs=True`.
    """
    ordered_ids = list(get_step_controlled_ids(adapter))
    seen_ids = set(ordered_ids)
    for veh_id in sorted(processed_rtg_veh_ids | dead_ids):
        veh_id_int = int(veh_id)
        if veh_id_int in seen_ids:
            continue
        if veh_id_int not in adapter._vehicle_data_dict:
            continue
        ordered_ids.append(veh_id_int)
        seen_ids.add(veh_id_int)
    return ordered_ids


def apply_predictions(
    adapter: Any,
    model_outputs: Optional[Dict[str, Any]],
) -> Dict[int, Tuple[float, float]]:
    if adapter._policy is None:
        return {}

    model_outputs = unpack_model_outputs(model_outputs)
    if model_outputs is None:
        adapter._ego_action_scale = 1.0
        pending_actions = consume_pending_sparse_actions(adapter)
        if pending_actions is not None:
            return pending_actions
        return {}

    validate_model_outputs_payload(model_outputs)

    step_t = int(model_outputs["step_t"])
    status = str(model_outputs["status"])
    if status == "skip":
        adapter._ego_action_scale = 1.0
        pending_actions = consume_pending_sparse_actions(adapter, step_t=step_t)
        if pending_actions is not None:
            return pending_actions
        return {}

    adapter._ego_action_scale = float(model_outputs.get("ego_action_scale", 1.0))

    action_results = model_outputs["action_results"]
    rtg_results = model_outputs["rtg_results"]
    processed_rtg_veh_ids = set(model_outputs["processed_rtg_veh_ids"])
    dead_ids = set(model_outputs["dead_ids"])

    for veh_id, (goal_val, veh_val, road_val) in rtg_results.items():
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "rtg_results", step_t)
        veh_data["next_rtg_goal"] = goal_val
        veh_data["next_rtg_veh"] = veh_val
        veh_data["next_rtg_road"] = road_val

    if adapter._policy.predict_rtgs:
        zero_rtg = np.zeros(
            adapter._policy.cfg_model.num_reward_components,
            dtype=np.float32,
        )
        for veh_id in _get_rtg_history_vehicle_ids(
            adapter=adapter,
            processed_rtg_veh_ids=processed_rtg_veh_ids,
            dead_ids=dead_ids,
        ):
            veh_data = require_vehicle_data(
                adapter._vehicle_data_dict,
                veh_id,
                "predict_rtgs_history",
                step_t,
            )
            if veh_id in processed_rtg_veh_ids:
                missing = [key for key in NEXT_RTG_KEYS if key not in veh_data]
                if missing:
                    raise ValueError(
                        f"Missing RTG fields for veh_id={veh_id} at step_t={step_t}: {missing}"
                    )
                veh_data["rtgs"].append(
                    np.array(
                        [
                            veh_data["next_rtg_goal"],
                            veh_data["next_rtg_veh"],
                            veh_data["next_rtg_road"],
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                veh_data["rtgs"].append(zero_rtg.copy())

    for veh_id, (accel, steer) in action_results.items():
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "action_results", step_t)
        veh_data["next_acceleration"] = accel
        veh_data["next_steering"] = steer

    for veh_id in dead_ids:
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "dead_ids", step_t)
        veh_data["next_acceleration"] = 0.0
        veh_data["next_steering"] = 0.0

    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in get_step_controlled_ids(adapter):
        veh_data = require_vehicle_data(
            adapter._vehicle_data_dict,
            veh_id,
            "controlled_vehicle_ids_step",
            step_t,
        )
        if not veh_data["existence"][-1]:
            veh = adapter._last_vehicle_by_id.get(veh_id)
            if veh is not None:
                veh.setPosition(-1000000, -1000000)
            actions[veh_id] = (0.0, 0.0)
            continue

        if veh_id in action_results:
            actions[veh_id] = action_results[veh_id]
            continue
        if veh_id in dead_ids:
            actions[veh_id] = (0.0, 0.0)
            continue
        raise ValueError(
            f"Missing action for non-dead controlled vehicle veh_id={veh_id} at step_t={step_t}."
        )

    return actions
