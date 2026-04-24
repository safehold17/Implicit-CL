"""
负责消费 batch inference 返回的 model_outputs，并生成当前步的对手动作结果。
该模块同时处理 skip 状态、稀疏动作复用和 RNG 恢复，是 worker 输出回接 adapter 的出口。
Consumes `model_outputs` from batch inference and produces opponent actions for the current step.
Handles skip states, sparse action reuse, and RNG restoration as the adapter-side output bridge.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from batch_inference.batch_ipc import unpack_model_outputs, validate_model_outputs_payload

from .prepare_inference_payload import (
    NEXT_RTG_KEYS,
    consume_pending_sparse_actions,
    get_step_controlled_ids,
    require_vehicle_data,
)


def _iter_flat_result_rows(
    veh_ids: Any,
    values: Any,
    *,
    value_width: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """按行遍历 flat 结果数组，返回 `(veh_id, values_row)` 迭代器。

    Iterate over a flat result array row by row and yield `(veh_id, values_row)` pairs.
    """
    veh_ids_array = np.asarray(veh_ids, dtype=np.int64).reshape((-1,))
    values_array = np.asarray(values, dtype=np.float32).reshape((-1, value_width))
    if veh_ids_array.shape[0] != values_array.shape[0]:
        raise ValueError("Flat result arrays length mismatch.")
    for veh_id, value_row in zip(veh_ids_array.tolist(), values_array):
        yield int(veh_id), value_row


def reset_ego_action_scale(adapter: Any) -> None:
    """把 adapter 上缓存的 ego_action_scale 收口到 unity。 / Reset the adapter's cached ego_action_scale back to unity.

    这个 helper 统一承载所有“当前 step 不应保留上一步 scale”的语义，包括
    replay/disable/no-opponent 路径，以及 batch inference 返回 `None` 或 `skip`
    的情况。这样 runtime 和 apply bridge 都只表达“需要收口”，不再各自直接写字段。

    This helper owns the shared "collapse to unity" semantics for any path
    where the current step must not retain the previous scale, including
    replay/disable/no-opponent branches and `None`/`skip` outputs from batched
    inference. It lets both runtime and the apply bridge express intent
    without directly mutating the field in multiple places.
    """
    adapter._ego_action_scale = 1.0


def _append_rtg_history_for_tracked_vehicles(
    adapter: Any,
    *,
    processed_rtg_veh_ids: set[int],
    step_t: int,
) -> None:
    """Append one RTG row for every tracked vehicle at the current step."""
    zero_rtg = np.zeros(
        adapter._policy.cfg_model.num_reward_components,
        dtype=np.float32,
    )
    for veh_id, veh_data in adapter._vehicle_data_dict.items():
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
            continue
        veh_data["rtgs"].append(zero_rtg.copy())


def apply_predictions(
    adapter: Any,
    model_outputs: Optional[Dict[str, Any]],
) -> Dict[int, Tuple[float, float]]:
    if adapter._policy is None:
        return {}

    model_outputs = unpack_model_outputs(model_outputs)
    if model_outputs is None:
        reset_ego_action_scale(adapter)
        pending_actions = consume_pending_sparse_actions(adapter)
        if pending_actions is not None:
            return pending_actions
        return {}

    validate_model_outputs_payload(model_outputs)
    step_t = int(model_outputs["step_t"])
    status = str(model_outputs["status"])
    if status == "skip":
        reset_ego_action_scale(adapter)
        pending_actions = consume_pending_sparse_actions(adapter, step_t=step_t)
        if pending_actions is not None:
            return pending_actions
        return {}

    adapter._ego_action_scale = float(model_outputs.get("ego_action_scale", 1.0))

    action_veh_ids = np.asarray(model_outputs["action_veh_ids"], dtype=np.int64)
    action_values = np.asarray(model_outputs["action_values"], dtype=np.float32).reshape((-1, 2))
    rtg_veh_ids = np.asarray(model_outputs["rtg_veh_ids"], dtype=np.int64)
    rtg_values = np.asarray(model_outputs["rtg_values"], dtype=np.float32).reshape((-1, 3))
    processed_rtg_veh_ids = set(
        np.asarray(model_outputs["processed_rtg_veh_ids"], dtype=np.int64).tolist()
    )
    dead_ids = set(np.asarray(model_outputs["dead_ids"], dtype=np.int64).tolist())

    for veh_id, rtg_row in _iter_flat_result_rows(rtg_veh_ids, rtg_values, value_width=3):
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "rtg_results", step_t)
        veh_data["next_rtg_goal"] = float(rtg_row[0])
        veh_data["next_rtg_veh"] = float(rtg_row[1])
        veh_data["next_rtg_road"] = float(rtg_row[2])

    if adapter._policy.predict_rtgs:
        _append_rtg_history_for_tracked_vehicles(
            adapter,
            processed_rtg_veh_ids=processed_rtg_veh_ids,
            step_t=step_t,
        )

    action_by_vehicle: Dict[int, Tuple[float, float]] = {}
    for veh_id, action_row in _iter_flat_result_rows(action_veh_ids, action_values, value_width=2):
        accel = float(action_row[0])
        steer = float(action_row[1])
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "action_results", step_t)
        veh_data["next_acceleration"] = accel
        veh_data["next_steering"] = steer
        action_by_vehicle[veh_id] = (accel, steer)

    for veh_id in dead_ids:
        veh_data = require_vehicle_data(adapter._vehicle_data_dict, veh_id, "dead_ids", step_t)
        veh_data["next_acceleration"] = 0.0
        veh_data["next_steering"] = 0.0

    if step_t < adapter.history_steps - 1:
        pending_actions = consume_pending_sparse_actions(adapter, step_t=step_t)
        if pending_actions is not None:
            return pending_actions

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

        if veh_id in action_by_vehicle:
            actions[veh_id] = action_by_vehicle[veh_id]
            continue
        if veh_id in dead_ids:
            actions[veh_id] = (0.0, 0.0)
            continue
        raise ValueError(
            f"Missing action for non-dead controlled vehicle veh_id={veh_id} at step_t={step_t}."
        )

    return actions
