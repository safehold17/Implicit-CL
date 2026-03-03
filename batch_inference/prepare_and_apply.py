"""Batch inference 模式下 adapter 侧的数据准备与结果应用。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ipc_codec import pack_prepared, unpack_model_outputs

_REQUIRED_MODEL_OUTPUT_KEYS = (
    "status",
    "env_idx",
    "step_t",
    "token_index",
    "action_results",
    "rtg_results",
    "processed_rtg_veh_ids",
    "dead_ids",
)


def _slice_policy_window(policy: Any, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tcl = policy.cfg_rl_waymo.train_context_length
    if t < tcl:
        start = 0
        end = tcl
    else:
        start = t - (tcl - 1)
        end = t + 1

    return (
        policy.states[:, start:end].copy(),
        policy.types.copy(),
        policy.actions[:, start:end].copy(),
        policy.rtgs[:, start:end].copy(),
        policy.goals[:, start:end].copy(),
        policy.timesteps[0, start:end].astype(int).copy(),
    )


def _normalize_rtgs_inplace(rtgs: np.ndarray, rl: Any) -> None:
    rtgs[:, :, 0] = (np.clip(rtgs[:, :, 0], rl.min_rtg_pos, rl.max_rtg_pos) - rl.min_rtg_pos) / (
        rl.max_rtg_pos - rl.min_rtg_pos
    )
    rtgs[:, :, 1] = (np.clip(rtgs[:, :, 1], rl.min_rtg_veh, rl.max_rtg_veh) - rl.min_rtg_veh) / (
        rl.max_rtg_veh - rl.min_rtg_veh
    )
    rtgs[:, :, 2] = (np.clip(rtgs[:, :, 2], rl.min_rtg_road, rl.max_rtg_road) - rl.min_rtg_road) / (
        rl.max_rtg_road - rl.min_rtg_road
    )


def _get_control_vehicle_queue(adapter: Any) -> List[int]:
    if adapter._vehicles_to_control_sorted:
        return list(adapter._vehicles_to_control_sorted)
    return list(adapter._vehicles_to_control)


def _build_motion_data_np(
    rel_ag_states: np.ndarray,
    rel_ag_types: np.ndarray,
    rel_goals: np.ndarray,
    rel_actions: np.ndarray,
    rel_rtgs: np.ndarray,
    rel_timesteps: np.ndarray,
    rel_moving_agent_mask: np.ndarray,
    rel_road_points: np.ndarray,
    rel_road_types: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "agent_states": rel_ag_states,
        "agent_types": rel_ag_types,
        "goals": rel_goals,
        "actions": rel_actions,
        "rtgs": rel_rtgs,
        "timesteps": rel_timesteps,
        "moving_agent_mask": rel_moving_agent_mask,
        "road_points": rel_road_points,
        "road_types": rel_road_types,
    }


def _build_focal_batch(
    adapter: Any,
    t: int,
    focal_id: int,
    remaining_veh_ids: List[int],
    ag_states: np.ndarray,
    ag_types: np.ndarray,
    actions: np.ndarray,
    rtgs: np.ndarray,
    goals: np.ndarray,
    timesteps: np.ndarray,
    moving_agent_mask: np.ndarray,
) -> Tuple[Optional[Dict[str, Any]], List[int], bool]:
    policy = adapter._policy
    dset = adapter.dataset
    rl = policy.cfg_rl_waymo

    origin_agent_idx = policy.veh_id_to_idx[focal_id]
    if not policy.states[origin_agent_idx, t, -1]:
        return None, [], True

    road_points = adapter._preproc_data["road_points"].copy()
    road_types = adapter._preproc_data["road_types"].copy()
    if len(road_points) == 0:
        return None, [], True

    if t == 0:
        policy.relevant_agent_idxs[focal_id] = []

    normalize_timestep = 0
    rel_timesteps = np.repeat(np.expand_dims(timesteps, 0), rl.max_num_agents, axis=0)
    (
        rel_ag_states,
        rel_ag_types,
        rel_actions,
        rel_rtgs,
        rel_goals,
        rel_moving_agent_mask,
        new_agent_idx_dict,
        relevant_agent_idxs,
    ) = dset.select_relevant_agents(
        ag_states,
        ag_types,
        actions,
        rtgs,
        goals[:, 0],
        origin_agent_idx,
        normalize_timestep,
        moving_agent_mask,
        policy.relevant_agent_idxs[focal_id],
    )

    accounted_veh_ids = [policy.idx_to_veh_id[idx] for idx in new_agent_idx_dict.keys()]
    additionally_accounted = [veh_id for veh_id in remaining_veh_ids if veh_id in accounted_veh_ids]
    cur_data_veh_ids = [focal_id] + additionally_accounted

    if t == 0:
        relevant_ids_for_store = list(new_agent_idx_dict.keys())
    else:
        relevant_ids_for_store = relevant_agent_idxs
    for veh_id in cur_data_veh_ids:
        policy.relevant_agent_idxs[veh_id] = relevant_ids_for_store

    new_origin_agent_idx = new_agent_idx_dict[origin_agent_idx]
    rel_actions = dset.discretize_actions(rel_actions)
    if policy.discretize_rtgs:
        rel_rtgs = dset.discretize_rtgs(rel_rtgs)
    rel_ag_states, rel_road_points, rel_road_types, rel_goals = dset.normalize_scene(
        rel_ag_states,
        road_points,
        road_types,
        rel_goals,
        new_origin_agent_idx,
    )
    motion_data_np = _build_motion_data_np(
        rel_ag_states,
        rel_ag_types,
        rel_goals,
        rel_actions,
        rel_rtgs,
        rel_timesteps,
        rel_moving_agent_mask,
        rel_road_points,
        rel_road_types,
    )
    veh_ids_in_context = [policy.idx_to_veh_id[idx] for idx in policy.relevant_agent_idxs[focal_id]]
    focal_batch = {
        "focal_id": focal_id,
        "motion_data_np": motion_data_np,
        "new_agent_idx_dict": {int(k): int(v) for k, v in new_agent_idx_dict.items()},
        "data_veh_ids": cur_data_veh_ids,
        "veh_ids_in_context": veh_ids_in_context,
        "predict_rtgs": bool(policy.predict_rtgs),
    }
    return focal_batch, additionally_accounted, False


def _validate_model_outputs_payload(model_outputs: Dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_MODEL_OUTPUT_KEYS if key not in model_outputs]
    if missing:
        raise ValueError(f"model_outputs missing required keys: {missing}")
    if model_outputs["status"] not in {"ok", "skip"}:
        raise ValueError(f"model_outputs has invalid status={model_outputs['status']!r}")


def prepare_step(adapter: Any, t: int, vehicles: List[Any]) -> Optional[Dict[str, Any]]:
    """构建 prepared_dict 供主进程 ExternalTeacher.batched_forward() 使用。"""
    if adapter._policy is None or len(vehicles) == 0:
        return None

    adapter._last_vehicles = vehicles
    if t < adapter.history_steps - 1:
        adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}
    else:
        adapter._last_vehicle_by_id = {}

    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(t, vehicles, adapter._vehicle_data_dict)
    adapter._policy.update_state(adapter._vehicle_data_dict, adapter._vehicles_to_control, t)

    focal_batches, dead_ids = build_focal_batches(adapter, t)
    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1
    if not focal_batches and not dead_ids:
        return pack_prepared({"status": "skip", "step_t": t, "token_index": token_index, "dead_ids": []})

    tilt_by_veh_id: Dict[int, Tuple[int, int, int]] = {}
    if adapter.per_vehicle_tilting:
        tilt_by_veh_id = dict(adapter.per_vehicle_tilting)

    prepared_dict = {
        "status": "ok",
        "step_t": t,
        "token_index": token_index,
        "dead_ids": dead_ids,
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


def build_focal_batches(adapter: Any, t: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """执行 get_data() 逻辑，在 from_numpy() 之前停下，返回 (focal_batches, dead_ids)。"""
    policy = adapter._policy
    moving_ids = np.where(
        np.linalg.norm(policy.states[:, 0, :2] - policy.goals[:, 0, :2], axis=1)
        > policy.cfg_rl_waymo.moving_threshold
    )[0]
    moving_agent_mask = np.isin(np.arange(policy.states.shape[0]), moving_ids)

    ag_states, ag_types, actions, rtgs, goals, timesteps = _slice_policy_window(policy, t)
    _normalize_rtgs_inplace(rtgs, policy.cfg_rl_waymo)

    dead_ids: List[int] = []
    focal_batches: List[Dict[str, Any]] = []
    unaccounted_veh_ids = _get_control_vehicle_queue(adapter)
    while unaccounted_veh_ids:
        focal_id = unaccounted_veh_ids.pop(0)
        focal_batch, additionally_accounted, is_dead = _build_focal_batch(
            adapter=adapter,
            t=t,
            focal_id=focal_id,
            remaining_veh_ids=unaccounted_veh_ids,
            ag_states=ag_states,
            ag_types=ag_types,
            actions=actions,
            rtgs=rtgs,
            goals=goals,
            timesteps=timesteps,
            moving_agent_mask=moving_agent_mask,
        )
        if is_dead:
            dead_ids.append(focal_id)
            continue

        for veh_id in additionally_accounted:
            unaccounted_veh_ids.remove(veh_id)
        focal_batches.append(focal_batch)

    return focal_batches, dead_ids


def apply_predictions(adapter: Any, model_outputs: Optional[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    """接收主进程推理结果，写回 vehicle_data_dict，返回 opponent actions。"""
    if adapter._policy is None:
        return {}

    model_outputs = unpack_model_outputs(model_outputs)
    if model_outputs is None:
        return {}
    _validate_model_outputs_payload(model_outputs)

    step_t = int(model_outputs["step_t"])
    action_results = model_outputs["action_results"]
    rtg_results = model_outputs["rtg_results"]
    processed_rtg_veh_ids = set(model_outputs["processed_rtg_veh_ids"])
    dead_ids = set(model_outputs["dead_ids"])

    for veh_id, (goal_val, veh_val, road_val) in rtg_results.items():
        if veh_id not in adapter._vehicle_data_dict:
            raise ValueError(f"Unknown veh_id={veh_id} in rtg_results at step_t={step_t}")
        adapter._vehicle_data_dict[veh_id]["next_rtg_goal"] = goal_val
        adapter._vehicle_data_dict[veh_id]["next_rtg_veh"] = veh_val
        adapter._vehicle_data_dict[veh_id]["next_rtg_road"] = road_val

    if adapter._policy.predict_rtgs:
        for veh_id, veh_data in adapter._vehicle_data_dict.items():
            if veh_id in processed_rtg_veh_ids:
                required_rtg_keys = ("next_rtg_goal", "next_rtg_veh", "next_rtg_road")
                missing = [key for key in required_rtg_keys if key not in veh_data]
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
                        ]
                    )
                )
            else:
                veh_data["rtgs"].append(np.array([0] * adapter._policy.cfg_model.num_reward_components))

    for veh_id, (accel, steer) in action_results.items():
        if veh_id not in adapter._vehicle_data_dict:
            raise ValueError(f"Unknown veh_id={veh_id} in action_results at step_t={step_t}")
        adapter._vehicle_data_dict[veh_id]["next_acceleration"] = accel
        adapter._vehicle_data_dict[veh_id]["next_steering"] = steer

    for veh_id in dead_ids:
        if veh_id not in adapter._vehicle_data_dict:
            raise ValueError(f"Unknown veh_id={veh_id} in dead_ids at step_t={step_t}")
        adapter._vehicle_data_dict[veh_id]["next_acceleration"] = 0.0
        adapter._vehicle_data_dict[veh_id]["next_steering"] = 0.0

    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in adapter._vehicles_to_control:
        if step_t < adapter.history_steps - 1:
            veh = adapter._last_vehicle_by_id.get(veh_id)
            actions[veh_id] = adapter._get_gt_action(veh_id, step_t, veh)
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
