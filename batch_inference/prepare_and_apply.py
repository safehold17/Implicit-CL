"""Batch inference 模式下 adapter 侧的数据准备与结果应用。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ipc_codec import pack_prepared, unpack_model_outputs, validate_model_outputs_payload

_NEXT_RTG_KEYS = ("next_rtg_goal", "next_rtg_veh", "next_rtg_road")


def _slice_policy_window(policy: Any, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tcl = policy.cfg_rl_waymo.train_context_length
    start = 0 if t < tcl else t - (tcl - 1)
    end = tcl if t < tcl else t + 1

    return (
        policy.states[:, start:end],
        policy.types,
        policy.actions[:, start:end],
        policy.rtgs[:, start:end],
        policy.goals[:, start:end],
        policy.timesteps[0, start:end],
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


def _get_or_create_prepare_buffer(
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


def _get_control_vehicle_queue(adapter: Any) -> List[int]:
    return list(adapter._vehicles_to_control_sorted or adapter._vehicles_to_control)


def _require_vehicle_data(
    vehicle_data_dict: Dict[int, Dict[str, Any]],
    veh_id: int,
    source_name: str,
    step_t: int,
) -> Dict[str, Any]:
    veh_data = vehicle_data_dict.get(veh_id)
    if veh_data is None:
        raise ValueError(f"Unknown veh_id={veh_id} in {source_name} at step_t={step_t}")
    return veh_data


def _get_step_controlled_ids(adapter: Any) -> List[int]:
    step_ids = list(getattr(adapter, "_controlled_vehicle_ids_step", []))
    if step_ids:
        return step_ids
    return list(adapter._vehicles_to_control)


def _build_sparse_repeat_actions(
    adapter: Any,
    step_t: int,
) -> Dict[int, Tuple[float, float]]:
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in _get_step_controlled_ids(adapter):
        veh_data = _require_vehicle_data(
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


def _build_warmup_gt_actions(
    adapter: Any,
    step_t: int,
) -> Dict[int, Tuple[float, float]]:
    actions: Dict[int, Tuple[float, float]] = {}
    for veh_id in _get_step_controlled_ids(adapter):
        veh_data = _require_vehicle_data(
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


def _clear_pending_sparse_actions(adapter: Any) -> None:
    adapter._pending_sparse_actions_step_t = None
    adapter._pending_sparse_actions = {}


def _set_pending_sparse_actions(
    adapter: Any,
    step_t: int,
    actions: Dict[int, Tuple[float, float]],
) -> None:
    adapter._pending_sparse_actions_step_t = int(step_t)
    adapter._pending_sparse_actions = dict(actions)


def _consume_pending_sparse_actions(
    adapter: Any,
    step_t: Optional[int] = None,
) -> Optional[Dict[int, Tuple[float, float]]]:
    pending_step = getattr(adapter, "_pending_sparse_actions_step_t", None)
    pending_actions = dict(getattr(adapter, "_pending_sparse_actions", {}))
    if pending_step is None:
        return None
    if step_t is not None and int(pending_step) != int(step_t):
        return None
    _clear_pending_sparse_actions(adapter)
    return pending_actions


def _set_predict_rtgs_override(adapter: Any, step_t: int, predict_rtgs: bool) -> None:
    policy = adapter._policy
    if policy is None:
        return
    adapter._predict_rtgs_override_prev = bool(policy.predict_rtgs)
    adapter._predict_rtgs_override_step_t = int(step_t)
    policy.predict_rtgs = bool(predict_rtgs)


def _clear_predict_rtgs_override(adapter: Any, step_t: Optional[int] = None) -> None:
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


def _angle_sub_np(current_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    diff = (target_angle - current_angle) % (2 * np.pi)
    mask = diff > np.pi
    diff[mask] = -(2 * np.pi - diff[mask])
    return diff


def _apply_se2_transform_np(
    coordinates: np.ndarray,
    translation: np.ndarray,
    yaw: float,
) -> np.ndarray:
    shifted = coordinates - translation
    x = shifted[..., 0]
    y = shifted[..., 1]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    out = np.empty_like(shifted)
    out[..., 0] = cos_yaw * x - sin_yaw * y
    out[..., 1] = sin_yaw * x + cos_yaw * y
    return out


def _select_relevant_agents_fast(
    dset: Any,
    ag_states: np.ndarray,
    ag_types: np.ndarray,
    actions_values: np.ndarray,
    pad_action_values: np.ndarray,
    rtgs_values: np.ndarray,
    pad_rtg_values: np.ndarray,
    goals_step: np.ndarray,
    origin_agent_idx: int,
    moving_agent_mask: np.ndarray,
    cached_relevant_agent_idxs: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], List[int]]:
    cfg = dset.cfg_dataset
    max_agents = int(cfg.max_num_agents)
    num_agents = int(ag_states.shape[0])
    origin_xy = ag_states[origin_agent_idx, 0, :2]
    delta = ag_states[:, 0, :2] - origin_xy[np.newaxis, :]
    dist_sq = np.sum(delta * delta, axis=-1)
    valid_mask = dist_sq < float(cfg.agent_dist_threshold) ** 2

    has_cached = len(cached_relevant_agent_idxs) > 0
    if has_cached:
        cached = np.asarray(cached_relevant_agent_idxs, dtype=np.int64)
        in_range_mask = (cached >= 0) & (cached < num_agents)
        cached = cached[in_range_mask]
        if cached.size > 0:
            valid_cached = cached[valid_mask[cached]]
            closest_ag_ids = np.unique(valid_cached)
        else:
            closest_ag_ids = np.empty((0,), dtype=np.int64)
        valid_cached_set = set(int(idx) for idx in closest_ag_ids.tolist())
        relevant_agent_idxs = [int(idx) for idx in cached_relevant_agent_idxs if int(idx) in valid_cached_set]
    else:
        candidate_count = min(max_agents, num_agents)
        if candidate_count <= 0:
            closest_ag_ids = np.empty((0,), dtype=np.int64)
        elif candidate_count < num_agents:
            candidate_idxs = np.argpartition(dist_sq, candidate_count - 1)[:candidate_count]
            closest_ag_ids = candidate_idxs[valid_mask[candidate_idxs]]
        else:
            candidate_idxs = np.arange(num_agents, dtype=np.int64)
            closest_ag_ids = candidate_idxs[valid_mask[candidate_idxs]]
        if closest_ag_ids.size > 1:
            closest_ag_ids = np.sort(closest_ag_ids)
            if dset.split_name == "train":
                np.random.shuffle(closest_ag_ids)
        relevant_agent_idxs = []

    if closest_ag_ids.size > max_agents:
        closest_ag_ids = closest_ag_ids[:max_agents]

    final_agent_states = np.zeros((max_agents, *ag_states.shape[1:]), dtype=ag_states.dtype)
    final_agent_types = -np.ones((max_agents, *ag_types.shape[1:]), dtype=ag_types.dtype)
    final_actions = np.empty((max_agents, *actions_values.shape[1:]), dtype=actions_values.dtype)
    final_actions[:] = pad_action_values
    final_rtgs = np.empty((max_agents, *rtgs_values.shape[1:]), dtype=rtgs_values.dtype)
    final_rtgs[:] = pad_rtg_values
    final_goals = np.zeros((max_agents, *goals_step.shape[1:]), dtype=goals_step.dtype)
    final_moving_agent_mask = np.zeros(max_agents, dtype=moving_agent_mask.dtype)

    num_selected = int(closest_ag_ids.shape[0])
    if num_selected > 0:
        final_agent_states[:num_selected] = ag_states[closest_ag_ids]
        final_agent_types[:num_selected] = ag_types[closest_ag_ids]
        final_actions[:num_selected] = actions_values[closest_ag_ids]
        final_rtgs[:num_selected] = rtgs_values[closest_ag_ids]
        final_goals[:num_selected] = goals_step[closest_ag_ids]
        final_moving_agent_mask[:num_selected] = moving_agent_mask[closest_ag_ids]

    new_agent_idx_dict = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(closest_ag_ids.tolist())}
    return (
        final_agent_states,
        final_agent_types,
        final_actions,
        final_rtgs,
        final_goals,
        final_moving_agent_mask,
        new_agent_idx_dict,
        relevant_agent_idxs,
    )


def _normalize_scene_fast(
    rl: Any,
    rel_ag_states: np.ndarray,
    rel_goals: np.ndarray,
    new_origin_agent_idx: int,
    road_points_src: np.ndarray,
    road_types_src: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = float(rel_ag_states[new_origin_agent_idx, 0, 4])
    angle_of_rotation = (np.pi / 2.0) + np.sign(-yaw) * np.abs(yaw)
    translation_xy = rel_ag_states[new_origin_agent_idx, 0, :2].copy()
    translation = translation_xy[np.newaxis, np.newaxis, :]
    zero_translation = np.zeros_like(translation)

    rel_ag_states[:, :, :2] = _apply_se2_transform_np(
        coordinates=rel_ag_states[:, :, :2],
        translation=translation,
        yaw=angle_of_rotation,
    )
    rel_ag_states[:, :, 2:4] = _apply_se2_transform_np(
        coordinates=rel_ag_states[:, :, 2:4],
        translation=zero_translation,
        yaw=angle_of_rotation,
    )
    rel_ag_states[:, :, 4] = _angle_sub_np(
        rel_ag_states[:, :, 4],
        -np.asarray(angle_of_rotation).reshape(1, 1),
    )

    rel_goals[:, :2] = _apply_se2_transform_np(
        coordinates=rel_goals[:, :2],
        translation=translation[:, 0],
        yaw=angle_of_rotation,
    )
    if int(rl.goal_dim) == 5:
        rel_goals[:, 2:4] = _apply_se2_transform_np(
            coordinates=rel_goals[:, 2:4],
            translation=np.zeros_like(translation[:, 0]),
            yaw=angle_of_rotation,
        )
        rel_goals[:, 4] = _angle_sub_np(
            rel_goals[:, 4],
            -np.asarray(angle_of_rotation).reshape(1),
        )

    max_roads = int(rl.max_num_road_polylines)
    final_road_points = np.zeros((max_roads, *road_points_src.shape[1:]), dtype=road_points_src.dtype)
    final_road_types = -np.ones((max_roads, *road_types_src.shape[1:]), dtype=road_types_src.dtype)

    num_roads = int(road_points_src.shape[0])
    if max_roads <= 0:
        return rel_ag_states, final_road_points, final_road_types, rel_goals

    if num_roads > 0:
        selected_road_points = road_points_src
        selected_road_types = road_types_src
        if num_roads > max_roads:
            road_valid_mask = road_points_src[:, :, -1]
            road_delta = road_points_src[:, :, :2] - translation_xy[np.newaxis, np.newaxis, :]
            road_dist_sq = np.sum(road_delta * road_delta, axis=-1) * road_valid_mask
            selected = np.argsort(np.max(road_dist_sq, axis=1))[:max_roads]
            selected_road_points = road_points_src[selected]
            selected_road_types = road_types_src[selected]

        normalized_road_points = selected_road_points.copy()
        normalized_road_points[:, :, :2] = _apply_se2_transform_np(
            coordinates=normalized_road_points[:, :, :2],
            translation=translation,
            yaw=angle_of_rotation,
        )
        selected_count = int(normalized_road_points.shape[0])
        if selected_count > max_roads:
            final_road_points[:] = normalized_road_points[:max_roads]
            final_road_types[:] = selected_road_types[:max_roads]
        else:
            final_road_points[:selected_count] = normalized_road_points
            final_road_types[:selected_count] = selected_road_types

    return rel_ag_states, final_road_points, final_road_types, rel_goals


def _build_focal_batch(
    adapter: Any,
    t: int,
    focal_id: int,
    remaining_veh_ids: List[int],
    remaining_veh_id_set: set[int],
    ag_states: np.ndarray,
    ag_types: np.ndarray,
    actions_values: np.ndarray,
    rtgs_values: np.ndarray,
    goals_step: np.ndarray,
    rel_timesteps_template: np.ndarray,
    moving_agent_mask: np.ndarray,
    road_points_src: np.ndarray,
    road_types_src: np.ndarray,
) -> Tuple[Optional[Dict[str, Any]], List[int], bool]:
    policy = adapter._policy
    dset = adapter.dataset
    rl = policy.cfg_rl_waymo

    origin_agent_idx = policy.veh_id_to_idx[focal_id]
    if not policy.states[origin_agent_idx, t, -1]:
        return None, [], True

    if len(road_points_src) == 0:
        return None, [], True

    has_cached_relevant_agents = focal_id in policy.relevant_agent_idxs
    cached_relevant_agent_idxs = policy.relevant_agent_idxs.get(focal_id, [])

    rel_timesteps = rel_timesteps_template
    (
        rel_ag_states,
        rel_ag_types,
        rel_actions,
        rel_rtgs,
        rel_goals,
        rel_moving_agent_mask,
        new_agent_idx_dict,
        relevant_agent_idxs,
    ) = _select_relevant_agents_fast(
        dset=dset,
        ag_states=ag_states,
        ag_types=ag_types,
        actions_values=actions_values,
        pad_action_values=adapter._pad_action_values_step,
        rtgs_values=rtgs_values,
        pad_rtg_values=adapter._pad_rtg_values_step,
        goals_step=goals_step,
        origin_agent_idx=origin_agent_idx,
        moving_agent_mask=moving_agent_mask,
        cached_relevant_agent_idxs=cached_relevant_agent_idxs,
    )

    accounted_veh_ids = [policy.idx_to_veh_id[idx] for idx in new_agent_idx_dict.keys()]
    cur_data_veh_ids = [focal_id]
    additionally_accounted: List[int] = []
    for veh_id in remaining_veh_ids:
        if veh_id in remaining_veh_id_set and veh_id in accounted_veh_ids:
            cur_data_veh_ids.append(veh_id)
            additionally_accounted.append(veh_id)
            remaining_veh_id_set.discard(veh_id)
            remaining_veh_ids.remove(veh_id)

    if not has_cached_relevant_agents:
        relevant_ids_for_store = list(new_agent_idx_dict.keys())
    else:
        relevant_ids_for_store = relevant_agent_idxs
    for veh_id in cur_data_veh_ids:
        policy.relevant_agent_idxs[veh_id] = relevant_ids_for_store

    new_origin_agent_idx = new_agent_idx_dict.get(origin_agent_idx)
    if new_origin_agent_idx is None:
        return None, [], True
    rel_ag_states, rel_road_points, rel_road_types, rel_goals = _normalize_scene_fast(
        rl=rl,
        rel_ag_states=rel_ag_states,
        rel_goals=rel_goals,
        new_origin_agent_idx=new_origin_agent_idx,
        road_points_src=road_points_src,
        road_types_src=road_types_src,
    )
    motion_data_np = {
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


def prepare_step(adapter: Any, t: int, vehicles: List[Any]) -> Optional[Dict[str, Any]]:
    """构建 prepared_dict 供主进程 ExternalTeacher.batched_forward() 使用。"""
    if adapter._policy is None or len(vehicles) == 0:
        _clear_pending_sparse_actions(adapter)
        _clear_predict_rtgs_override(adapter)
        return None

    _clear_predict_rtgs_override(adapter)
    adapter._last_vehicles = vehicles
    adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}

    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(t, vehicles, adapter._vehicle_data_dict)
    adapter.update_policy_state(t)

    # warm-up 阶段仍需要执行 predict() 来推进 RTG 历史，但最终动作仍回退到 GT。
    if t < adapter.history_steps - 1:
        warmup_actions = _build_warmup_gt_actions(adapter, t)
        _set_pending_sparse_actions(adapter, step_t=t, actions=warmup_actions)

    is_sparse_step = adapter.sparse_inference.is_sparse_step(
        t=t,
        history_steps=adapter.history_steps,
    )
    if is_sparse_step and adapter.sparse_inference_action_repeat:
        actions = _build_sparse_repeat_actions(adapter, t)
        _set_pending_sparse_actions(adapter, step_t=t, actions=actions)
        return None

    _clear_pending_sparse_actions(adapter)
    if is_sparse_step:
        _set_predict_rtgs_override(adapter, step_t=t, predict_rtgs=False)

    focal_batches, dead_ids = build_focal_batches(adapter, t)
    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1
    if not focal_batches and not dead_ids:
        return pack_prepared({"status": "skip", "step_t": t, "token_index": token_index, "dead_ids": []})

    tilt_by_veh_id: Dict[int, Tuple[int, int, int]] = (
        dict(adapter.per_vehicle_tilting) if adapter.per_vehicle_tilting else {}
    )

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
    moving_agent_mask = adapter._moving_agent_mask_cache
    if moving_agent_mask is None:
        moving_ids = np.where(
            np.linalg.norm(policy.states[:, 0, :2] - policy.goals[:, 0, :2], axis=1)
            > policy.cfg_rl_waymo.moving_threshold
        )[0]
        moving_agent_mask = np.isin(np.arange(policy.states.shape[0]), moving_ids)
        adapter._moving_agent_mask_cache = moving_agent_mask

    ag_states, ag_types, actions_src, rtgs_src, goals, timesteps_src = _slice_policy_window(policy, t)
    actions_buffer = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="actions_buffer",
        shape=(actions_src.shape[0] + 1, *actions_src.shape[1:]),
        dtype=actions_src.dtype,
    )
    actions_buffer.fill(0)
    actions_buffer[: actions_src.shape[0]] = actions_src
    actions_discrete = adapter.dataset.discretize_actions(actions_buffer)
    actions_values = actions_discrete[: actions_src.shape[0]]
    adapter._pad_action_values_step = np.asarray(actions_discrete[actions_src.shape[0]]).copy()
    rtgs = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="rtgs_norm_buffer",
        shape=rtgs_src.shape,
        dtype=rtgs_src.dtype,
    )
    np.copyto(rtgs, rtgs_src)
    _normalize_rtgs_inplace(rtgs, policy.cfg_rl_waymo)
    if policy.discretize_rtgs:
        rtgs_discrete = _get_or_create_prepare_buffer(
            adapter=adapter,
            name="rtgs_discrete_buffer",
            shape=(rtgs.shape[0] + 1, *rtgs.shape[1:]),
            dtype=rtgs.dtype,
        )
        rtgs_discrete.fill(0)
        rtgs_discrete[: rtgs.shape[0]] = rtgs
        rtgs_discrete_values = adapter.dataset.discretize_rtgs(rtgs_discrete)
        rtgs_values = rtgs_discrete_values[: rtgs.shape[0]]
        adapter._pad_rtg_values_step = np.asarray(rtgs_discrete_values[rtgs.shape[0]]).copy()
    else:
        rtgs_values = rtgs
        adapter._pad_rtg_values_step = np.zeros(rtgs.shape[1:], dtype=rtgs.dtype)

    timesteps = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="timesteps_int_buffer",
        shape=timesteps_src.shape,
        dtype=np.dtype(np.int64),
    )
    np.copyto(timesteps, timesteps_src, casting="unsafe")
    rel_timesteps_template = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="rel_timesteps_template",
        shape=(
            policy.cfg_rl_waymo.max_num_agents,
            timesteps.shape[0],
            timesteps.shape[1],
        ),
        dtype=timesteps.dtype,
    )
    rel_timesteps_template[...] = timesteps[np.newaxis, ...]
    road_points_src = adapter._preproc_data.get("road_points")
    road_types_src = adapter._preproc_data.get("road_types")
    if road_points_src is None:
        road_points_src = np.zeros((0, 1, 3), dtype=np.float32)
    if road_types_src is None:
        road_types_src = np.zeros((0, 1), dtype=np.float32)
    goals_step = goals[:, 0]

    dead_ids: List[int] = []
    focal_batches: List[Dict[str, Any]] = []
    unaccounted_veh_ids = _get_control_vehicle_queue(adapter)
    unaccounted_veh_id_set = set(unaccounted_veh_ids)
    while unaccounted_veh_ids and unaccounted_veh_id_set:
        focal_id = unaccounted_veh_ids.pop(0)
        if focal_id not in unaccounted_veh_id_set:
            continue
        unaccounted_veh_id_set.remove(focal_id)
        focal_batch, additionally_accounted, is_dead = _build_focal_batch(
            adapter=adapter,
            t=t,
            focal_id=focal_id,
            remaining_veh_ids=unaccounted_veh_ids,
            remaining_veh_id_set=unaccounted_veh_id_set,
            ag_states=ag_states,
            ag_types=ag_types,
            actions_values=actions_values,
            rtgs_values=rtgs_values,
            goals_step=goals_step,
            rel_timesteps_template=rel_timesteps_template,
            moving_agent_mask=moving_agent_mask,
            road_points_src=road_points_src,
            road_types_src=road_types_src,
        )
        if is_dead:
            dead_ids.append(focal_id)
            continue

        for veh_id in additionally_accounted:
            unaccounted_veh_id_set.discard(veh_id)
        focal_batches.append(focal_batch)

    return focal_batches, dead_ids


def apply_predictions(adapter: Any, model_outputs: Optional[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    """接收主进程推理结果，写回 vehicle_data_dict，返回 opponent actions。"""
    if adapter._policy is None:
        return {}

    model_outputs = unpack_model_outputs(model_outputs)
    if model_outputs is None:
        pending_actions = _consume_pending_sparse_actions(adapter)
        _clear_predict_rtgs_override(adapter)
        if pending_actions is not None:
            return pending_actions
        return {}
    validate_model_outputs_payload(model_outputs)

    step_t = int(model_outputs["step_t"])
    try:
        status = str(model_outputs["status"])
        if status == "skip":
            pending_actions = _consume_pending_sparse_actions(adapter, step_t=step_t)
            if pending_actions is not None:
                return pending_actions
            return {}

        action_results = model_outputs["action_results"]
        rtg_results = model_outputs["rtg_results"]
        processed_rtg_veh_ids = set(model_outputs["processed_rtg_veh_ids"])
        dead_ids = set(model_outputs["dead_ids"])

        for veh_id, (goal_val, veh_val, road_val) in rtg_results.items():
            veh_data = _require_vehicle_data(adapter._vehicle_data_dict, veh_id, "rtg_results", step_t)
            veh_data["next_rtg_goal"] = goal_val
            veh_data["next_rtg_veh"] = veh_val
            veh_data["next_rtg_road"] = road_val

        if adapter._policy.predict_rtgs:
            for veh_id, veh_data in adapter._vehicle_data_dict.items():
                if veh_id in processed_rtg_veh_ids:
                    missing = [key for key in _NEXT_RTG_KEYS if key not in veh_data]
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
                    veh_data["rtgs"].append(
                        np.array([0] * adapter._policy.cfg_model.num_reward_components)
                    )

        for veh_id, (accel, steer) in action_results.items():
            veh_data = _require_vehicle_data(adapter._vehicle_data_dict, veh_id, "action_results", step_t)
            veh_data["next_acceleration"] = accel
            veh_data["next_steering"] = steer

        for veh_id in dead_ids:
            veh_data = _require_vehicle_data(adapter._vehicle_data_dict, veh_id, "dead_ids", step_t)
            veh_data["next_acceleration"] = 0.0
            veh_data["next_steering"] = 0.0

        if step_t < adapter.history_steps - 1:
            _consume_pending_sparse_actions(adapter, step_t=step_t)

        actions: Dict[int, Tuple[float, float]] = {}
        for veh_id in _get_step_controlled_ids(adapter):
            veh_data = _require_vehicle_data(
                adapter._vehicle_data_dict,
                veh_id,
                "controlled_vehicle_ids_step",
                step_t,
            )
            if step_t < adapter.history_steps - 1:
                veh = adapter._last_vehicle_by_id.get(veh_id)
                actions[veh_id] = adapter._get_gt_action(veh_id, step_t, veh)
                continue

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
    finally:
        _clear_predict_rtgs_override(adapter, step_t=step_t)
