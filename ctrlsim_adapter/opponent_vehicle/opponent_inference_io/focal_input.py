"""
负责从 policy 历史窗口中抽取每个 focal vehicle 的局部 batch 输入。
该模块执行上下文筛选、坐标归一化和 RTG 归一化，生成送往 ExternalTeacher 的核心数据块。
Builds per-focal-vehicle local batch inputs from the policy history window.
Performs context selection, coordinate normalization, and RTG normalization to form ExternalTeacher inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .prepare_inference_payload import (
    get_control_vehicle_queue,
    get_or_create_prepare_buffer,
)


def _format_value_range(values: np.ndarray) -> str:
    """Return a compact min/max summary for logging."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return "all_non_finite"
    return f"min={finite.min():.3f}, max={finite.max():.3f}"


def sanitize_index_array(
    values: np.ndarray,
    *,
    field_name: str,
    step_t: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    """Clamp a discrete index array into the valid integer range."""
    raw_values = np.asarray(values)
    sanitized = np.nan_to_num(
        raw_values,
        nan=float(min_value),
        posinf=float(max_value),
        neginf=float(min_value),
    )
    sanitized = np.clip(sanitized, min_value, max_value).astype(np.int64, copy=False)
    if not np.array_equal(raw_values, sanitized):
        print(
            f"[focal_input] sanitize[{field_name}] step={int(step_t)} "
            f"{_format_value_range(raw_values)} -> {_format_value_range(sanitized)}"
        )
    return sanitized


def slice_policy_window(
    policy: Any,
    t: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def normalize_rtgs_inplace(rtgs: np.ndarray, rl: Any) -> None:
    rtgs[:, :, 0] = (np.clip(rtgs[:, :, 0], rl.min_rtg_pos, rl.max_rtg_pos) - rl.min_rtg_pos) / (
        rl.max_rtg_pos - rl.min_rtg_pos
    )
    rtgs[:, :, 1] = (np.clip(rtgs[:, :, 1], rl.min_rtg_veh, rl.max_rtg_veh) - rl.min_rtg_veh) / (
        rl.max_rtg_veh - rl.min_rtg_veh
    )
    rtgs[:, :, 2] = (np.clip(rtgs[:, :, 2], rl.min_rtg_road, rl.max_rtg_road) - rl.min_rtg_road) / (
        rl.max_rtg_road - rl.min_rtg_road
    )


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


def select_relevant_agents_fast(
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


def normalize_scene_fast(
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
    if max_roads <= 0:
        return rel_ag_states, final_road_points, final_road_types, rel_goals

    num_roads = int(road_points_src.shape[0])
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
        final_road_points[:selected_count] = normalized_road_points[:max_roads]
        final_road_types[:selected_count] = selected_road_types[:max_roads]

    return rel_ag_states, final_road_points, final_road_types, rel_goals


def build_focal_batch(
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
    moving_agent_mask: np.ndarray,
    road_points_src: np.ndarray,
    road_types_src: np.ndarray,
    predict_rtgs: Optional[bool] = None,
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

    (
        rel_ag_states,
        rel_ag_types,
        rel_actions,
        rel_rtgs,
        rel_goals,
        rel_moving_agent_mask,
        new_agent_idx_dict,
        relevant_agent_idxs,
    ) = select_relevant_agents_fast(
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

    rel_ag_states, rel_road_points, rel_road_types, rel_goals = normalize_scene_fast(
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
        "predict_rtgs": bool(
            policy.predict_rtgs if predict_rtgs is None else predict_rtgs
        ),
    }
    return focal_batch, additionally_accounted, False


def build_focal_batches(
    adapter: Any,
    t: int,
    focal_vehicle_ids: Optional[List[int]] = None,
    predict_rtgs: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], List[int], np.ndarray]:
    """为指定 focal 车辆构建 batch 输入。 / Build batched focal inputs for the specified vehicles."""
    shared_context = build_step_shared_context(adapter, t)
    return build_focal_batches_from_shared_context(
        adapter,
        t,
        shared_context,
        focal_vehicle_ids=focal_vehicle_ids,
        predict_rtgs=predict_rtgs,
    )


def build_step_shared_context(
    adapter: Any,
    t: int,
) -> Dict[str, Any]:
    """构建单个 step 共享的预处理上下文。 / Build the shared per-step preprocessing context."""
    policy = adapter._policy
    moving_agent_mask = adapter._moving_agent_mask_cache
    if moving_agent_mask is None:
        moving_ids = np.where(
            np.linalg.norm(policy.states[:, 0, :2] - policy.goals[:, 0, :2], axis=1)
            > policy.cfg_rl_waymo.moving_threshold
        )[0]
        moving_agent_mask = np.isin(np.arange(policy.states.shape[0]), moving_ids)
        adapter._moving_agent_mask_cache = moving_agent_mask

    ag_states, ag_types, actions_src, rtgs_src, goals, timesteps_src = slice_policy_window(policy, t)
    actions_buffer = get_or_create_prepare_buffer(
        adapter=adapter,
        name="actions_buffer",
        shape=(actions_src.shape[0] + 1, *actions_src.shape[1:]),
        dtype=actions_src.dtype,
    )
    actions_buffer.fill(0)
    actions_buffer[: actions_src.shape[0]] = actions_src
    actions_discrete = adapter.dataset.discretize_actions(actions_buffer)
    action_dim = int(policy.action_dim)
    actions_values = sanitize_index_array(
        actions_discrete[: actions_src.shape[0]],
        field_name="actions",
        step_t=t,
        min_value=0,
        max_value=action_dim - 1,
    )
    adapter._pad_action_values_step = sanitize_index_array(
        np.asarray(actions_discrete[actions_src.shape[0]]).copy(),
        field_name="pad_actions",
        step_t=t,
        min_value=0,
        max_value=action_dim - 1,
    )

    rtgs = get_or_create_prepare_buffer(
        adapter=adapter,
        name="rtgs_norm_buffer",
        shape=rtgs_src.shape,
        dtype=rtgs_src.dtype,
    )
    np.copyto(rtgs, rtgs_src)
    normalize_rtgs_inplace(rtgs, policy.cfg_rl_waymo)
    if policy.discretize_rtgs:
        rtgs_discrete = get_or_create_prepare_buffer(
            adapter=adapter,
            name="rtgs_discrete_buffer",
            shape=(rtgs.shape[0] + 1, *rtgs.shape[1:]),
            dtype=rtgs.dtype,
        )
        rtgs_discrete.fill(0)
        rtgs_discrete[: rtgs.shape[0]] = rtgs
        rtgs_discrete_values = adapter.dataset.discretize_rtgs(rtgs_discrete)
        rtgs_values = sanitize_index_array(
            rtgs_discrete_values[: rtgs.shape[0]],
            field_name="rtgs",
            step_t=t,
            min_value=0,
            max_value=int(policy.cfg_rl_waymo.rtg_discretization) - 1,
        )
        adapter._pad_rtg_values_step = sanitize_index_array(
            np.asarray(rtgs_discrete_values[rtgs.shape[0]]).copy(),
            field_name="pad_rtgs",
            step_t=t,
            min_value=0,
            max_value=int(policy.cfg_rl_waymo.rtg_discretization) - 1,
        )
    else:
        rtgs_values = rtgs
        adapter._pad_rtg_values_step = np.zeros(rtgs.shape[1:], dtype=rtgs.dtype)

    timesteps = get_or_create_prepare_buffer(
        adapter=adapter,
        name="timesteps_int_buffer",
        shape=timesteps_src.shape,
        dtype=np.dtype(np.int64),
    )
    np.copyto(timesteps, timesteps_src, casting="unsafe")
    rel_timesteps_template = get_or_create_prepare_buffer(
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

    return {
        "ag_states": ag_states,
        "ag_types": ag_types,
        "actions_values": actions_values,
        "rtgs_values": rtgs_values,
        "goals_step": goals_step,
        "moving_agent_mask": moving_agent_mask,
        "road_points_src": road_points_src,
        "road_types_src": road_types_src,
        "shared_timesteps": rel_timesteps_template,
    }


def build_focal_batches_from_shared_context(
    adapter: Any,
    t: int,
    shared_context: Dict[str, Any],
    focal_vehicle_ids: Optional[List[int]] = None,
    predict_rtgs: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], List[int], np.ndarray]:
    """基于共享上下文构建 focal batches。 / Build focal batches from a shared step context."""
    ag_states = shared_context["ag_states"]
    ag_types = shared_context["ag_types"]
    actions_values = shared_context["actions_values"]
    rtgs_values = shared_context["rtgs_values"]
    goals_step = shared_context["goals_step"]
    moving_agent_mask = shared_context["moving_agent_mask"]
    road_points_src = shared_context["road_points_src"]
    road_types_src = shared_context["road_types_src"]
    shared_timesteps = shared_context["shared_timesteps"]

    dead_ids: List[int] = []
    focal_batches: List[Dict[str, Any]] = []
    if focal_vehicle_ids is None:
        unaccounted_veh_ids = get_control_vehicle_queue(adapter)
    else:
        unaccounted_veh_ids = list(focal_vehicle_ids)
    unaccounted_veh_id_set = set(unaccounted_veh_ids)
    while unaccounted_veh_ids and unaccounted_veh_id_set:
        focal_id = unaccounted_veh_ids.pop(0)
        if focal_id not in unaccounted_veh_id_set:
            continue
        unaccounted_veh_id_set.remove(focal_id)

        focal_batch, additionally_accounted, is_dead = build_focal_batch(
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
            moving_agent_mask=moving_agent_mask,
            road_points_src=road_points_src,
            road_types_src=road_types_src,
            predict_rtgs=predict_rtgs,
        )
        if is_dead:
            dead_ids.append(focal_id)
            continue

        for veh_id in additionally_accounted:
            unaccounted_veh_id_set.discard(veh_id)
        focal_batches.append(focal_batch)

    return focal_batches, dead_ids, shared_timesteps
