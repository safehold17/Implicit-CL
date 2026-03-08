from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


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

