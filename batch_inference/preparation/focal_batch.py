from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scene import normalize_rtgs_inplace, normalize_scene_fast, select_relevant_agents_fast, slice_policy_window
from .shared import get_control_vehicle_queue, get_or_create_prepare_buffer


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
    # Match ctrlsim.get_data() exactly: consume the remaining queue in place so
    # downstream focal grouping stays byte-for-byte aligned with no-batch mode.
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
        "timesteps": rel_timesteps_template,
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


def build_focal_batches(adapter: Any, t: int) -> Tuple[List[Dict[str, Any]], List[int]]:
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
    actions_values = actions_discrete[: actions_src.shape[0]]
    adapter._pad_action_values_step = np.asarray(actions_discrete[actions_src.shape[0]]).copy()

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
        rtgs_values = rtgs_discrete_values[: rtgs.shape[0]]
        adapter._pad_rtg_values_step = np.asarray(rtgs_discrete_values[rtgs.shape[0]]).copy()
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

    dead_ids: List[int] = []
    focal_batches: List[Dict[str, Any]] = []
    unaccounted_veh_ids = get_control_vehicle_queue(adapter)
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
