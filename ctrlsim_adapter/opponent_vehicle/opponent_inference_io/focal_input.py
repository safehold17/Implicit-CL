"""
负责从 policy 历史窗口中抽取每个 focal vehicle 的局部 batch 输入。
该模块执行上下文筛选、坐标归一化和 RTG 归一化，生成送往 ExternalTeacher 的核心数据块。
Builds per-focal-vehicle local batch inputs from the policy history window.
Performs context selection, coordinate normalization, and RTG normalization to form ExternalTeacher inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from batch_inference.batch_ipc.arrays import pack_ragged_int_lists

from ..step_tensor_context import build_policy_step_tensor_context
from .prepare_inference_payload import get_control_vehicle_queue

def _apply_se2_transform_batched_np(
    coordinates: np.ndarray,
    translation: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    """对 batched 坐标执行 SE(2) 变换。

    输入张量的第一个维度视为 batch 维，后续任意维度都保持原样广播。
    该函数用于一次性归一化多个 focal 的 agent / goal / road 坐标。

    Apply an SE(2) transform to batched coordinates.
    The first axis is treated as the batch axis, and all remaining dimensions are preserved
    through broadcasting. This helper normalizes agent/goal/road coordinates for multiple
    focals in one pass.
    """
    shifted = coordinates - translation
    x = shifted[..., 0]
    y = shifted[..., 1]
    expand_shape = (coordinates.shape[0],) + (1,) * (coordinates.ndim - 2)
    cos_yaw = np.cos(yaw).reshape(expand_shape)
    sin_yaw = np.sin(yaw).reshape(expand_shape)
    out = np.empty_like(shifted)
    out[..., 0] = cos_yaw * x - sin_yaw * y
    out[..., 1] = sin_yaw * x + cos_yaw * y
    return out


def _angle_sub_batched_np(
    current_angle: np.ndarray,
    target_angle: np.ndarray,
) -> np.ndarray:
    """对 batched heading 计算包角差。

    该函数允许 `current_angle` 和 `target_angle` 以 batch 维对齐，
    从而在一次向量化调用中处理多个 focal 的 heading 归一化。

    Compute wrapped angular differences for batched heading tensors.
    Both operands may carry a batch axis so multiple focal heading normalizations can be handled
    in one vectorized call.
    """
    diff = (target_angle - current_angle) % (2 * np.pi)
    mask = diff > np.pi
    diff = diff.copy()
    diff[mask] = -(2 * np.pi - diff[mask])
    return diff


def _resolve_selected_agent_indices(
    adapter: Any,
    ag_states: np.ndarray,
    focal_id: int,
) -> tuple[np.ndarray, List[int], bool]:
    """解析单个 focal 的相关 agent 索引集合。

    该函数负责保留当前实现的语义边界：
    1. 优先复用 `policy.relevant_agent_idxs` 的缓存；
    2. 当缓存缺失时，基于距离阈值重新选择上下文车辆；
    3. 保证返回的索引列表仍是 policy 全局 agent index，而不是局部重排后的 index。

    Resolve the relevant agent-index set for one focal vehicle.
    The helper preserves the current semantics by:
    1. Reusing `policy.relevant_agent_idxs` when available;
    2. Recomputing the context from distance thresholds on cache miss;
    3. Returning policy-global agent indices rather than local reordered indices.
    """
    policy = adapter._policy
    dataset = adapter.dataset
    cfg = dataset.cfg_dataset
    origin_agent_idx = int(policy.veh_id_to_idx[focal_id])
    num_agents = int(ag_states.shape[0])
    max_agents = int(cfg.max_num_agents)

    origin_xy = ag_states[origin_agent_idx, 0, :2]
    delta = ag_states[:, 0, :2] - origin_xy[np.newaxis, :]
    dist_sq = np.sum(delta * delta, axis=-1)
    valid_mask = dist_sq < float(cfg.agent_dist_threshold) ** 2

    cached_indices = list(policy.relevant_agent_idxs.get(focal_id, []))
    has_cached_indices = len(cached_indices) > 0
    if has_cached_indices:
        cached_array = np.asarray(cached_indices, dtype=np.int64)
        in_range_mask = (cached_array >= 0) & (cached_array < num_agents)
        cached_array = cached_array[in_range_mask]
        if cached_array.size == 0:
            selected_indices = np.empty((0,), dtype=np.int64)
        else:
            selected_indices = np.unique(cached_array[valid_mask[cached_array]])
        selected_index_set = set(int(idx) for idx in selected_indices.tolist())
        relevant_indices = [
            int(idx) for idx in cached_indices if int(idx) in selected_index_set
        ]
    else:
        candidate_count = min(max_agents, num_agents)
        if candidate_count <= 0:
            selected_indices = np.empty((0,), dtype=np.int64)
        elif candidate_count < num_agents:
            candidate_indices = np.argpartition(dist_sq, candidate_count - 1)[:candidate_count]
            selected_indices = candidate_indices[valid_mask[candidate_indices]]
        else:
            candidate_indices = np.arange(num_agents, dtype=np.int64)
            selected_indices = candidate_indices[valid_mask[candidate_indices]]
        if selected_indices.size > 1:
            selected_indices = np.sort(selected_indices)
            if dataset.split_name == "train":
                np.random.shuffle(selected_indices)
        relevant_indices = []

    if selected_indices.size > max_agents:
        selected_indices = selected_indices[:max_agents]
    origin_in_selection = bool(np.any(selected_indices == origin_agent_idx))
    return selected_indices, relevant_indices, origin_in_selection


def _build_focal_group_plans(
    adapter: Any,
    t: int,
    focal_vehicle_ids: List[int],
    ag_states: np.ndarray,
    road_points_src: np.ndarray,
    predict_rtgs: Optional[bool],
) -> tuple[List[Dict[str, Any]], List[int]]:
    """为多个 focal 构建批处理计划，而不立即复制 motion 张量。

    计划阶段只做轻量级工作：
    1. 判断 focal 是否死亡；
    2. 解析每个 focal 的相关 agent index；
    3. 按当前语义决定哪些 controlled vehicle 会并入同一个 focal batch；
    4. 记录后续 batched gather/normalize 所需的索引信息。

    Build lightweight batch plans for multiple focals without copying motion tensors yet.
    The planning stage only:
    1. Marks dead focals;
    2. Resolves relevant agent indices per focal;
    3. Groups controlled vehicles into the same focal batch using the current semantics;
    4. Records the indices required by the later batched gather/normalize stage.
    """
    policy = adapter._policy
    ordered_ids = list(focal_vehicle_ids)
    ordered_ids_arr = np.asarray(ordered_ids, dtype=np.int64)
    dead_ids: List[int] = []
    plans: List[Dict[str, Any]] = []
    has_roads = int(len(road_points_src)) > 0
    num_focals = len(ordered_ids)
    if num_focals == 0:
        return plans, dead_ids

    resolved_indices: List[np.ndarray] = []
    relevant_indices_by_row: List[List[int]] = []
    origin_in_selection = np.zeros((num_focals,), dtype=np.bool_)
    focal_alive = np.zeros((num_focals,), dtype=np.bool_)
    max_selected_agents = 0
    for row_idx, focal_id in enumerate(ordered_ids):
        origin_agent_idx = int(policy.veh_id_to_idx[focal_id])
        focal_alive[row_idx] = bool(policy.states[origin_agent_idx, t, -1]) and has_roads
        if not focal_alive[row_idx]:
            resolved_indices.append(np.zeros((0,), dtype=np.int64))
            relevant_indices_by_row.append([])
            continue
        selected_indices, relevant_indices, has_origin = _resolve_selected_agent_indices(
            adapter=adapter,
            ag_states=ag_states,
            focal_id=int(focal_id),
        )
        resolved_indices.append(selected_indices.astype(np.int64, copy=False))
        relevant_indices_by_row.append(relevant_indices)
        origin_in_selection[row_idx] = bool(has_origin)
        max_selected_agents = max(max_selected_agents, int(selected_indices.shape[0]))

    selected_agent_indices = np.zeros((num_focals, max_selected_agents), dtype=np.int64)
    selected_agent_mask = np.zeros((num_focals, max_selected_agents), dtype=np.bool_)
    for row_idx, selected_indices in enumerate(resolved_indices):
        selected_count = int(selected_indices.shape[0])
        if selected_count <= 0:
            continue
        selected_agent_indices[row_idx, :selected_count] = selected_indices
        selected_agent_mask[row_idx, :selected_count] = True

    controlled_agent_indices = np.asarray(
        [int(policy.veh_id_to_idx[focal_id]) for focal_id in ordered_ids],
        dtype=np.int64,
    )
    selected_controlled_mask = np.any(
        (selected_agent_indices[:, :, None] == controlled_agent_indices[None, None, :])
        & selected_agent_mask[:, :, None],
        axis=1,
    )
    accounted_mask = np.zeros((num_focals,), dtype=np.bool_)

    for row_idx, focal_id in enumerate(ordered_ids):
        if accounted_mask[row_idx]:
            continue
        accounted_mask[row_idx] = True

        origin_agent_idx = int(policy.veh_id_to_idx[focal_id])
        if not focal_alive[row_idx]:
            dead_ids.append(int(focal_id))
            continue

        selected_indices = resolved_indices[row_idx]
        relevant_indices = relevant_indices_by_row[row_idx]
        if not origin_in_selection[row_idx]:
            dead_ids.append(int(focal_id))
            continue

        grouped_mask = selected_controlled_mask[row_idx] & (~accounted_mask)
        grouped_mask[row_idx] = True
        accounted_mask[grouped_mask] = True
        grouped_controlled_ids = ordered_ids_arr[grouped_mask].tolist()

        relevant_ids_for_store = (
            list(selected_indices.tolist())
            if not policy.relevant_agent_idxs.get(focal_id, [])
            else relevant_indices
        )
        for grouped_id in grouped_controlled_ids:
            policy.relevant_agent_idxs[int(grouped_id)] = relevant_ids_for_store

        plans.append(
            {
                "focal_id": int(focal_id),
                "origin_agent_idx": origin_agent_idx,
                "selected_agent_indices": selected_indices.astype(np.int64, copy=False),
                "grouped_controlled_ids": grouped_controlled_ids,
                "veh_ids_in_context": [
                    int(policy.idx_to_veh_id[idx])
                    for idx in policy.relevant_agent_idxs[int(focal_id)]
                ],
                "predict_rtgs": bool(
                    policy.predict_rtgs if predict_rtgs is None else predict_rtgs
                ),
            }
        )

    return plans, dead_ids


def _select_roads_for_focals_batched(
    rl_cfg: Any,
    road_points_src: np.ndarray,
    road_types_src: np.ndarray,
    translation_xy: np.ndarray,
    angle_of_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """为多个 focal 一次性选择并归一化道路折线。

    该函数把道路选择与局部坐标系变换统一放到 batched 路径中，
    以减少重复的距离计算、排序和 Python 层循环。

    Select and normalize road polylines for multiple focals in one batched pass.
    It performs road selection and local-frame transforms in one batched path,
    reducing repeated distance computation, sorting, and Python loops.
    """
    batch_size = int(translation_xy.shape[0])
    max_roads = int(rl_cfg.max_num_road_polylines)
    final_road_points = np.zeros(
        (batch_size, max_roads, *road_points_src.shape[1:]),
        dtype=road_points_src.dtype,
    )
    final_road_types = -np.ones(
        (batch_size, max_roads, *road_types_src.shape[1:]),
        dtype=road_types_src.dtype,
    )
    if max_roads <= 0 or int(road_points_src.shape[0]) == 0:
        return final_road_points, final_road_types

    num_roads = int(road_points_src.shape[0])
    if num_roads > max_roads:
        road_valid_mask = road_points_src[:, :, -1]
        road_delta = road_points_src[None, :, :, :2] - translation_xy[:, None, None, :]
        road_dist_sq = np.sum(road_delta * road_delta, axis=-1) * road_valid_mask[None, :, :]
        selected_indices = np.argsort(np.max(road_dist_sq, axis=2), axis=1)[:, :max_roads]
        selected_road_points = road_points_src[selected_indices]
        selected_road_types = road_types_src[selected_indices]
    else:
        selected_road_points = np.broadcast_to(
            road_points_src[None, :, :, :],
            (batch_size, num_roads, *road_points_src.shape[1:]),
        ).copy()
        selected_road_types = np.broadcast_to(
            road_types_src[None, :, :],
            (batch_size, num_roads, *road_types_src.shape[1:]),
        ).copy()

    normalized_road_points = selected_road_points.copy()
    normalized_road_points[:, :, :, :2] = _apply_se2_transform_batched_np(
        coordinates=normalized_road_points[:, :, :, :2],
        translation=translation_xy[:, None, None, :],
        yaw=angle_of_rotation,
    )
    selected_count = int(selected_road_points.shape[1])
    final_road_points[:, :selected_count] = normalized_road_points[:, :max_roads]
    final_road_types[:, :selected_count] = selected_road_types[:, :max_roads]
    return final_road_points, final_road_types


def _build_flat_focal_layout_from_plans(
    adapter: Any,
    plans: List[Dict[str, Any]],
    ag_states: np.ndarray,
    ag_types: np.ndarray,
    actions_values: np.ndarray,
    rtgs_values: np.ndarray,
    goals_step: np.ndarray,
    moving_agent_mask: np.ndarray,
    road_points_src: np.ndarray,
    road_types_src: np.ndarray,
) -> Dict[str, Any]:
    """基于 focal 计划构建 flat focal layout。

    该函数是新的热点主路径：先把每个 focal 的 `selected_agent_indices` 拼成统一二维索引矩阵，
    再通过 advanced indexing 一次性 gather agent/action/RTG/goal 张量，并在 batch 维上统一完成
    heading、goal、road 的局部坐标系归一化，最后直接保留 batched motion 张量与
    flat ragged metadata，而不是重新拆回 Python `focal_batch` 列表。

    Build the flat focal layout from the focal plans.
    It stacks each focal plan's `selected_agent_indices` into one 2D index matrix, gathers
    agent/action/RTG/goal tensors via advanced indexing, performs local-frame normalization for
    headings/goals/roads across the batch dimension, and keeps the result as batched motion tensors
    plus flat ragged metadata instead of rebuilding Python `focal_batch` dicts.
    """
    if not plans:
        return {
            "focal_ids": np.zeros((0,), dtype=np.int64),
            "predict_rtgs": np.zeros((0,), dtype=np.bool_),
            "data_veh_ids_flat": np.zeros((0,), dtype=np.int64),
            "data_veh_ids_offsets": np.zeros((1,), dtype=np.int64),
            "veh_ids_in_context_flat": np.zeros((0,), dtype=np.int64),
            "veh_ids_in_context_offsets": np.zeros((1,), dtype=np.int64),
            "data_veh_model_indices_flat": np.zeros((0,), dtype=np.int64),
            "data_veh_model_indices_offsets": np.zeros((1,), dtype=np.int64),
            "context_veh_model_indices_flat": np.zeros((0,), dtype=np.int64),
            "context_veh_model_indices_offsets": np.zeros((1,), dtype=np.int64),
            "motion_data_np": {
                "agent_states": np.zeros((0, 0, 0, 0), dtype=np.float32),
                "agent_types": np.zeros((0, 0, 0), dtype=np.int64),
                "goals": np.zeros((0, 0, 0), dtype=np.float32),
                "actions": np.zeros((0, 0, 0), dtype=np.int64),
                "rtgs": np.zeros((0, 0, 0, 0), dtype=np.int64),
                "moving_agent_mask": np.zeros((0, 0), dtype=np.bool_),
                "road_points": np.zeros((0, 0, 0, 0), dtype=np.float32),
                "road_types": np.zeros((0, 0, 0), dtype=np.int64),
            },
        }

    policy = adapter._policy
    rl_cfg = policy.cfg_rl_waymo
    max_agents = int(rl_cfg.max_num_agents)
    batch_size = len(plans)
    selected_indices = np.zeros((batch_size, max_agents), dtype=np.int64)
    selected_mask = np.zeros((batch_size, max_agents), dtype=np.bool_)
    for batch_idx, plan in enumerate(plans):
        row_indices = plan["selected_agent_indices"]
        count = min(int(row_indices.shape[0]), max_agents)
        if count <= 0:
            continue
        selected_indices[batch_idx, :count] = row_indices[:count]
        selected_mask[batch_idx, :count] = True

    gathered_agent_states = ag_states[selected_indices].copy()
    gathered_agent_types = ag_types[selected_indices].copy()
    gathered_actions = actions_values[selected_indices].copy()
    gathered_rtgs = rtgs_values[selected_indices].copy()
    gathered_goals = goals_step[selected_indices].copy()
    gathered_moving_mask = moving_agent_mask[selected_indices].copy()

    invalid_mask = ~selected_mask
    if np.any(invalid_mask):
        gathered_agent_states[invalid_mask] = 0
        gathered_agent_types[invalid_mask] = -1
        gathered_actions[invalid_mask] = adapter._pad_action_values_step
        gathered_rtgs[invalid_mask] = adapter._pad_rtg_values_step
        gathered_goals[invalid_mask] = 0
        gathered_moving_mask[invalid_mask] = 0

    origin_agent_indices = np.asarray(
        [int(plan["origin_agent_idx"]) for plan in plans],
        dtype=np.int64,
    )
    origin_in_model = np.argmax(selected_indices == origin_agent_indices[:, None], axis=1)
    row_idx = np.arange(batch_size, dtype=np.int64)
    yaw = gathered_agent_states[row_idx, origin_in_model, 0, 4].astype(np.float32, copy=False)
    angle_of_rotation = (np.pi / 2.0) + np.sign(-yaw) * np.abs(yaw)
    translation_xy = gathered_agent_states[row_idx, origin_in_model, 0, :2].copy()

    gathered_agent_states[:, :, :, :2] = _apply_se2_transform_batched_np(
        coordinates=gathered_agent_states[:, :, :, :2],
        translation=translation_xy[:, None, None, :],
        yaw=angle_of_rotation,
    )
    zero_translation = np.zeros((batch_size, 1, 1, 2), dtype=translation_xy.dtype)
    gathered_agent_states[:, :, :, 2:4] = _apply_se2_transform_batched_np(
        coordinates=gathered_agent_states[:, :, :, 2:4],
        translation=zero_translation,
        yaw=angle_of_rotation,
    )
    gathered_agent_states[:, :, :, 4] = _angle_sub_batched_np(
        current_angle=gathered_agent_states[:, :, :, 4],
        target_angle=(-angle_of_rotation)[:, None, None],
    )

    gathered_goals[:, :, :2] = _apply_se2_transform_batched_np(
        coordinates=gathered_goals[:, :, :2],
        translation=translation_xy[:, None, :],
        yaw=angle_of_rotation,
    )
    if int(rl_cfg.goal_dim) == 5:
        gathered_goals[:, :, 2:4] = _apply_se2_transform_batched_np(
            coordinates=gathered_goals[:, :, 2:4],
            translation=np.zeros((batch_size, 1, 2), dtype=translation_xy.dtype),
            yaw=angle_of_rotation,
        )
        gathered_goals[:, :, 4] = _angle_sub_batched_np(
            current_angle=gathered_goals[:, :, 4],
            target_angle=(-angle_of_rotation)[:, None],
        )

    gathered_road_points, gathered_road_types = _select_roads_for_focals_batched(
        rl_cfg=rl_cfg,
        road_points_src=road_points_src,
        road_types_src=road_types_src,
        translation_xy=translation_xy,
        angle_of_rotation=angle_of_rotation,
    )

    focal_ids: List[int] = []
    predict_rtgs_list: List[bool] = []
    data_veh_rows: List[List[int]] = []
    context_veh_rows: List[List[int]] = []
    data_model_index_rows: List[List[int]] = []
    context_model_index_rows: List[List[int]] = []
    for batch_idx, plan in enumerate(plans):
        valid_indices = selected_indices[batch_idx, selected_mask[batch_idx]]
        new_agent_idx_dict = {
            int(old_idx): int(new_idx)
            for new_idx, old_idx in enumerate(valid_indices.tolist())
        }
        veh_ids_in_context = list(plan["veh_ids_in_context"])
        data_veh_ids = list(plan["grouped_controlled_ids"])
        focal_ids.append(int(plan["focal_id"]))
        predict_rtgs_list.append(bool(plan["predict_rtgs"]))
        data_veh_rows.append(data_veh_ids)
        context_veh_rows.append(veh_ids_in_context)
        data_model_index_rows.append(
            [
                int(new_agent_idx_dict.get(policy.veh_id_to_idx[int(veh_id)], -1))
                for veh_id in data_veh_ids
            ]
        )
        context_model_index_rows.append(
            [
                int(new_agent_idx_dict.get(policy.veh_id_to_idx[int(veh_id)], -1))
                for veh_id in veh_ids_in_context
            ]
        )

    data_veh_ids_flat, data_veh_ids_offsets = pack_ragged_int_lists(data_veh_rows)
    context_veh_ids_flat, context_veh_ids_offsets = pack_ragged_int_lists(context_veh_rows)
    data_model_indices_flat, data_model_indices_offsets = pack_ragged_int_lists(
        data_model_index_rows
    )
    context_model_indices_flat, context_model_indices_offsets = pack_ragged_int_lists(
        context_model_index_rows
    )
    return {
        "focal_ids": np.asarray(focal_ids, dtype=np.int64),
        "predict_rtgs": np.asarray(predict_rtgs_list, dtype=np.bool_),
        "data_veh_ids_flat": np.asarray(data_veh_ids_flat, dtype=np.int64),
        "data_veh_ids_offsets": np.asarray(data_veh_ids_offsets, dtype=np.int64),
        "veh_ids_in_context_flat": np.asarray(context_veh_ids_flat, dtype=np.int64),
        "veh_ids_in_context_offsets": np.asarray(
            context_veh_ids_offsets,
            dtype=np.int64,
        ),
        "data_veh_model_indices_flat": np.asarray(
            data_model_indices_flat,
            dtype=np.int64,
        ),
        "data_veh_model_indices_offsets": np.asarray(
            data_model_indices_offsets,
            dtype=np.int64,
        ),
        "context_veh_model_indices_flat": np.asarray(
            context_model_indices_flat,
            dtype=np.int64,
        ),
        "context_veh_model_indices_offsets": np.asarray(
            context_model_indices_offsets,
            dtype=np.int64,
        ),
        "motion_data_np": {
            "agent_states": gathered_agent_states,
            "agent_types": gathered_agent_types,
            "goals": gathered_goals,
            "actions": gathered_actions,
            "rtgs": gathered_rtgs,
            "moving_agent_mask": gathered_moving_mask,
            "road_points": gathered_road_points,
            "road_types": gathered_road_types,
        },
    }


def build_focal_batches(
    adapter: Any,
    t: int,
    focal_vehicle_ids: Optional[List[int]] = None,
    predict_rtgs: Optional[bool] = None,
) -> Tuple[Dict[str, Any], List[int]]:
    """为指定 focal 车辆构建 flat focal layout。

    Build the flat focal layout for the specified focal vehicles.
    """
    shared_context = build_step_shared_context(adapter, t)
    focal_layout, dead_ids, _ = build_focal_batches_from_shared_context(
        adapter,
        t,
        shared_context,
        focal_vehicle_ids=focal_vehicle_ids,
        predict_rtgs=predict_rtgs,
    )
    return focal_layout, dead_ids


def build_step_shared_context(
    adapter: Any,
    t: int,
) -> Dict[str, Any]:
    """构建单个 step 共享的预处理上下文。 / Build the shared per-step preprocessing context."""
    tensor_context = build_policy_step_tensor_context(adapter, t)

    return {
        "ag_states": tensor_context.policy_window_states,
        "ag_types": tensor_context.policy_window_types,
        "actions_values": tensor_context.actions_values,
        "rtgs_values": tensor_context.rtgs_values,
        "goals_step": tensor_context.goals_step,
        "moving_agent_mask": tensor_context.moving_agent_mask,
        "road_points_src": tensor_context.road_points_src,
        "road_types_src": tensor_context.road_types_src,
        "shared_timesteps": tensor_context.shared_timesteps,
        "tensor_context": tensor_context,
    }


def build_focal_batches_from_shared_context(
    adapter: Any,
    t: int,
    shared_context: Dict[str, Any],
    focal_vehicle_ids: Optional[List[int]] = None,
    predict_rtgs: Optional[bool] = None,
) -> Tuple[Dict[str, Any], List[int], np.ndarray]:
    """基于共享上下文构建 flat focal layout。

    Build the flat focal layout from a shared step context.
    """
    ag_states = shared_context["ag_states"]
    ag_types = shared_context["ag_types"]
    actions_values = shared_context["actions_values"]
    rtgs_values = shared_context["rtgs_values"]
    goals_step = shared_context["goals_step"]
    moving_agent_mask = shared_context["moving_agent_mask"]
    road_points_src = shared_context["road_points_src"]
    road_types_src = shared_context["road_types_src"]
    shared_timesteps = shared_context["shared_timesteps"]

    if focal_vehicle_ids is None:
        ordered_focal_vehicle_ids = get_control_vehicle_queue(adapter)
    else:
        ordered_focal_vehicle_ids = list(focal_vehicle_ids)

    plans, dead_ids = _build_focal_group_plans(
        adapter=adapter,
        t=t,
        focal_vehicle_ids=ordered_focal_vehicle_ids,
        ag_states=ag_states,
        road_points_src=road_points_src,
        predict_rtgs=predict_rtgs,
    )
    focal_layout = _build_flat_focal_layout_from_plans(
        adapter=adapter,
        plans=plans,
        ag_states=ag_states,
        ag_types=ag_types,
        actions_values=actions_values,
        rtgs_values=rtgs_values,
        goals_step=goals_step,
        moving_agent_mask=moving_agent_mask,
        road_points_src=road_points_src,
        road_types_src=road_types_src,
    )

    return focal_layout, dead_ids, shared_timesteps
