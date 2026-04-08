"""
统一管理 adapter 当前 step 的张量化上下文。
该模块把原先分散在 update、reward、policy_sync、focal_input 中的中间数组收敛到同一份连续内存表示，
用于减少重复扫描、重复拷贝和反复的 Python 容器转换。
Manage the adapter's tensorized per-step runtime context in one place.
This module centralizes the intermediate arrays previously rebuilt independently by update, reward,
policy_sync, and focal_input so the pipeline can reuse one contiguous representation per step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List

import numpy as np


@dataclass(slots=True)
class StepTensorContext:
    """保存 adapter 当前 step 可复用张量视图的统一上下文。

    该数据结构的目标是让 adapter 在一次 step 内共享同一份数组化中间结果：
    运行时状态更新、reward 计算、policy 写回、focal batch 构造都只读取这份上下文，
    不再各自从 `vehicle_data_dict` 或 policy 缓存中重复扫描和重建 Python 列表。

    This structure stores the reusable tensor views for the adapter's current step.
    Its purpose is to let runtime update, reward computation, policy write-back, and focal-batch
    construction operate on the same per-step arrays instead of repeatedly rebuilding Python lists
    from `vehicle_data_dict` or policy buffers.
    """

    step_t: int  # Simulation step index represented by this context.
    controlled_vehicle_ids: List[int]  # Controlled vehicle ids active at this step.
    update_vehicle_ids: List[int]  # Vehicle ids updated in this step.
    policy_vehicle_indices: np.ndarray | None = None  # Policy indices for `update_vehicle_ids`.
    positions_xy: np.ndarray | None = None  # Updated-vehicle xy positions with shape `[U, 2]`.
    velocities_xy: np.ndarray | None = None  # Updated-vehicle xy velocities with shape `[U, 2]`.
    headings: np.ndarray | None = None  # Updated-vehicle headings.
    existence: np.ndarray | None = None  # Updated-vehicle existence flags.
    timesteps: np.ndarray | None = None  # Updated-vehicle timestep values.
    goals_state: np.ndarray | None = None  # Updated-vehicle goal-state vectors.
    latest_rewards: np.ndarray | None = None  # Updated-vehicle latest reward vectors.
    latest_dense_rewards: np.ndarray | None = None  # Updated-vehicle dense reward vectors.
    latest_nearest_dist: np.ndarray | None = None  # Updated-vehicle nearest distances.
    latest_rtgs: np.ndarray | None = None  # Updated-vehicle RTG vectors.
    latest_actions: np.ndarray | None = None  # Updated-vehicle latest actions with shape `[U, 2]`.
    context_vehicle_ids: List[int] | None = None  # Vehicle ids participating in reward context.
    context_positions_xy: np.ndarray | None = None  # Reward-context xy positions.
    context_existence: np.ndarray | None = None  # Reward-context existence flags.
    target_vehicle_ids: List[int] | None = None  # Target vehicle ids for reward computation.
    target_context_indices: np.ndarray | None = None  # Context indices of reward targets.
    target_update_indices: np.ndarray | None = None  # Update-row indices of reward targets.
    policy_window_states: np.ndarray | None = None  # Policy-window state slice for this step.
    policy_window_types: np.ndarray | None = None  # Policy type array for this step.
    actions_values: np.ndarray | None = None  # Discretized action window for this step.
    rtgs_values: np.ndarray | None = None  # RTG window for this step.
    goals_step: np.ndarray | None = None  # Goal vectors used by focal-batch preparation.
    moving_agent_mask: np.ndarray | None = None  # Moving-agent mask for focal preparation.
    road_points_src: np.ndarray | None = None  # Road polyline points for focal preparation.
    road_types_src: np.ndarray | None = None  # Road type array for focal preparation.
    shared_timesteps: np.ndarray | None = None  # Shared timestep template for focal preparation.


@dataclass(slots=True)
class PolicyWindowCache:
    """保存 policy 窗口离散化缓存。 / Store the discretized policy-window cache.

    该结构把 focal/input 路径需要复用的 action/RTG/timestep 窗口与 pad 值显式组织起来，
    代替基于字符串键的松散字典访问，使增量更新路径更直观也更不容易出错。

    This structure keeps the reusable action/RTG/timestep windows and pad values used by
    the focal/input path in one explicit place. It replaces loose string-keyed dictionaries
    so the incremental-update path is easier to read and harder to misuse.
    """

    actions_values: np.ndarray
    rtgs_values: np.ndarray
    shared_timesteps: np.ndarray
    pad_action_values: np.ndarray
    pad_rtg_values: np.ndarray
    action_dim: int
    rtg_discretization: int
    last_t: int = -1
    window_start: int = -1
    window_end: int = -1


def _copy_context_field(
    current: StepTensorContext,
    incoming: StepTensorContext,
    field_name: str,
) -> StepTensorContext:
    value = getattr(incoming, field_name)
    if value is None:
        return current
    return replace(current, **{field_name: value})


def merge_step_tensor_context(
    adapter: Any,
    context: StepTensorContext,
) -> StepTensorContext:
    """将新生成的 step 上下文合并回 adapter，并尽量复用已有字段。

    若 adapter 上已经存在同一 `step_t` 的上下文，则本函数会保留旧上下文中未被新值覆盖的字段，
    以便 update/reward/focal_input 在不同阶段逐步填充同一份数据结构。

    Merge a newly generated step context back into the adapter while preserving
    already-computed fields for the same `step_t`.
    This lets update/reward/focal_input fill one shared structure incrementally.
    """
    existing = getattr(adapter, "_step_tensor_context", None)
    if existing is None or int(existing.step_t) != int(context.step_t):
        adapter._step_tensor_context = context
        return context

    merged = existing
    for field_name in StepTensorContext.__dataclass_fields__.keys():
        merged = _copy_context_field(merged, context, field_name)
    adapter._step_tensor_context = merged
    return merged


def _get_controlled_vehicle_ids(adapter: Any) -> List[int]:
    step_ids = list(getattr(adapter, "_controlled_vehicle_ids_step", []))
    if step_ids:
        return step_ids
    sorted_ids = list(getattr(adapter, "_vehicles_to_control_sorted", []))
    if sorted_ids:
        return sorted_ids
    return list(getattr(adapter, "_vehicles_to_control", []))


def _get_or_create_prepare_buffer(
    adapter: Any,
    name: str,
    shape: tuple[int, ...],
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


def _slice_policy_window(
    policy: Any,
    t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, end = _resolve_policy_window_bounds(policy, t)
    return (
        policy.states[:, start:end],
        policy.types,
        policy.actions[:, start:end],
        policy.rtgs[:, start:end],
        policy.goals[:, start:end],
        policy.timesteps[0, start:end],
    )


def _resolve_policy_window_bounds(policy: Any, t: int) -> tuple[int, int]:
    """计算当前 step 的 policy 窗口边界。 / Compute the policy-window bounds for the current step.

    在 warmup 阶段，窗口固定锚定在序列开头；进入稳定阶段后，窗口会随 step 向前滑动。
    这个边界函数被离散化缓存与原始 view 切片共同复用，确保两侧始终对齐。

    During warmup the window stays anchored at the beginning of the sequence; once the
    rollout passes warmup, the window slides forward with each step. This helper is shared
    by both the discretization cache and the raw view slicing path so they stay aligned.
    """
    train_context_length = int(policy.cfg_rl_waymo.train_context_length)
    start = 0 if t < train_context_length else t - (train_context_length - 1)
    end = train_context_length if t < train_context_length else t + 1
    return start, end


def _normalize_rtgs_inplace(rtgs: np.ndarray, rl_cfg: Any) -> None:
    rtgs[:, :, 0] = (
        np.clip(rtgs[:, :, 0], rl_cfg.min_rtg_pos, rl_cfg.max_rtg_pos)
        - rl_cfg.min_rtg_pos
    ) / (rl_cfg.max_rtg_pos - rl_cfg.min_rtg_pos)
    rtgs[:, :, 1] = (
        np.clip(rtgs[:, :, 1], rl_cfg.min_rtg_veh, rl_cfg.max_rtg_veh)
        - rl_cfg.min_rtg_veh
    ) / (rl_cfg.max_rtg_veh - rl_cfg.min_rtg_veh)
    rtgs[:, :, 2] = (
        np.clip(rtgs[:, :, 2], rl_cfg.min_rtg_road, rl_cfg.max_rtg_road)
        - rl_cfg.min_rtg_road
    ) / (rl_cfg.max_rtg_road - rl_cfg.min_rtg_road)


def _sanitize_index_array(
    values: np.ndarray,
    *,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    sanitized = np.nan_to_num(
        np.asarray(values),
        nan=float(min_value),
        posinf=float(max_value),
        neginf=float(min_value),
    )
    return np.clip(sanitized, min_value, max_value).astype(np.int64, copy=False)


def _get_or_create_policy_window_cache(
    adapter: Any,
    *,
    policy: Any,
    actions_src: np.ndarray,
    rtgs_src: np.ndarray,
    timesteps_src: np.ndarray,
) -> PolicyWindowCache:
    """获取当前 step 可复用的 policy 窗口缓存。 / Get the reusable policy-window cache for the current step.

    该缓存保存离散化后的 action/RTG 窗口、共享 timestep 模板以及对应的 pad 值。
    对于顺序推进的 step，它允许我们只更新最新一列，而不是每次都对整窗重新离散化。

    This cache stores the discretized action/RTG windows, the shared timestep template,
    and the pad values derived from the current policy configuration. For sequential steps
    it lets us update only the newest column instead of re-discretizing the whole window.
    """
    cache = getattr(adapter, "_policy_window_prepare_cache", None)
    action_dim = int(policy.action_dim)
    rtg_dim = int(getattr(policy.cfg_rl_waymo, "rtg_discretization", 0))
    expected_action_prefix = tuple(actions_src.shape[:2])
    expected_rtg_prefix = tuple(rtgs_src.shape[:2])
    expected_shapes = {
        "shared_timesteps": (
            int(policy.cfg_rl_waymo.max_num_agents),
            *timesteps_src.shape,
        ),
    }
    if cache is not None:
        valid_shapes = (
            tuple(cache.actions_values.shape[:2]) == expected_action_prefix
            and tuple(cache.rtgs_values.shape[:2]) == expected_rtg_prefix
            and tuple(cache.shared_timesteps.shape) == expected_shapes["shared_timesteps"]
            and int(cache.pad_action_values.shape[0]) == int(actions_src.shape[1])
            and int(cache.pad_rtg_values.shape[0]) == int(rtgs_src.shape[1])
        )
        valid_ranges = (
            int(cache.action_dim) == action_dim
            and int(cache.rtg_discretization) == rtg_dim
        )
        if valid_shapes and valid_ranges:
            return cache

    rtg_cache_dtype = np.dtype(np.int64) if bool(policy.discretize_rtgs) else np.dtype(rtgs_src.dtype)
    cache = PolicyWindowCache(
        actions_values=np.empty(actions_src.shape, dtype=np.int64),
        rtgs_values=np.empty(rtgs_src.shape, dtype=rtg_cache_dtype),
        shared_timesteps=np.empty(
            (int(policy.cfg_rl_waymo.max_num_agents), *timesteps_src.shape),
            dtype=np.int64,
        ),
        pad_action_values=np.empty(actions_src.shape[1:], dtype=np.int64),
        pad_rtg_values=np.empty(rtgs_src.shape[1:], dtype=rtg_cache_dtype),
        action_dim=action_dim,
        rtg_discretization=rtg_dim,
    )
    adapter._policy_window_prepare_cache = cache
    return cache


def _rebuild_policy_window_cache_full(
    adapter: Any,
    *,
    policy: Any,
    cache: PolicyWindowCache,
    actions_src: np.ndarray,
    rtgs_src: np.ndarray,
    timesteps_src: np.ndarray,
) -> None:
    """全量重建 policy 窗口缓存。 / Rebuild the full policy-window cache from scratch.

    当 step 不连续、窗口形状变化、或缓存首次初始化时，必须回到全量路径，确保缓存与
    当前 policy 历史完全一致。该函数也负责同步当前 step 的 pad 值。

    When steps are non-sequential, the window layout changes, or the cache is being
    initialized for the first time, the logic falls back to a full rebuild so the cache
    stays exactly aligned with the current policy history. This function also refreshes
    the pad values for the current step.
    """
    actions_buffer = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="actions_buffer",
        shape=(actions_src.shape[0] + 1, *actions_src.shape[1:]),
        dtype=np.dtype(actions_src.dtype),
    )
    actions_buffer.fill(0)
    actions_buffer[: actions_src.shape[0]] = actions_src
    actions_discrete = adapter.dataset.discretize_actions(actions_buffer)
    sanitized_actions = _sanitize_index_array(
        actions_discrete[: actions_src.shape[0]],
        min_value=0,
        max_value=int(cache.action_dim) - 1,
    )
    sanitized_pad_action = _sanitize_index_array(
        np.asarray(actions_discrete[actions_src.shape[0]]).copy(),
        min_value=0,
        max_value=int(cache.action_dim) - 1,
    )
    if tuple(cache.actions_values.shape) != tuple(sanitized_actions.shape):
        cache.actions_values = np.empty(sanitized_actions.shape, dtype=np.int64)
    if tuple(cache.pad_action_values.shape) != tuple(sanitized_pad_action.shape):
        cache.pad_action_values = np.empty(sanitized_pad_action.shape, dtype=np.int64)
    cache.actions_values[...] = sanitized_actions
    cache.pad_action_values[...] = sanitized_pad_action

    rtgs_norm = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="rtgs_norm_buffer",
        shape=rtgs_src.shape,
        dtype=np.dtype(rtgs_src.dtype),
    )
    np.copyto(rtgs_norm, rtgs_src)
    _normalize_rtgs_inplace(rtgs_norm, policy.cfg_rl_waymo)
    if bool(policy.discretize_rtgs):
        rtgs_discrete = _get_or_create_prepare_buffer(
            adapter=adapter,
            name="rtgs_discrete_buffer",
            shape=(rtgs_norm.shape[0] + 1, *rtgs_norm.shape[1:]),
            dtype=np.dtype(rtgs_norm.dtype),
        )
        rtgs_discrete.fill(0)
        rtgs_discrete[: rtgs_norm.shape[0]] = rtgs_norm
        rtgs_discrete_values = adapter.dataset.discretize_rtgs(rtgs_discrete)
        cache.rtgs_values[...] = _sanitize_index_array(
            rtgs_discrete_values[: rtgs_norm.shape[0]],
            min_value=0,
            max_value=int(cache.rtg_discretization) - 1,
        )
        cache.pad_rtg_values[...] = _sanitize_index_array(
            np.asarray(rtgs_discrete_values[rtgs_norm.shape[0]]).copy(),
            min_value=0,
            max_value=int(cache.rtg_discretization) - 1,
        )
    else:
        cache.rtgs_values[...] = rtgs_norm
        cache.pad_rtg_values[...] = 0

    cache.shared_timesteps[...] = np.asarray(
        timesteps_src,
        dtype=np.int64,
    )[np.newaxis, ...]


def _update_policy_window_cache_incrementally(
    adapter: Any,
    *,
    policy: Any,
    cache: PolicyWindowCache,
    t: int,
    window_start: int,
    window_end: int,
) -> bool:
    """按最新一列增量更新 policy 窗口缓存。 / Incrementally update the policy-window cache with the newest column.

    只有在当前 step 与上一个 step 连续、且窗口推进方式符合 warmup/滑窗语义时，才走增量路径。
    warmup 阶段只覆盖当前列；进入滑窗阶段后，缓存先整体左移，再写入最后一列。

    The incremental path is only valid when the current step directly follows the previous
    one and the window progression matches the warmup/sliding semantics. During warmup it
    only overwrites the active column; once the window starts sliding, it shifts the cache
    left and writes the newest column at the tail.
    """
    last_t = int(cache.last_t)
    last_start = int(cache.window_start)
    last_end = int(cache.window_end)
    if last_t != int(t) - 1:
        return False

    current_width = int(window_end - window_start)
    if current_width != int(cache.actions_values.shape[1]):
        return False

    if window_start == last_start and window_end == last_end:
        update_col = int(t)
        if update_col < 0 or update_col >= current_width:
            return False
    elif window_start == last_start + 1 and window_end == last_end + 1:
        cache.actions_values[:, :-1] = cache.actions_values[:, 1:]
        cache.rtgs_values[:, :-1] = cache.rtgs_values[:, 1:]
        cache.shared_timesteps[:, :-1] = cache.shared_timesteps[:, 1:]
        update_col = current_width - 1
    else:
        return False

    action_tail = policy.actions[:, t : t + 1].copy()
    action_tail_buffer = _get_or_create_prepare_buffer(
        adapter=adapter,
        name="actions_tail_buffer",
        shape=(action_tail.shape[0] + 1, *action_tail.shape[1:]),
        dtype=np.dtype(action_tail.dtype),
    )
    action_tail_buffer.fill(0)
    action_tail_buffer[: action_tail.shape[0]] = action_tail
    action_tail_discrete = adapter.dataset.discretize_actions(action_tail_buffer)
    sanitized_action_tail = _sanitize_index_array(
        action_tail_discrete[: action_tail.shape[0]],
        min_value=0,
        max_value=int(cache.action_dim) - 1,
    )
    if cache.actions_values.ndim == 2:
        cache.actions_values[:, update_col] = sanitized_action_tail.reshape((-1,))
    else:
        cache.actions_values[:, update_col] = sanitized_action_tail[:, 0]

    rtg_tail = policy.rtgs[:, t : t + 1].copy()
    _normalize_rtgs_inplace(rtg_tail, policy.cfg_rl_waymo)
    if bool(policy.discretize_rtgs):
        rtg_tail_buffer = _get_or_create_prepare_buffer(
            adapter=adapter,
            name="rtgs_tail_buffer",
            shape=(rtg_tail.shape[0] + 1, *rtg_tail.shape[1:]),
            dtype=np.dtype(rtg_tail.dtype),
        )
        rtg_tail_buffer.fill(0)
        rtg_tail_buffer[: rtg_tail.shape[0]] = rtg_tail
        rtg_tail_discrete = adapter.dataset.discretize_rtgs(rtg_tail_buffer)
        cache.rtgs_values[:, update_col] = _sanitize_index_array(
            rtg_tail_discrete[: rtg_tail.shape[0]],
            min_value=0,
            max_value=int(cache.rtg_discretization) - 1,
        )[:, 0]
    else:
        cache.rtgs_values[:, update_col] = rtg_tail[:, 0]

    cache.shared_timesteps[:, update_col] = np.asarray(
        policy.timesteps[0, t : t + 1],
        dtype=np.int64,
    )
    return True


def build_policy_step_tensor_context(
    adapter: Any,
    t: int,
) -> StepTensorContext:
    """构建 focal/input 路径使用的 policy 级 step 张量上下文。

    该函数负责从 ctrl-sim policy 的历史窗口中提取当前 step 所需的连续数组，
    并完成离散动作、离散 RTG、共享 timestep 模板、道路数组等复用数据的准备。
    产物会被 focal batch 构造直接消费，也会回写到 adapter 上供同一步的其他模块复用。

    Build the policy-side tensor context used by focal-input preparation.
    The function extracts the current policy window, prepares discretized actions/RTGs,
    shared timestep templates, and road arrays, then stores the reusable result on the adapter.
    """
    policy = adapter._policy
    moving_agent_mask = getattr(adapter, "_moving_agent_mask_cache", None)
    if moving_agent_mask is None:
        moving_ids = np.where(
            np.linalg.norm(policy.states[:, 0, :2] - policy.goals[:, 0, :2], axis=1)
            > float(policy.cfg_rl_waymo.moving_threshold)
        )[0]
        moving_agent_mask = np.isin(np.arange(policy.states.shape[0]), moving_ids)
        adapter._moving_agent_mask_cache = moving_agent_mask

    ag_states, ag_types, actions_src, rtgs_src, goals, timesteps_src = (
        _slice_policy_window(policy, t)
    )

    window_start, window_end = _resolve_policy_window_bounds(policy, t)
    policy_window_cache = _get_or_create_policy_window_cache(
        adapter,
        policy=policy,
        actions_src=actions_src,
        rtgs_src=rtgs_src,
        timesteps_src=timesteps_src,
    )
    cache_updated = _update_policy_window_cache_incrementally(
        adapter,
        policy=policy,
        cache=policy_window_cache,
        t=t,
        window_start=window_start,
        window_end=window_end,
    )
    if not cache_updated:
        _rebuild_policy_window_cache_full(
            adapter,
            policy=policy,
            cache=policy_window_cache,
            actions_src=actions_src,
            rtgs_src=rtgs_src,
            timesteps_src=timesteps_src,
        )
    policy_window_cache.last_t = int(t)
    policy_window_cache.window_start = int(window_start)
    policy_window_cache.window_end = int(window_end)
    actions_values = policy_window_cache.actions_values
    rtgs_values = policy_window_cache.rtgs_values
    shared_timesteps = policy_window_cache.shared_timesteps
    adapter._pad_action_values_step = policy_window_cache.pad_action_values
    adapter._pad_rtg_values_step = policy_window_cache.pad_rtg_values

    road_points_src = adapter._preproc_data.get("road_points")
    road_types_src = adapter._preproc_data.get("road_types")
    if road_points_src is None:
        road_points_src = np.zeros((0, 1, 3), dtype=np.float32)
    if road_types_src is None:
        road_types_src = np.zeros((0, 1), dtype=np.float32)

    context = StepTensorContext(
        step_t=int(t),
        controlled_vehicle_ids=_get_controlled_vehicle_ids(adapter),
        update_vehicle_ids=list(getattr(adapter, "_state_update_vehicle_ids_step", [])),
        policy_window_states=ag_states,
        policy_window_types=ag_types,
        actions_values=actions_values,
        rtgs_values=rtgs_values,
        goals_step=goals[:, 0],
        moving_agent_mask=np.asarray(moving_agent_mask),
        road_points_src=np.asarray(road_points_src),
        road_types_src=np.asarray(road_types_src),
        shared_timesteps=shared_timesteps,
    )
    return merge_step_tensor_context(adapter, context)


def _get_step_or_last(entries: List[Any], t: int) -> Any:
    if not entries:
        raise IndexError("Cannot read from empty history.")
    if t < len(entries):
        return entries[t]
    return entries[-1]


def _stack_xy(vehicle_data_list: List[Dict[str, Any]], t: int, key: str) -> np.ndarray:
    return np.asarray(
        [
            [
                float(_get_step_or_last(vehicle_data[key], t)["x"]),
                float(_get_step_or_last(vehicle_data[key], t)["y"]),
            ]
            for vehicle_data in vehicle_data_list
        ],
        dtype=np.float32,
    )


def build_runtime_step_tensor_context(
    adapter: Any,
    t: int,
    vehicle_data_dict: Dict[int, Dict[str, Any]],
) -> StepTensorContext:
    """构建 update/reward/policy_sync 共享的运行时 step 张量上下文。

    该函数从 `vehicle_data_dict` 中提取当前 step 需要写回 policy 和计算 reward 的最小连续数组，
    包括位置、速度、朝向、存在性、目标状态、最近一次 reward/RTG/动作，以及 reward 上下文车辆索引。
    构建出的上下文会缓存到 adapter 上，供 reward 与 policy_sync 直接复用。

    Build the runtime tensor context shared by update, reward, and policy_sync.
    It extracts the minimal contiguous arrays needed for policy write-back and reward computation
    from `vehicle_data_dict`, then caches the result on the adapter for reuse in the same step.
    """
    update_vehicle_ids = list(getattr(adapter, "_state_update_vehicle_ids_step", []))
    controlled_vehicle_ids = list(getattr(adapter, "_controlled_vehicle_ids_step", []))
    policy = getattr(adapter, "_policy", None)
    goal_dim = int(getattr(getattr(policy, "cfg_rl_waymo", None), "goal_dim", 2))
    use_rtg = bool(getattr(policy, "use_rtg", False)) if policy is not None else False
    context_vehicle_ids = [
        veh_id
        for veh_id in getattr(
            adapter, "_all_vehicle_ids", list(vehicle_data_dict.keys())
        )
        if (
            veh_id in vehicle_data_dict
            and vehicle_data_dict[veh_id].get("position")
            and vehicle_data_dict[veh_id].get("existence")
        )
    ]
    target_vehicle_ids = [
        veh_id for veh_id in update_vehicle_ids if veh_id in context_vehicle_ids
    ]

    policy_vehicle_indices = None
    if policy is not None:
        policy_vehicle_indices = np.asarray(
            [int(policy.veh_id_to_idx[veh_id]) for veh_id in update_vehicle_ids],
            dtype=np.int64,
        )

    vehicle_data_list = [vehicle_data_dict[veh_id] for veh_id in update_vehicle_ids]
    positions_xy = (
        _stack_xy(vehicle_data_list, t, "position")
        if vehicle_data_list
        else np.zeros((0, 2), dtype=np.float32)
    )
    velocities_xy = (
        _stack_xy(vehicle_data_list, t, "velocity")
        if vehicle_data_list
        else np.zeros((0, 2), dtype=np.float32)
    )
    headings = (
        np.asarray(
            [
                float(_get_step_or_last(vehicle_data["heading"], t))
                for vehicle_data in vehicle_data_list
            ],
            dtype=np.float32,
        )
        if vehicle_data_list
        else np.zeros((0,), dtype=np.float32)
    )
    existence = (
        np.asarray(
            [
                float(_get_step_or_last(vehicle_data["existence"], t))
                for vehicle_data in vehicle_data_list
            ],
            dtype=np.float32,
        )
        if vehicle_data_list
        else np.zeros((0,), dtype=np.float32)
    )
    timesteps = (
        np.asarray(
            [
                int(_get_step_or_last(vehicle_data["timestep"], t))
                for vehicle_data in vehicle_data_list
            ],
            dtype=np.int64,
        )
        if vehicle_data_list
        else np.zeros((0,), dtype=np.int64)
    )

    goals_state = np.zeros((len(vehicle_data_list), goal_dim), dtype=np.float32)
    latest_rewards_list: List[np.ndarray] = []
    latest_dense_rewards_list: List[np.ndarray] = []
    latest_rtgs_list: List[np.ndarray] = []
    latest_nearest_dist = np.zeros((len(vehicle_data_list),), dtype=np.float32)
    latest_actions = np.zeros((len(vehicle_data_list), 2), dtype=np.float32)
    for index, vehicle_data in enumerate(vehicle_data_list):
        goals_state[index, 0] = float(vehicle_data["goal_position"]["x"])
        goals_state[index, 1] = float(vehicle_data["goal_position"]["y"])
        if goal_dim > 2:
            goals_state[index, 2] = float(vehicle_data["goal_velocity_x"])
        if goal_dim > 3:
            goals_state[index, 3] = float(vehicle_data["goal_velocity_y"])
        if goal_dim > 4:
            goals_state[index, 4] = float(vehicle_data["goal_heading"])

        reward_value = (
            vehicle_data["reward"][-1]
            if vehicle_data.get("reward")
            else np.zeros((0,), dtype=np.float32)
        )
        latest_rewards_list.append(np.asarray(reward_value, dtype=np.float32))

        dense_reward_value = (
            vehicle_data["dense_reward"][-1]
            if vehicle_data.get("dense_reward")
            else np.zeros((0,), dtype=np.float32)
        )
        latest_dense_rewards_list.append(
            np.asarray(dense_reward_value, dtype=np.float32)
        )

        if use_rtg and vehicle_data.get("rtgs"):
            latest_rtgs_list.append(
                np.asarray(vehicle_data["rtgs"][-1], dtype=np.float32)
            )
        elif vehicle_data.get("rtgs"):
            latest_rtgs_list.append(
                np.asarray(vehicle_data["rtgs"][-1], dtype=np.float32)
            )

        if vehicle_data.get("nearest_dist"):
            latest_nearest_dist[index] = float(vehicle_data["nearest_dist"][-1])
        if vehicle_data.get("acceleration"):
            latest_actions[index, 0] = float(vehicle_data["acceleration"][-1])
        if vehicle_data.get("steering"):
            latest_actions[index, 1] = float(vehicle_data["steering"][-1])

    latest_rewards = (
        np.stack(latest_rewards_list, axis=0)
        if latest_rewards_list and latest_rewards_list[0].size > 0
        else np.zeros((len(vehicle_data_list), 0), dtype=np.float32)
    )
    latest_dense_rewards = (
        np.stack(latest_dense_rewards_list, axis=0)
        if latest_dense_rewards_list and latest_dense_rewards_list[0].size > 0
        else np.zeros((len(vehicle_data_list), 0), dtype=np.float32)
    )
    latest_rtgs = (
        np.stack(latest_rtgs_list, axis=0)
        if latest_rtgs_list and latest_rtgs_list[0].size > 0
        else np.zeros((len(vehicle_data_list), 0), dtype=np.float32)
    )

    context_vehicle_data_list = [
        vehicle_data_dict[veh_id] for veh_id in context_vehicle_ids
    ]
    context_positions_xy = (
        _stack_xy(context_vehicle_data_list, t, "position")
        if context_vehicle_data_list
        else np.zeros((0, 2), dtype=np.float32)
    )
    context_existence = (
        np.asarray(
            [
                float(_get_step_or_last(vehicle_data["existence"], t))
                for vehicle_data in context_vehicle_data_list
            ],
            dtype=np.float32,
        )
        if context_vehicle_data_list
        else np.zeros((0,), dtype=np.float32)
    )
    context_idx_map = {veh_id: idx for idx, veh_id in enumerate(context_vehicle_ids)}
    target_context_indices = (
        np.asarray(
            [int(context_idx_map[veh_id]) for veh_id in target_vehicle_ids],
            dtype=np.int64,
        )
        if target_vehicle_ids
        else np.zeros((0,), dtype=np.int64)
    )
    update_idx_map = {veh_id: idx for idx, veh_id in enumerate(update_vehicle_ids)}
    target_update_indices = (
        np.asarray(
            [int(update_idx_map[veh_id]) for veh_id in target_vehicle_ids],
            dtype=np.int64,
        )
        if target_vehicle_ids
        else np.zeros((0,), dtype=np.int64)
    )

    context = StepTensorContext(
        step_t=int(t),
        controlled_vehicle_ids=controlled_vehicle_ids,
        update_vehicle_ids=update_vehicle_ids,
        policy_vehicle_indices=policy_vehicle_indices,
        positions_xy=positions_xy,
        velocities_xy=velocities_xy,
        headings=headings,
        existence=existence,
        timesteps=timesteps,
        goals_state=goals_state,
        latest_rewards=latest_rewards,
        latest_dense_rewards=latest_dense_rewards,
        latest_nearest_dist=latest_nearest_dist,
        latest_rtgs=latest_rtgs,
        latest_actions=latest_actions,
        context_vehicle_ids=context_vehicle_ids,
        context_positions_xy=context_positions_xy,
        context_existence=context_existence,
        target_vehicle_ids=target_vehicle_ids,
        target_context_indices=target_context_indices,
        target_update_indices=target_update_indices,
    )
    return merge_step_tensor_context(adapter, context)
