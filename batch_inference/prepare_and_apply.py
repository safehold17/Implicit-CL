"""
Batch inference 模式下 adapter 侧的数据准备与结果应用。

- prepare_step(): 构建 prepared_dict 供 ExternalTeacher 使用
- build_focal_batches(): 执行 get_data 逻辑但在 from_numpy() 之前停下
- apply_predictions(): 接收推理结果，写回 vehicle_data_dict 并返回动作
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

from .ipc_codec import pack_prepared, unpack_model_outputs


def prepare_step(adapter, t: int, vehicles: List) -> Optional[Dict]:
    """构建 prepared_dict 供主进程 ExternalTeacher.batched_forward() 使用。

    在 get_data() 的 from_numpy() 之前停下，返回原始 numpy dict，
    避免子进程 tensor → numpy → 主进程 tensor 的双重转换。

    Args:
        adapter: CtrlSimOpponentAdapter 实例
        t: 当前时间步（current_step 递增前）
        vehicles: 场景中的所有车辆列表

    Returns:
        prepared_dict | None
    """
    if adapter._policy is None or len(vehicles) == 0:
        return None

    adapter._last_vehicles = vehicles
    if t < adapter.history_steps - 1:
        adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}
    else:
        adapter._last_vehicle_by_id = {}

    # 1. 更新 vehicle_data_dict
    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(
        t, vehicles, adapter._vehicle_data_dict
    )

    # 2. 更新策略内部状态
    adapter._policy.update_state(
        adapter._vehicle_data_dict,
        adapter._vehicles_to_control,
        t,
    )

    # 3. 构建 focal batches（numpy 级别）
    focal_batches, dead_ids = build_focal_batches(adapter, t)

    if not focal_batches and not dead_ids:
        return pack_prepared({'status': 'skip', 'step_t': t, 'dead_ids': []})

    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1

    # 构建 tilt_by_veh_id
    tilt_by_veh_id = {}
    if adapter.per_vehicle_tilting:
        tilt_by_veh_id = dict(adapter.per_vehicle_tilting)

    prepared_dict = {
        'status': 'ok',
        'step_t': t,
        'token_index': token_index,
        'dead_ids': dead_ids,
        'sampling': {
            'action_temperature': adapter.action_temperature,
            'nucleus_sampling': adapter.nucleus_sampling,
            'nucleus_threshold': adapter.nucleus_threshold,
        },
        'default_tilt': (
            adapter.current_tilt.goal_tilt,
            adapter.current_tilt.veh_veh_tilt,
            adapter.current_tilt.veh_edge_tilt,
        ),
        'tilt_by_veh_id': tilt_by_veh_id,
        'veh_id_to_idx': dict(adapter._policy.veh_id_to_idx),
        'focal_batches': focal_batches,
    }
    return pack_prepared(prepared_dict)


def build_focal_batches(adapter, t: int):
    """执行 get_data() 的逻辑，但在 from_numpy() 之前停下。

    返回 (focal_batches, dead_ids)
    """
    policy = adapter._policy
    dset = adapter.dataset

    # — 与 AutoregressivePolicy.get_data() 完全对齐 —
    moving_ids = np.where(
        np.linalg.norm(
            policy.states[:, 0, :2] - policy.goals[:, 0, :2], axis=1
        ) > policy.cfg_rl_waymo.moving_threshold
    )[0]
    moving_agent_mask = np.isin(np.arange(policy.states.shape[0]), moving_ids)

    tcl = policy.cfg_rl_waymo.train_context_length
    if t < tcl:
        ag_states = policy.states[:, :tcl].copy()
        ag_types = policy.types.copy()
        actions = policy.actions[:, :tcl].copy()
        rtgs = policy.rtgs[:, :tcl].copy()
        goals = policy.goals[:, :tcl].copy()
        timesteps = policy.timesteps[0, :tcl].astype(int).copy()
    else:
        ag_states = policy.states[:, t - (tcl - 1):t + 1].copy()
        ag_types = policy.types.copy()
        actions = policy.actions[:, t - (tcl - 1):t + 1].copy()
        rtgs = policy.rtgs[:, t - (tcl - 1):t + 1].copy()
        goals = policy.goals[:, t - (tcl - 1):t + 1].copy()
        timesteps = policy.timesteps[0, t - (tcl - 1):t + 1].astype(int).copy()

    normalize_timestep = 0
    rl = policy.cfg_rl_waymo
    rtgs[:, :, 0] = (np.clip(rtgs[:, :, 0], rl.min_rtg_pos, rl.max_rtg_pos) - rl.min_rtg_pos) / (rl.max_rtg_pos - rl.min_rtg_pos)
    rtgs[:, :, 1] = (np.clip(rtgs[:, :, 1], rl.min_rtg_veh, rl.max_rtg_veh) - rl.min_rtg_veh) / (rl.max_rtg_veh - rl.min_rtg_veh)
    rtgs[:, :, 2] = (np.clip(rtgs[:, :, 2], rl.min_rtg_road, rl.max_rtg_road) - rl.min_rtg_road) / (rl.max_rtg_road - rl.min_rtg_road)

    dead_ids = []
    focal_batches = []
    if adapter._vehicles_to_control_sorted:
        unaccounted_veh_ids = list(adapter._vehicles_to_control_sorted)
    else:
        unaccounted_veh_ids = list(adapter._vehicles_to_control)

    while len(unaccounted_veh_ids) > 0:
        focal_id = unaccounted_veh_ids[0]
        unaccounted_veh_ids.remove(focal_id)

        origin_agent_idx = policy.veh_id_to_idx[focal_id]
        if not policy.states[origin_agent_idx, t, -1]:
            dead_ids.append(focal_id)
            continue

        road_points = adapter._preproc_data['road_points'].copy()
        road_types = adapter._preproc_data['road_types'].copy()
        if len(road_points) == 0:
            dead_ids.append(focal_id)
            continue

        cur_data_veh_ids = [focal_id]
        rel_timesteps = np.repeat(
            np.expand_dims(timesteps, 0), rl.max_num_agents, axis=0
        )
        # 统一约定 timesteps shape 为 (A, T, 1)
        if rel_timesteps.ndim == 2:
            rel_timesteps = rel_timesteps[..., np.newaxis]

        if t == 0:
            policy.relevant_agent_idxs[focal_id] = []

        (
            rel_ag_states, rel_ag_types, rel_actions, rel_rtgs,
            rel_goals, rel_moving_agent_mask,
            new_agent_idx_dict, relevant_agent_idxs,
        ) = dset.select_relevant_agents(
            ag_states, ag_types, actions, rtgs, goals[:, 0],
            origin_agent_idx, normalize_timestep, moving_agent_mask,
            policy.relevant_agent_idxs[focal_id],
        )

        accounted_veh_ids = [
            policy.idx_to_veh_id[idx] for idx in new_agent_idx_dict.keys()
        ]
        for unacc in list(unaccounted_veh_ids):
            if unacc in accounted_veh_ids:
                cur_data_veh_ids.append(unacc)
                unaccounted_veh_ids.remove(unacc)

        if t == 0:
            for vid in cur_data_veh_ids:
                policy.relevant_agent_idxs[vid] = list(new_agent_idx_dict.keys())
        else:
            for vid in cur_data_veh_ids:
                policy.relevant_agent_idxs[vid] = relevant_agent_idxs

        new_origin_agent_idx = new_agent_idx_dict[origin_agent_idx]
        rel_actions = dset.discretize_actions(rel_actions)
        if policy.discretize_rtgs:
            rel_rtgs = dset.discretize_rtgs(rel_rtgs)
        rel_ag_states, rel_road_points, rel_road_types, rel_goals = dset.normalize_scene(
            rel_ag_states, road_points, road_types, rel_goals, new_origin_agent_idx
        )

        # — 在 from_numpy() 之前停下，返回原始 numpy dict —
        motion_data_np = {
            'agent_states': rel_ag_states,
            'agent_types': rel_ag_types,
            'goals': rel_goals,
            'actions': rel_actions,
            'rtgs': rel_rtgs,
            'timesteps': rel_timesteps,
            'moving_agent_mask': rel_moving_agent_mask,
            'road_points': rel_road_points,
            'road_types': rel_road_types,
        }

        # 供 V2 collate(pad+mask) 使用的变长元信息
        seq_len = int(rel_ag_states.shape[1])
        valid_agent_count = int(len(new_agent_idx_dict))
        # rel_road_types 对无效 polyline 使用 -1 padding（来自 normalize_scene）
        valid_road_count = int(np.sum(rel_road_types[:, 0] != -1))

        veh_ids_in_context = [
            policy.idx_to_veh_id[idx]
            for idx in policy.relevant_agent_idxs[focal_id]
        ]

        focal_batches.append({
            'focal_id': focal_id,
            'motion_data_np': motion_data_np,
            'new_agent_idx_dict': {int(k): int(v) for k, v in new_agent_idx_dict.items()},
            'data_veh_ids': cur_data_veh_ids,
            'veh_ids_in_context': veh_ids_in_context,
            'seq_len': seq_len,
            'valid_agent_count': valid_agent_count,
            'valid_road_count': valid_road_count,
            'predict_rtgs': bool(policy.predict_rtgs),
        })

    return focal_batches, dead_ids


def apply_predictions(adapter, model_outputs: Optional[Dict]) -> Dict[int, Tuple[float, float]]:
    """接收主进程推理结果，写回 vehicle_data_dict，返回 opponent actions。

    Args:
        adapter: CtrlSimOpponentAdapter 实例
        model_outputs: ExternalTeacher.batched_forward() 对该 env 的输出，
                      或 None（该 env 不需要 teacher）

    Returns:
        actions: {veh_id: (accel, steer)}
    """
    if adapter._policy is None:
        return {}

    model_outputs = unpack_model_outputs(model_outputs)
    if model_outputs is None:
        return {}

    step_t = model_outputs.get('step_t', 0)
    action_results = model_outputs.get('action_results', {})
    rtg_results = model_outputs.get('rtg_results', {})
    processed_rtg_veh_ids = model_outputs.get('processed_rtg_veh_ids', [])
    dead_ids = model_outputs.get('dead_ids', [])

    # 1. 写回 RTG 到 vehicle_data_dict
    for veh_id, (goal_val, veh_val, road_val) in rtg_results.items():
        if veh_id in adapter._vehicle_data_dict:
            adapter._vehicle_data_dict[veh_id]['next_rtg_goal'] = goal_val
            adapter._vehicle_data_dict[veh_id]['next_rtg_veh'] = veh_val
            adapter._vehicle_data_dict[veh_id]['next_rtg_road'] = road_val

    # predict_rtgs 后续 RTG 追加（对齐 predict() 末尾逻辑）
    if adapter._policy.predict_rtgs:
        for veh_id in adapter._vehicle_data_dict.keys():
            if veh_id in processed_rtg_veh_ids:
                adapter._vehicle_data_dict[veh_id]['rtgs'].append(np.array([
                    adapter._vehicle_data_dict[veh_id].get('next_rtg_goal', 0),
                    adapter._vehicle_data_dict[veh_id].get('next_rtg_veh', 0),
                    adapter._vehicle_data_dict[veh_id].get('next_rtg_road', 0),
                ]))
            else:
                adapter._vehicle_data_dict[veh_id]['rtgs'].append(
                    np.array([0] * adapter._policy.cfg_model.num_reward_components)
                )

    # 2. 写回 action 到 vehicle_data_dict
    for veh_id, (accel, steer) in action_results.items():
        if veh_id in adapter._vehicle_data_dict:
            adapter._vehicle_data_dict[veh_id]['next_acceleration'] = accel
            adapter._vehicle_data_dict[veh_id]['next_steering'] = steer

    # 3. dead agents → 空动作
    for veh_id in dead_ids:
        if veh_id in adapter._vehicle_data_dict:
            adapter._vehicle_data_dict[veh_id]['next_acceleration'] = 0.0
            adapter._vehicle_data_dict[veh_id]['next_steering'] = 0.0

    # 4. 提取最终 actions（同 step() 的提取逻辑）
    actions = {}
    for veh_id in adapter._vehicles_to_control:
        if step_t >= adapter.history_steps - 1:
            if veh_id in action_results:
                actions[veh_id] = action_results[veh_id]
            elif veh_id in dead_ids:
                actions[veh_id] = (0.0, 0.0)
            else:
                actions[veh_id] = (0.0, 0.0)
        else:
            # warm-up 期间用 GT action
            veh = adapter._last_vehicle_by_id.get(veh_id)
            actions[veh_id] = adapter._get_gt_action(veh_id, step_t, veh)

    return actions
