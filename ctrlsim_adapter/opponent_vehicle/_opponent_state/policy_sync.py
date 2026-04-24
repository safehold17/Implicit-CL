"""
负责把适配器维护的逐车状态历史写回 ctrl-sim policy 的张量缓存。
该模块在每个时间步同步 states、actions、rtgs、goals 等字段，保持 policy 输入一致。
Writes adapter-maintained per-vehicle histories back into ctrl-sim policy tensor buffers.
Synchronizes states, actions, rtgs, goals, and related fields each timestep to keep policy inputs consistent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.data import get_object_type_onehot


def update_policy_state(service: Any, t: int) -> None:
    adapter = service.adapter
    policy = adapter._policy
    if policy is None:
        return

    states = policy.states
    types = policy.types
    actions = policy.actions
    rtgs = policy.rtgs
    timesteps = policy.timesteps
    goals = policy.goals
    goal_dim = policy.cfg_rl_waymo.goal_dim
    use_rtg = policy.use_rtg
    use_real_time_rtgs = policy.real_time_rewards and policy.use_rtg
    step_tensor_context = getattr(adapter, "_step_tensor_context", None)
    if (
        step_tensor_context is not None
        and int(step_tensor_context.step_t) == int(t)
        and step_tensor_context.policy_vehicle_indices is not None
        and step_tensor_context.positions_xy is not None
        and step_tensor_context.velocities_xy is not None
        and step_tensor_context.headings is not None
        and step_tensor_context.existence is not None
        and step_tensor_context.timesteps is not None
        and step_tensor_context.goals_state is not None
    ):
        policy_vehicle_indices = step_tensor_context.policy_vehicle_indices
        states[policy_vehicle_indices, t, 0:2] = step_tensor_context.positions_xy
        states[policy_vehicle_indices, t, 2:4] = step_tensor_context.velocities_xy
        states[policy_vehicle_indices, t, 4] = step_tensor_context.headings
        states[policy_vehicle_indices, t, 5] = np.asarray(
            [
                float(adapter._vehicle_data_dict[veh_id]["length"])
                for veh_id in step_tensor_context.update_vehicle_ids
            ],
            dtype=np.float32,
        )
        states[policy_vehicle_indices, t, 6] = np.asarray(
            [
                float(adapter._vehicle_data_dict[veh_id]["width"])
                for veh_id in step_tensor_context.update_vehicle_ids
            ],
            dtype=np.float32,
        )
        states[policy_vehicle_indices, t, 7] = step_tensor_context.existence

        if t == 0:
            for veh_idx, veh_id in zip(
                policy_vehicle_indices.tolist(),
                step_tensor_context.update_vehicle_ids,
            ):
                types[veh_idx] = get_object_type_onehot(
                    adapter._vehicle_data_dict[veh_id]["type"]
                )

        timesteps[policy_vehicle_indices, t, 0] = step_tensor_context.timesteps
        if t > 0 and step_tensor_context.latest_actions is not None:
            actions[policy_vehicle_indices, t - 1] = step_tensor_context.latest_actions
            if (
                use_rtg
                and step_tensor_context.latest_rtgs is not None
                and step_tensor_context.latest_rtgs.shape[0] == len(policy_vehicle_indices)
                and step_tensor_context.latest_rtgs.shape[1] > 0
            ):
                rtgs[policy_vehicle_indices, t - 1] = step_tensor_context.latest_rtgs
        if (
            use_real_time_rtgs
            and step_tensor_context.current_rtgs is not None
            and step_tensor_context.current_rtgs.shape[0] == len(policy_vehicle_indices)
            and step_tensor_context.current_rtgs.shape[1] > 0
        ):
            rtgs[policy_vehicle_indices, t] = step_tensor_context.current_rtgs

        goal_width = min(int(goal_dim), int(step_tensor_context.goals_state.shape[1]))
        goals[policy_vehicle_indices, t, :goal_width] = step_tensor_context.goals_state[
            :, :goal_width
        ]
        return

    for veh_id in adapter._state_update_vehicle_ids_step:
        veh_data = adapter._vehicle_data_dict.get(veh_id)
        if veh_data is None or len(veh_data["position"]) <= t:
            continue

        veh_idx = policy.veh_id_to_idx[veh_id]
        state_slot = states[veh_idx, t]
        state_slot[0] = veh_data["position"][t]["x"]
        state_slot[1] = veh_data["position"][t]["y"]
        state_slot[2] = veh_data["velocity"][t]["x"]
        state_slot[3] = veh_data["velocity"][t]["y"]
        state_slot[4] = veh_data["heading"][t]
        state_slot[5] = veh_data["length"]
        state_slot[6] = veh_data["width"]
        state_slot[7] = veh_data["existence"][t]

        if t == 0:
            types[veh_idx] = get_object_type_onehot(veh_data["type"])
        timesteps[veh_idx, t, 0] = veh_data["timestep"][t]

        rtg_hist = veh_data["rtgs"]
        if t > 0:
            action_slot = actions[veh_idx, t - 1]
            action_slot[0] = veh_data["acceleration"][t - 1]
            action_slot[1] = veh_data["steering"][t - 1]
            if use_rtg and len(rtg_hist) > t - 1:
                rtgs[veh_idx, t - 1] = rtg_hist[t - 1]

        if use_real_time_rtgs and len(rtg_hist) > t:
            rtgs[veh_idx, t] = rtg_hist[t]

        goal_slot = goals[veh_idx, t]
        goal_slot[0] = veh_data["goal_position"]["x"]
        goal_slot[1] = veh_data["goal_position"]["y"]
        if goal_dim > 2:
            goal_slot[2] = veh_data["goal_velocity_x"]
        if goal_dim > 3:
            goal_slot[3] = veh_data["goal_velocity_y"]
        if goal_dim > 4:
            goal_slot[4] = veh_data["goal_heading"]
