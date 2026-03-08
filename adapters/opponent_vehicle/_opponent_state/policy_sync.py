from __future__ import annotations

from typing import Any

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

