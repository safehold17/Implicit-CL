from __future__ import annotations

from typing import Any, Dict, List

from ..opponent_state_helpers import (
    append_gt_state_for_step,
    get_sim_state_entries,
    get_state_update_vehicle_ids,
    resolve_vehicle_exists,
)


def update_vehicle_data_dict(
    service: Any,
    t: int,
    vehicles: List[Any],
    vehicle_data_dict: Dict,
) -> Dict:
    adapter = service.adapter
    vehicles_to_control_set = getattr(
        adapter,
        "_vehicles_to_control_set",
        set(getattr(adapter, "_vehicles_to_control", [])),
    )
    ego_id = getattr(adapter, "_ego_id", None)
    rew_cfg = adapter.cfg.nocturne["rew_cfg"]
    collision_fix = getattr(adapter.cfg.nocturne, "collision_fix", True)
    goal_dict = adapter._goal_dict
    goal_dist_normalizer = adapter._goal_dist_normalizer
    from .. import opponent_adapter as _opponent_adapter_module

    reward_fn = _opponent_adapter_module.compute_reward
    step_vehicle_by_id: Dict[int, Any] = {}
    controlled_vehicle_ids_step: List[int] = []
    vehicles_by_id = getattr(adapter, "_vehicles_by_id_step", None)
    if vehicles_by_id is None:
        vehicles_by_id = {}
        adapter._vehicles_by_id_step = vehicles_by_id
    vehicles_by_id.clear()
    for veh in vehicles:
        vehicles_by_id[veh.getID()] = veh

    update_vehicle_ids = get_state_update_vehicle_ids(
        adapter=adapter,
        t=t,
        vehicles_by_id=vehicles_by_id,
    )
    for veh_id in update_vehicle_ids:
        veh = vehicles_by_id[veh_id]
        gt_traj_data = service._get_gt_traj_data(veh_id)
        if gt_traj_data is None:
            continue
        veh_data = vehicle_data_dict[veh_id]
        veh_idx = adapter._veh_id_to_idx.get(veh_id, 0)
        is_controlled = veh_id in vehicles_to_control_set
        constant_state = (
            adapter._constant_state_by_id.get(veh_id)
            if (not is_controlled and veh_id in adapter._constant_state_vehicle_ids)
            else None
        )

        append_gt_state_for_step(
            veh_data=veh_data,
            gt_traj_data=gt_traj_data,
            t=t,
            steps=adapter.steps,
            dt=adapter.dt,
            constant_state=constant_state,
        )

        pos, pos_entry, velocity_entry, heading = get_sim_state_entries(
            veh=veh,
            constant_state=constant_state,
        )
        veh_data["position"].append(pos_entry)
        veh_data["velocity"].append(velocity_entry)
        veh_data["heading"].append(heading)
        veh_data["timestep"].append(t)

        if is_controlled:
            controlled_vehicle_ids_step.append(veh_id)
            step_vehicle_by_id[veh_id] = veh

        veh_exists = resolve_vehicle_exists(
            adapter=adapter,
            veh_id=veh_id,
            t=t,
            is_controlled=is_controlled,
            ego_id=ego_id,
            gt_traj_data=gt_traj_data,
            pos=pos,
            constant_state=constant_state,
        )
        if t > 0 and not is_controlled and veh_data["existence"][-1] == 0:
            veh_exists = 0
        veh_data["existence"].append(veh_exists)

        if t == 0:
            veh_data["rtgs"].append(service._get_initial_rtg(veh_id, veh_idx, t))
        else:
            dense_rewards = veh_data["dense_reward"]
            if dense_rewards:
                veh_data["rtgs"].append(veh_data["rtgs"][-1] - dense_rewards[-1])

        reward = reward_fn(
            rew_cfg,
            veh,
            goal_dict[veh_id],
            goal_dist_normalizer[veh_id],
            vehicle_data_dict,
            collision_fix=collision_fix,
        )
        veh_data["reward"].append(reward)

    adapter._controlled_vehicle_ids_step = controlled_vehicle_ids_step
    adapter._state_update_vehicle_ids_step = update_vehicle_ids
    adapter._step_vehicle_by_id = step_vehicle_by_id

    if adapter._policy.real_time_rewards:
        return adapter._compute_dense_reward(t, vehicle_data_dict)
    return adapter._compute_nearest_dist_all(t, vehicle_data_dict)
