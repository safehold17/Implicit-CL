"""
负责在每个仿真步更新 adapter 内部的 vehicle_data_dict 与相关缓存。
该模块汇总车辆真实状态、奖励、存在性与 GT 信息，为后续 policy 同步和批量准备提供输入。
Updates adapter-side `vehicle_data_dict` and related caches on every simulation step.
Collects live vehicle state, reward, existence, and GT data for later policy sync and batch preparation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import nocturne
import numpy as np

from ..step_tensor_context import build_runtime_step_tensor_context
from .existence import resolve_vehicle_exists
from .state_helpers import get_sim_state_entries, get_state_update_vehicle_ids


def _angle_sub_array(
    current_angle: np.ndarray,
    target_angle: np.ndarray,
) -> np.ndarray:
    """按数组批量计算最短包角差。 / Compute wrapped angular differences for arrays."""
    diff = (target_angle - current_angle) % (2 * np.pi)
    wrapped_mask = diff > np.pi
    if np.any(wrapped_mask):
        diff = diff.copy()
        diff[wrapped_mask] = -(2 * np.pi - diff[wrapped_mask])
    return diff


def _compute_rewards_batched(
    *,
    rew_cfg: Dict[str, Any],
    vehicles_by_id: Dict[int, Any],
    update_vehicle_ids: List[int],
    vehicle_data_dict: Dict[int, Dict[str, Any]],
    goal_dict: Dict[int, Dict[str, Any]],
    goal_dist_normalizer: Dict[int, float],
    collision_fix: bool,
) -> np.ndarray:
    """按当前 `compute_reward` 语义批量计算 reward。 / Compute per-vehicle rewards in one vectorized pass."""
    reward_count = len(update_vehicle_ids)
    if reward_count == 0:
        return np.zeros((0, 8), dtype=np.float32)

    obj_pos = np.asarray(
        [
            [float(vehicles_by_id[veh_id].position.x), float(vehicles_by_id[veh_id].position.y)]
            for veh_id in update_vehicle_ids
        ],
        dtype=np.float32,
    )
    obj_speed = np.asarray(
        [float(vehicles_by_id[veh_id].speed) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    obj_heading = np.asarray(
        [float(vehicles_by_id[veh_id].heading) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    goal_pos = np.asarray(
        [np.asarray(goal_dict[veh_id]["pos"], dtype=np.float32) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    goal_speed = np.asarray(
        [float(goal_dict[veh_id]["speed"]) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    goal_heading = np.asarray(
        [float(goal_dict[veh_id]["heading"]) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    goal_dist_norm = np.asarray(
        [float(goal_dist_normalizer.get(veh_id, 1.0)) for veh_id in update_vehicle_ids],
        dtype=np.float32,
    )
    goal_dist_norm = np.where(goal_dist_norm == 0.0, 1.0, goal_dist_norm)
    previous_position_achieved = np.asarray(
        [
            float(vehicle_data_dict[veh_id]["reward"][-1][0])
            if vehicle_data_dict[veh_id]["reward"]
            else 0.0
            for veh_id in update_vehicle_ids
        ],
        dtype=np.float32,
    )

    position_target_achieved = np.ones((reward_count,), dtype=np.float32)
    speed_target_achieved = np.ones((reward_count,), dtype=np.float32)
    heading_target_achieved = np.ones((reward_count,), dtype=np.float32)

    goal_distance = np.linalg.norm(goal_pos - obj_pos, axis=1)
    heading_error = np.abs(_angle_sub_array(obj_heading, goal_heading))
    speed_error = np.abs(goal_speed - obj_speed)

    if rew_cfg["position_target"]:
        position_target_achieved = np.where(
            previous_position_achieved.astype(bool),
            1.0,
            (goal_distance < float(rew_cfg["position_target_tolerance"])).astype(np.float32),
        )
    if rew_cfg["speed_target"]:
        speed_target_achieved = (
            speed_error < float(rew_cfg["speed_target_tolerance"])
        ).astype(np.float32)
    if rew_cfg["heading_target"]:
        heading_target_achieved = (
            heading_error < float(rew_cfg["heading_target_tolerance"])
        ).astype(np.float32)

    pos_goal_rew = np.zeros((reward_count,), dtype=np.float32)
    speed_goal_rew = np.zeros((reward_count,), dtype=np.float32)
    heading_goal_rew = np.zeros((reward_count,), dtype=np.float32)
    if rew_cfg["shaped_goal_distance"] and rew_cfg["position_target"]:
        goal_dist_scaling = float(rew_cfg.get("shaped_goal_distance_scaling", 1.0))
        reward_scaling = float(rew_cfg["reward_scaling"])
        max_goal_reward = goal_dist_scaling / reward_scaling
        pos_goal_rew = goal_dist_scaling * (1.0 - goal_distance / goal_dist_norm) / reward_scaling
        pos_goal_rew = np.where(previous_position_achieved.astype(bool), max_goal_reward, pos_goal_rew)
        if rew_cfg["speed_target"]:
            speed_goal_rew = goal_dist_scaling * (1.0 - speed_error / 40.0) / reward_scaling
        if rew_cfg["heading_target"]:
            heading_goal_rew = goal_dist_scaling * (1.0 - heading_error / (2.0 * np.pi)) / reward_scaling

    if collision_fix:
        veh_veh_collision = np.asarray(
            [
                float(
                    vehicles_by_id[veh_id].collision_type_veh
                    == nocturne.CollisionType.VEHICLE_VEHICLE
                )
                for veh_id in update_vehicle_ids
            ],
            dtype=np.float32,
        )
        veh_edge_collision = np.asarray(
            [
                float(
                    vehicles_by_id[veh_id].collision_type_edge
                    == nocturne.CollisionType.VEHICLE_ROAD
                )
                for veh_id in update_vehicle_ids
            ],
            dtype=np.float32,
        )
    else:
        veh_veh_collision = np.asarray(
            [
                float(
                    vehicles_by_id[veh_id].collision_type
                    == nocturne.CollisionType.VEHICLE_VEHICLE
                )
                for veh_id in update_vehicle_ids
            ],
            dtype=np.float32,
        )
        veh_edge_collision = np.asarray(
            [
                float(
                    vehicles_by_id[veh_id].collision_type
                    == nocturne.CollisionType.VEHICLE_ROAD
                )
                for veh_id in update_vehicle_ids
            ],
            dtype=np.float32,
        )

    return np.stack(
        [
            position_target_achieved,
            heading_target_achieved,
            speed_target_achieved,
            pos_goal_rew,
            speed_goal_rew,
            heading_goal_rew,
            veh_veh_collision,
            veh_edge_collision,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


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
    step_vehicle_by_id: Dict[int, Any] = {}
    controlled_vehicle_ids_step: List[int] = []
    vehicles_by_id = getattr(adapter, "_vehicles_by_id_step", None)
    if vehicles_by_id is None:
        vehicles_by_id = {}
        adapter._vehicles_by_id_step = vehicles_by_id
    # The vehicles list is episode-stable (IDs never change mid-episode), so
    # rebuild the dict only at t=0.
    if t == 0:
        vehicles_by_id.clear()
        for veh in vehicles:
            vehicles_by_id[veh.getID()] = veh

    update_vehicle_ids = get_state_update_vehicle_ids(
        adapter=adapter,
        t=t,
        vehicles_by_id=vehicles_by_id,
    )
    reward_values = _compute_rewards_batched(
        rew_cfg=rew_cfg,
        vehicles_by_id=vehicles_by_id,
        update_vehicle_ids=update_vehicle_ids,
        vehicle_data_dict=vehicle_data_dict,
        goal_dict=goal_dict,
        goal_dist_normalizer=goal_dist_normalizer,
        collision_fix=collision_fix,
    )
    for reward_row, veh_id in zip(reward_values, update_vehicle_ids):
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

        veh_data["reward"].append(reward_row)

    adapter._controlled_vehicle_ids_step = controlled_vehicle_ids_step
    adapter._state_update_vehicle_ids_step = update_vehicle_ids
    adapter._step_vehicle_by_id = step_vehicle_by_id
    build_runtime_step_tensor_context(
        adapter=adapter,
        t=t,
        vehicle_data_dict=vehicle_data_dict,
    )

    if adapter._policy.real_time_rewards:
        return adapter._compute_dense_reward(t, vehicle_data_dict)
    return adapter._compute_nearest_dist_all(t, vehicle_data_dict)
