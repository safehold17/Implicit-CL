"""
负责在 episode 开始时重置对手状态服务，并初始化 policy/GT/道路等运行时缓存。
该模块连接场景输入、控制车辆集合和 adapter 内部状态，是一次 rollout 的初始化入口。
Resets the opponent state service at episode start and initializes policy, GT, road, and runtime caches.
Connects scenario inputs, controlled-vehicle sets, and adapter state as the rollout initialization entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ctrlsim_adapter.existence import sim_position_exists
from utils.sim import get_road_data

from ..existence_logic import (
    _compute_goal_hold_until,
    _keep_exists_on_invalid,
    _should_drop_after_goal,
)
from ..state_helpers import extract_road_edge_polylines


def reset(
    service: Any,
    scenario: Any,
    vehicles: List[Any],
    gt_data_dict: Dict,
    preproc_data: Dict,
    vehicles_to_control: List[int],
    ego_id: Optional[int],
    build_gt_action_target_cache_fn: Any,
) -> None:
    adapter = service.adapter
    if vehicles_to_control:
        if adapter._checkpoint_cfg is None:
            adapter._load_checkpoint_cfg()
            adapter._validate_external_cfg_compatibility()
        if adapter.dataset is None:
            adapter._load_dataset_only()
        adapter._policy = adapter._create_policy()
    else:
        adapter._policy = None

    adapter._gt_data_dict = gt_data_dict
    adapter._gt_traj_by_id = {
        veh_id: np.asarray(data["traj"])
        for veh_id, data in gt_data_dict.items()
        if isinstance(data, dict) and "traj" in data
    }
    adapter._gt_action_target_cache = build_gt_action_target_cache_fn(adapter._gt_traj_by_id)
    adapter._gt_action_runtime_cache = {}
    adapter._preproc_data = preproc_data
    adapter._vehicles_to_control = list(vehicles_to_control)
    adapter._vehicles_to_control_set = set(adapter._vehicles_to_control)
    if adapter._vehicles_to_control:
        trajectory_lengths = np.array(
            [
                int(gt_traj_data[:, 4].sum()) if gt_traj_data is not None else -1
                for gt_traj_data in (
                    service._get_gt_traj_data(veh_id)
                    for veh_id in adapter._vehicles_to_control
                )
            ],
            dtype=np.int64,
        )
        sorted_idxs = np.argsort(trajectory_lengths)[::-1]
        vehicles_np = np.asarray(adapter._vehicles_to_control, dtype=np.int64)
        adapter._vehicles_to_control_sorted = list(vehicles_np[sorted_idxs].tolist())
    else:
        adapter._vehicles_to_control_sorted = []
    adapter._ego_id = ego_id
    adapter._last_vehicles = None
    adapter._last_vehicle_by_id = {}

    road_data = get_road_data(scenario)
    adapter._road_edge_polylines = extract_road_edge_polylines(road_data)

    adapter._vehicle_data_dict = {}
    adapter._goal_dict = {}
    adapter._goal_dist_normalizer = {}
    adapter._opponent_vehicle_exits = {}
    adapter._opponent_last_valid_pos = {}
    adapter._opponent_goal_hold_until = {}
    adapter._moving_agent_mask_cache = None
    adapter._batch_prepare_cache = {}
    adapter._pending_sparse_actions_step_t = None
    adapter._pending_sparse_actions = {}
    adapter._constant_state_vehicle_ids = set()
    adapter._constant_state_by_id = {}
    static_speed_threshold = 1e-3

    adapter._veh_id_to_idx = {}
    for idx, veh in enumerate(vehicles):
        veh_id = veh.getID()
        adapter._veh_id_to_idx[veh_id] = idx
        gt_traj_data = service._get_gt_traj_data(veh_id)
        if gt_traj_data is None:
            raise KeyError(f"Missing gt traj data for veh_id={veh_id}")
        adapter._goal_dict[veh_id] = service._initialize_goal_dict(veh, gt_traj_data)
        adapter._vehicle_data_dict[veh_id] = service._initialize_vehicle_data_dict(
            veh,
            adapter._goal_dict[veh_id],
        )
        adapter._goal_dist_normalizer[veh_id] = service._compute_goal_dist_normalizer(
            veh,
            adapter._goal_dict[veh_id]["pos"],
        )
        if veh_id not in adapter._vehicles_to_control_set:
            speed = abs(float(veh.getSpeed()))
            if speed <= static_speed_threshold:
                pos = veh.getPosition()
                velocity = veh.velocity()
                gt_state = gt_traj_data[0]
                adapter._constant_state_vehicle_ids.add(veh_id)
                adapter._constant_state_by_id[veh_id] = {
                    "position": {"x": float(pos.x), "y": float(pos.y)},
                    "velocity": {"x": float(velocity.x), "y": float(velocity.y)},
                    "heading": float(veh.getHeading()),
                    "existence": 1 if sim_position_exists(pos.x, pos.y) else 0,
                    "gt_position": {"x": float(gt_state[0]), "y": float(gt_state[1])},
                    "gt_heading": float(gt_state[2]),
                    "gt_speed": float(gt_state[3]),
                }
        if veh_id in adapter._vehicles_to_control_set:
            pos = veh.getPosition()
            sim_exists = sim_position_exists(pos.x, pos.y)
            adapter._opponent_vehicle_exits[veh_id] = bool(sim_exists)
            if sim_exists:
                adapter._opponent_last_valid_pos[veh_id] = (float(pos.x), float(pos.y))
            adapter._opponent_goal_hold_until[veh_id] = None

    adapter._reward_service.prepare_road_edge_cache()

    adapter._all_vehicle_ids = list(adapter._vehicle_data_dict.keys())
    adapter._controlled_vehicle_ids_present = [
        veh_id
        for veh_id in adapter._vehicles_to_control
        if veh_id in adapter._vehicle_data_dict
    ]
    adapter._controlled_vehicle_ids_step = []
    adapter._state_update_vehicle_ids_step = list(adapter._all_vehicle_ids)
    adapter._step_vehicle_by_id = {}

    if adapter._policy is not None:
        adapter._policy.reset(adapter._vehicle_data_dict)
    adapter.sparse_inference.clear_on_reset()


def cache_last_valid_positions(service: Any, vehicles: List[Any]) -> None:
    adapter = service.adapter
    for veh in vehicles:
        veh_id = veh.getID()
        if veh_id not in adapter._vehicles_to_control_set:
            continue
        pos = veh.getPosition()
        if sim_position_exists(pos.x, pos.y):
            adapter._opponent_last_valid_pos[veh_id] = (float(pos.x), float(pos.y))


def get_opponent_vehicle_exists(service: Any, veh_id: int) -> Optional[bool]:
    adapter = service.adapter
    if veh_id not in adapter._vehicles_to_control_set:
        return None
    if veh_id in adapter._opponent_vehicle_exits:
        return bool(adapter._opponent_vehicle_exits[veh_id])
    exists_hist = adapter._vehicle_data_dict.get(veh_id, {}).get("existence")
    if exists_hist:
        return bool(exists_hist[-1])
    return None


def post_step_fix_opponent_positions(
    service: Any,
    vehicles: List[Any],
    goal_points_by_id: Optional[Dict[int, np.ndarray]],
    current_step: int,
) -> None:
    adapter = service.adapter
    if goal_points_by_id is None:
        goal_points_by_id = {}
    for veh in vehicles:
        veh_id = veh.getID()
        if veh_id not in adapter._vehicles_to_control_set:
            continue
        pos = veh.getPosition()
        sim_exists = sim_position_exists(pos.x, pos.y)
        prev_exists = adapter._opponent_vehicle_exits.get(veh_id, bool(sim_exists))

        if sim_exists:
            adapter._opponent_last_valid_pos[veh_id] = (float(pos.x), float(pos.y))
        elif prev_exists and veh_id in adapter._opponent_last_valid_pos:
            last_x, last_y = adapter._opponent_last_valid_pos[veh_id]
            try:
                veh.setPosition(last_x, last_y)
                pos = veh.getPosition()
                sim_exists = True
            except Exception:
                pass

        goal_pos = goal_points_by_id.get(veh_id)
        reached_goal = False
        if goal_pos is not None and sim_exists:
            try:
                goal_arr = np.asarray(goal_pos, dtype=np.float32)
                dist = np.linalg.norm(goal_arr[:2] - np.array([pos.x, pos.y], dtype=np.float32))
                reached_goal = dist < adapter._goal_pos_tolerance
            except Exception:
                reached_goal = False

        hold_until = adapter._opponent_goal_hold_until.get(veh_id)
        hold_until = _compute_goal_hold_until(
            hold_until,
            current_step=current_step,
            reached_goal=reached_goal,
            hold_steps=adapter._goal_hold_steps,
        )
        adapter._opponent_goal_hold_until[veh_id] = hold_until

        if _should_drop_after_goal(current_step, hold_until):
            adapter._opponent_vehicle_exits[veh_id] = False
            try:
                veh.setPosition(-1000000.0, -1000000.0)
            except Exception:
                pass
            continue

        adapter._opponent_vehicle_exits[veh_id] = _keep_exists_on_invalid(sim_exists, prev_exists)
