"""
负责对手状态更新过程中常用的辅助操作，如道路边界提取、车辆筛选与状态写入。
该模块将零散的列表/字典处理逻辑从主更新流程中拆出，保持状态服务实现清晰。
Provides reusable helpers for opponent-state updates such as road-edge extraction, vehicle filtering, and state writes.
Pulls list/dict manipulation details out of the main update flow to keep the state service readable.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ctrlsim_adapter.existence import sim_position_exists

from .existence_logic import _keep_exists_on_invalid, _should_drop_after_goal


def extract_road_edge_polylines(road_data: List[Dict]) -> List[np.ndarray]:
    road_edge_polylines: List[np.ndarray] = []
    for road in road_data:
        if road["type"] != "road_edge":
            continue
        geometry = road["geometry"]
        if not isinstance(geometry, list):
            continue
        polyline = np.array([[pt["x"], pt["y"]] for pt in geometry])
        road_edge_polylines.append(polyline)
    return road_edge_polylines


def get_state_update_vehicle_ids(
    adapter: Any,
    t: int,
    vehicles_by_id: Dict[int, Any],
) -> List[int]:
    controlled_ids = [
        veh_id
        for veh_id in adapter._controlled_vehicle_ids_present
        if veh_id in vehicles_by_id
    ]
    if not controlled_ids:
        return []

    all_existing_ids = [
        veh_id
        for veh_id in adapter._all_vehicle_ids
        if veh_id in vehicles_by_id
    ]
    policy = adapter._policy
    if t <= adapter.history_steps - 1 or policy is None:
        return all_existing_ids

    relevant_agent_idxs = policy.relevant_agent_idxs
    if not relevant_agent_idxs:
        return all_existing_ids

    update_id_set = set(controlled_ids)
    for veh_id in controlled_ids:
        for idx in relevant_agent_idxs.get(veh_id, ()):
            mapped_id = policy.idx_to_veh_id.get(int(idx))
            if mapped_id in vehicles_by_id:
                update_id_set.add(mapped_id)

    return [veh_id for veh_id in adapter._all_vehicle_ids if veh_id in update_id_set]


def store_next_action(
    veh_data: Dict[str, Any],
    veh_id: int,
    action: Tuple[float, float],
    actions: Dict[int, Tuple[float, float]],
) -> None:
    veh_data["next_acceleration"] = action[0]
    veh_data["next_steering"] = action[1]
    actions[veh_id] = action


def append_gt_state_for_step(
    veh_data: Dict[str, Any],
    gt_traj_data: np.ndarray,
    t: int,
    steps: int,
    dt: float,
    constant_state: Optional[Dict[str, Any]],
) -> None:
    if constant_state is not None:
        veh_data["gt_position"].append(constant_state["gt_position"].copy())
        veh_data["gt_heading"].append(constant_state["gt_heading"])
        veh_data["gt_speed"].append(constant_state["gt_speed"])
        veh_data["gt_acceleration"].append(0.0)
        return

    veh_data["gt_position"].append({"x": gt_traj_data[t, 0], "y": gt_traj_data[t, 1]})
    veh_data["gt_heading"].append(gt_traj_data[t, 2])
    veh_data["gt_speed"].append(gt_traj_data[t, 3])
    if t > 0 and t < steps - 1:
        gt_accel = (gt_traj_data[t + 1, 3] - gt_traj_data[t - 1, 3]) / (2 * dt)
        veh_data["gt_acceleration"].append(gt_accel)
    else:
        veh_data["gt_acceleration"].append(0)


def get_sim_state_entries(
    veh: Any,
    constant_state: Optional[Dict[str, Any]],
) -> Tuple[Any, Dict[str, float], Dict[str, float], float]:
    if constant_state is not None:
        return (
            None,
            constant_state["position"].copy(),
            constant_state["velocity"].copy(),
            constant_state["heading"],
        )

    pos = veh.getPosition()
    velocity = veh.velocity()
    return (
        pos,
        {"x": pos.x, "y": pos.y},
        {"x": velocity.x, "y": velocity.y},
        veh.getHeading(),
    )


def resolve_vehicle_exists(
    adapter: Any,
    *,
    veh_id: int,
    t: int,
    is_controlled: bool,
    ego_id: Optional[int],
    gt_traj_data: np.ndarray,
    pos: Any,
    constant_state: Optional[Dict[str, Any]],
) -> int:
    if constant_state is not None:
        return int(constant_state["existence"])

    protected = (veh_id == ego_id) or is_controlled
    if not protected:
        return int(gt_traj_data[t, 4])
    if not is_controlled:
        return 1 if sim_position_exists(pos.x, pos.y) else 0

    sim_exists = sim_position_exists(pos.x, pos.y)
    prev_exists = adapter._opponent_vehicle_exits.get(veh_id, bool(sim_exists))
    hold_until = adapter._opponent_goal_hold_until.get(veh_id)
    exists = _keep_exists_on_invalid(sim_exists, prev_exists)
    if _should_drop_after_goal(t, hold_until):
        exists = False
    adapter._opponent_vehicle_exits[veh_id] = bool(exists)
    return 1 if exists else 0
