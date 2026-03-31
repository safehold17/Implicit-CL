"""
负责对手状态更新过程中常用的辅助操作，如道路边界提取、车辆筛选与状态写入。
该模块将零散的列表/字典处理逻辑从主更新流程中拆出，保持状态服务实现清晰。
Provides reusable helpers for opponent-state updates such as road-edge extraction, vehicle filtering, and state writes.
Pulls list/dict manipulation details out of the main update flow to keep the state service readable.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
