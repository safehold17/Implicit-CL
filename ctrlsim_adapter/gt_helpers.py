"""
负责从 ground-truth 轨迹中提取缓存，并重建对手策略训练/评估所需的目标动作与状态。
该模块服务于 reset、逐步更新和奖励计算过程中的 GT 查询与反推。
Provides cached ground-truth trajectory access plus action/state reconstruction helpers for opponent policies.
Supports reset, step updates, and reward computation when querying or reconstructing GT signals.
"""

from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from tools.safe_bicycle import safe_backward_action_from_states


def build_gt_action_target_cache(
    gt_traj_by_id: Mapping[int, np.ndarray],
) -> Dict[int, Dict[str, np.ndarray]]:
    cache: Dict[int, Dict[str, np.ndarray]] = {}
    for veh_id, gt_traj in gt_traj_by_id.items():
        traj = np.asarray(gt_traj)
        if traj.ndim != 2 or traj.shape[0] < 2:
            continue
        cache[int(veh_id)] = {
            "curr_exists": traj[:-1, 4].astype(bool, copy=False),
            "next_exists": traj[1:, 4].astype(bool, copy=False),
            "next_pos": traj[1:, :2].astype(np.float32, copy=False),
            "next_heading": traj[1:, 2].astype(np.float32, copy=False),
            "next_speed": traj[1:, 3].astype(np.float32, copy=False),
            "wheel_base": traj[1:, -1].astype(np.float32, copy=False),
        }
    return cache


def get_gt_traj_array(
    gt_data_dict: Mapping[int, Any],
    gt_traj_cache: Optional[MutableMapping[int, np.ndarray]],
    veh_id: int,
) -> Optional[np.ndarray]:
    if gt_traj_cache is not None and veh_id in gt_traj_cache:
        return gt_traj_cache[veh_id]

    data = gt_data_dict.get(veh_id)
    if not isinstance(data, dict) or "traj" not in data:
        return None

    gt_traj_data = np.asarray(data["traj"])
    if gt_traj_cache is not None:
        gt_traj_cache[veh_id] = gt_traj_data
    return gt_traj_data


def resolve_goal_state(
    target_position: np.ndarray,
    target_heading: float,
    target_speed: float,
    gt_traj_data: np.ndarray,
) -> Dict[str, Any]:
    goal_pos = np.asarray(target_position, dtype=np.float32)
    goal_heading = float(target_heading)
    goal_speed = float(target_speed)

    idx_disappear = np.where(gt_traj_data[:, 4] == 0)[0]
    if len(idx_disappear) > 0:
        idx_goal = int(idx_disappear[0]) - 1
        if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
            goal_pos = gt_traj_data[idx_goal, :2].astype(np.float32, copy=False)
            goal_heading = float(gt_traj_data[idx_goal, 2])
            goal_speed = float(gt_traj_data[idx_goal, 3])

    return {
        "pos": goal_pos,
        "heading": goal_heading,
        "speed": goal_speed,
    }


def resolve_next_gt_state(
    gt_traj: np.ndarray,
    target_cache: Optional[Mapping[int, Dict[str, np.ndarray]]],
    veh_id: int,
    t: int,
) -> Tuple[np.ndarray, float, float, float]:
    target = target_cache.get(veh_id) if target_cache is not None else None
    if target is None:
        return (
            gt_traj[t + 1, :2].astype(np.float32, copy=False),
            float(gt_traj[t + 1, 2]),
            float(gt_traj[t + 1, 3]),
            float(gt_traj[t + 1, -1]),
        )

    return (
        target["next_pos"][t],
        float(target["next_heading"][t]),
        float(target["next_speed"][t]),
        float(target["wheel_base"][t]),
    )


def reconstruct_gt_action(
    pos_x: float,
    pos_y: float,
    heading: float,
    speed: float,
    next_pos: np.ndarray,
    next_heading: float,
    next_speed: float,
    wheel_base: float,
    dt: float,
) -> Tuple[float, float]:
    accel, steer = safe_backward_action_from_states(
        prev_pos=(float(pos_x), float(pos_y)),
        prev_theta=float(heading),
        prev_vel=float(speed),
        curr_pos=(float(next_pos[0]), float(next_pos[1])),
        curr_theta=float(next_heading),
        curr_vel=float(next_speed),
        wheel_base=float(wheel_base),
        dt=float(dt),
    )
    return (float(accel), float(steer))


def compute_goal_dist_normalizer(
    current_position: np.ndarray,
    goal_pos: np.ndarray,
) -> float:
    dist = np.linalg.norm(np.asarray(current_position) - np.asarray(goal_pos))
    return float(dist) if dist > 0 else 1.0
