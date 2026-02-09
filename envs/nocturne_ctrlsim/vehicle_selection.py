"""
Vehicle selection helpers for Nocturne CtrlSim environment.
"""
from typing import List, Optional, Tuple

import numpy as np

from .scenario_helpers import get_vehicle_by_id


def get_preproc_vehicle_ids(env) -> Optional[List[int]]:
    """
    Get vehicle IDs from preprocessed data.

    Returns:
        List of vehicle IDs that have RTG data, or None if cannot determine.

    过滤掉：existence 在第 0 帧就是 0（第一帧就不存在）的车辆。
    这类车“只有一个时间步/没有有效轨迹”，无法用逆自行车模型定义动作，所以被认为没有有效时间步。
    """
    if not hasattr(env, "_preproc_data") or env._preproc_data is None:
        return None

    preproc = env._preproc_data

    # 尝试从 filtered_ag_ids 获取
    if isinstance(preproc, dict):
        filtered_ids = preproc.get("filtered_ag_ids")
    else:
        filtered_ids = getattr(preproc, "filtered_ag_ids", None)

    if filtered_ids is not None:
        return list(filtered_ids)

    # 如果没有 filtered_ag_ids，尝试从 GT data 推断
    # 假设预处理数据的顺序与 GT data 的某个子集一致
    if hasattr(env, "_gt_data_dict") and env._gt_data_dict:
        # 获取 RTG 数组大小
        if isinstance(preproc, dict):
            rtgs = preproc.get("rtgs")
        else:
            rtgs = getattr(preproc, "rtgs", None)

        if rtgs is not None:
            import torch

            if isinstance(rtgs, torch.Tensor):
                rtgs = rtgs.cpu().numpy()

            num_agents_in_rtg = rtgs.shape[0] if len(rtgs.shape) >= 3 else 0

            # 取 GT data 中前 num_agents_in_rtg 个车辆的 ID
            # 注意：这是一个启发式方法，可能不准确
            all_veh_ids = sorted(list(env._gt_data_dict.keys()))
            return all_veh_ids[:num_agents_in_rtg]

    return None


def get_moving_vehicle_ids(env, filter_by_preproc: bool = True) -> List[int]:
    """
    Get all moving vehicles IDs in the scenario.

    See: ctrl-sim utils/sim.py get_moving_vehicles() function.

    Args:
        filter_by_preproc: If True, only return vehicles that exist in preprocessed data.
    """
    all_moving_ids = [v.getID() for v in env.scenario.getObjectsThatMoved()]
    moving_ids = list(all_moving_ids)

    if filter_by_preproc and hasattr(env, "_preproc_data") and env._preproc_data is not None:
        # 获取预处理数据中的车辆ID列表
        preproc_veh_ids = get_preproc_vehicle_ids(env)
        if preproc_veh_ids is not None:
            # 只返回同时在 moving 和 preproc 中的车辆
            moving_ids = [vid for vid in moving_ids if vid in preproc_veh_ids]
            if len(moving_ids) < len(all_moving_ids):
                filtered_count = len(all_moving_ids) - len(moving_ids)
                print(f"Info: Filtered out {filtered_count} vehicles not in preprocessed data")

    return moving_ids


def find_interesting_pair(env, moving_veh_ids: List[int]) -> Optional[Tuple[int, int]]:
    """
    Find interesting vehicle pairs (see ctrl-sim policy_evaluator.py line 362-412).

    Selection criteria:
    - Target position is close (<10 meters)
    - Target time step is close (<20 steps)
    - Trajectory is long enough (>=60 steps)

    Returns:
        (veh_id_1, veh_id_2) tuple. If no interesting pair is found, caller
        should fall back to dense vehicle selection.
    """
    # Configuration thresholds (see ctrl-sim cfg.eval)
    goal_dist_threshold = 10.0  # meters
    timestep_diff_threshold = 20  # steps
    traj_len_threshold = 60  # steps
    history_steps = getattr(env.cfg.nocturne, "history_steps", 10)

    goals = []
    goal_timesteps = []
    valid_traj_mask = []
    veh_ids = []

    for veh_id in moving_veh_ids:
        if veh_id not in env._gt_data_dict:
            continue

        gt_traj = np.array(env._gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]

        # Calculate target position and time step
        idx_goal = env.max_episode_steps - 1
        idx_disappear = np.where(existence_mask == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1

        veh = get_vehicle_by_id(env, veh_id)
        if veh is None:
            continue

        goal_pos = np.array([veh.target_position.x, veh.target_position.y])
        if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
            goal_pos = gt_traj[idx_goal, :2]

        # Check trajectory length
        has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold

        goals.append(goal_pos)
        goal_timesteps.append(idx_goal - history_steps)
        valid_traj_mask.append(1 if has_valid_traj else 0)
        veh_ids.append(veh_id)

    if len(goals) < 2:
        return None

    goals = np.array(goals)
    goal_timesteps = np.array(goal_timesteps)
    valid_traj_mask = np.array(valid_traj_mask)

    # Calculate target distance matrix
    dists = np.linalg.norm(goals[:, np.newaxis] - goals[np.newaxis, :], axis=-1)

    # Build mask
    nearby_mask = dists < goal_dist_threshold
    not_same_mask = dists > 0
    valid_traj_both = np.outer(valid_traj_mask, valid_traj_mask)
    timestep_diff = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
    within_time_mask = timestep_diff < timestep_diff_threshold

    goal_mask = nearby_mask & not_same_mask & valid_traj_both.astype(bool) & within_time_mask

    indices = np.where(goal_mask)
    valid_pairs = list(zip(indices[0], indices[1]))

    if len(valid_pairs) == 0:
        return None

    # Deterministic selection: select first pair (sorted by index, to ensure consistency)
    pair_idx = valid_pairs[0]
    return (veh_ids[pair_idx[0]], veh_ids[pair_idx[1]])


def select_dense_vehicle(
    env,
    moving_veh_ids: List[int],
    k_neighbors: int = 7,
    traj_len_threshold: int = 30,
) -> Optional[int]:
    """Select vehicle with the smallest average distance to its nearest neighbors."""
    history_steps = getattr(env.cfg.nocturne, "history_steps", 10)
    positions = {}
    for veh_id in moving_veh_ids:
        veh = get_vehicle_by_id(env, veh_id)
        if veh is None:
            continue
        if veh_id not in env._gt_data_dict:
            continue
        gt_traj = np.array(env._gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]
        has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
        if not has_valid_traj:
            continue
        pos = veh.getPosition()
        positions[veh_id] = np.array([pos.x, pos.y], dtype=np.float32)

    if len(positions) == 0:
        return None
    if len(positions) == 1:
        return next(iter(positions.keys()))

    best_vid = None
    best_avg = None
    for vid, pos in positions.items():
        dists = []
        for other_id, other_pos in positions.items():
            if other_id == vid:
                continue
            dists.append(np.linalg.norm(pos - other_pos))
        if len(dists) == 0:
            continue
        dists.sort()
        k = min(k_neighbors, len(dists))
        avg_dist = float(np.mean(dists[:k]))
        if best_avg is None or avg_dist < best_avg or (avg_dist == best_avg and vid < best_vid):
            best_avg = avg_dist
            best_vid = vid

    return best_vid


def select_ego_vehicle(env):
    """
    Select ego vehicle using dynamic fallback logic.

    Use find_interesting_pair logic to select two interesting vehicles,
    then deterministically select the vehicle with smaller veh_id as ego.

    If no interesting pair is found, select the dense vehicle (smallest
    average distance to neighbors). If dense selection fails, fall back
    to the first moving vehicle.

    Note: This is the fallback method. Primary vehicle ID loading from JSON
    is handled in adversarial.py's _load_vehicle_ids_for_scenario().
    """
    # 1. Get moving vehicles (filtered by preprocessed data)
    moving_veh_ids = get_moving_vehicle_ids(env, filter_by_preproc=True)

    if len(moving_veh_ids) == 0:
        # 如果预处理数据过滤后没有车辆，尝试不过滤
        print(
            f"Warning: No vehicles in preprocessed data for scenario {env.current_level.scenario_id}. "
            "Using all moving vehicles."
        )
        moving_veh_ids = get_moving_vehicle_ids(env, filter_by_preproc=False)
        if len(moving_veh_ids) == 0:
            raise ValueError(
                f"No moving vehicles found in scenario {env.current_level.scenario_id}. "
                "Scenario will be skipped."
            )

    # 2. Find interesting pair
    interesting_pair = find_interesting_pair(env, moving_veh_ids)

    if interesting_pair is None:
        # If no interesting pair is found, downgrade to dense vehicle selection
        print(
            f"Warning: No interesting vehicle pair found in scenario {env.current_level.scenario_id}. "
            f"Using dense vehicle selection for ego."
        )
        ego_veh_id = select_dense_vehicle(env, moving_veh_ids)
        if ego_veh_id is None:
            ego_veh_id = moving_veh_ids[0]
    else:
        # 3. Deterministic selection: select vehicle with smaller veh_id as ego
        ego_veh_id = min(interesting_pair)

    return get_vehicle_by_id(env, ego_veh_id)


def select_opponent_vehicles(env, k: int = 7, traj_len_threshold: int = 10) -> None:
    """
    Select opponent vehicles using dynamic fallback logic.

    Selects k nearest moving vehicles to ego with valid trajectory length.

    Note: This is the fallback method. Primary vehicle ID loading from JSON
    is handled in adversarial.py's _load_vehicle_ids_for_scenario().

    Args:
        k: Maximum number of opponent vehicles to select.
        traj_len_threshold: Minimum trajectory length (default 10 steps).
    """
    if env.ego_vehicle is None:
        env.opponent_vehicles = []
        env.opponent_vehicle_ids = []
        return

    # Dynamic selection: get moving vehicles (excluding ego, filtered by preprocessed data)
    moving_veh_ids = get_moving_vehicle_ids(env, filter_by_preproc=True)
    ego_id = env.ego_vehicle.getID()
    candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]

    # 如果预处理数据过滤后没有候选车辆，尝试不过滤
    if len(candidate_ids) == 0:
        print(
            f"Warning: No opponent candidates in preprocessed data for scenario {env.current_level.scenario_id}. "
            "Using all moving vehicles."
        )
        moving_veh_ids = get_moving_vehicle_ids(env, filter_by_preproc=False)
        candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]

    if len(candidate_ids) == 0:
        env.opponent_vehicles = []
        env.opponent_vehicle_ids = []
        return

    # 2. Calculate distance to ego, with trajectory length filter
    ego_pos = env.ego_vehicle.getPosition()
    ego_pos = np.array([ego_pos.x, ego_pos.y])
    history_steps = getattr(env.cfg.nocturne, "history_steps", 10)

    distances = []
    for veh_id in candidate_ids:
        veh = get_vehicle_by_id(env, veh_id)
        if veh is None:
            continue

        # Check trajectory length constraint
        if veh_id not in env._gt_data_dict:
            continue
        gt_traj = np.array(env._gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]
        has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
        if not has_valid_traj:
            continue

        pos = veh.getPosition()
        dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
        distances.append((dist, veh_id, veh))

    # 3. Sort by distance, select k nearest vehicles
    distances.sort(key=lambda x: x[0])
    selected = distances[:k]

    env.opponent_vehicles = [item[2] for item in selected]
    env.opponent_vehicle_ids = [item[1] for item in selected]

