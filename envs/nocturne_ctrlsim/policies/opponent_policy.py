"""
Opponent-policy related helper functions for Nocturne CtrlSim adversarial env.
"""

from typing import Optional, Tuple

import numpy as np

from tools.safe_bicycle import safe_backward_action_from_states

from ..scenario_helpers import get_vehicle_by_id


def _build_gt_action_target_cache(gt_traj_cache: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    cache: dict[int, dict[str, np.ndarray]] = {}
    for veh_id, gt_traj in gt_traj_cache.items():
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


def build_episode_gt_action_cache(env) -> None:
    gt_traj_cache = getattr(env, "_gt_traj_cache", {})
    env._gt_action_target_cache = _build_gt_action_target_cache(gt_traj_cache)
    env._gt_action_runtime_cache = {}


def _get_gt_traj_array(env, veh_id: int) -> Optional[np.ndarray]:
    if veh_id not in env._gt_data_dict:
        return None

    gt_traj_cache = getattr(env, "_gt_traj_cache", None)
    if gt_traj_cache is not None and veh_id in gt_traj_cache:
        return gt_traj_cache[veh_id]

    gt_traj_data = np.asarray(env._gt_data_dict[veh_id]["traj"])
    if gt_traj_cache is not None:
        gt_traj_cache[veh_id] = gt_traj_data
    return gt_traj_data


def initialize_ego_goal_state(env) -> None:
    """
    Initialize ego vehicle's target and reward related state.

    See: ctrl-sim evaluator.py initialize_goal_dict() and compute_goal_dist_normalizer()
    """
    if env.ego_vehicle is None:
        return

    ego_id = env.ego_vehicle.getID()

    # Get GT trajectory data
    if ego_id not in env._gt_data_dict:
        return

    gt_traj_data = _get_gt_traj_array(env, ego_id)
    if gt_traj_data is None:
        return

    # Calculate target position (see evaluator.py initialize_goal_dict)
    goal_pos = np.array([
        env.ego_vehicle.target_position.x,
        env.ego_vehicle.target_position.y,
    ])
    goal_heading = env.ego_vehicle.target_heading
    goal_speed = env.ego_vehicle.target_speed

    # Check if vehicle disappears before trajectory ends, if so, use last valid position as target
    existence_mask = gt_traj_data[:, 4]
    idx_disappear = np.where(existence_mask == 0)[0]
    if len(idx_disappear) > 0:
        idx_goal = idx_disappear[0] - 1
        if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
            goal_pos = gt_traj_data[idx_goal, :2]
            goal_heading = gt_traj_data[idx_goal, 2]
            goal_speed = gt_traj_data[idx_goal, 3]

    env._ego_goal_dict = {
        'pos': goal_pos,
        'heading': goal_heading,
        'speed': goal_speed,
    }

    # Calculate target distance normalization factor
    ego_pos = env.ego_vehicle.getPosition()
    ego_pos = np.array([ego_pos.x, ego_pos.y])
    dist = np.linalg.norm(ego_pos - goal_pos)
    env._ego_goal_dist_normalizer = dist if dist > 0 else 1.0

    # Initialize ego's vehicle_data_dict (for reward calculation)
    env._ego_vehicle_data_dict = {
        ego_id: {
            'reward': [],
            'position': [],
            'heading': [],
            'speed': [],
        }
    }



def get_goal_point_for_vehicle(env, veh_id: int) -> Optional[np.ndarray]:
    veh = get_vehicle_by_id(env, veh_id)
    if veh is None:
        return None
    if veh_id not in env._gt_data_dict:
        return None

    gt_traj_data = _get_gt_traj_array(env, veh_id)
    if gt_traj_data is None:
        return None
    goal_pos = np.array([veh.target_position.x, veh.target_position.y])

    existence_mask = gt_traj_data[:, 4]
    idx_disappear = np.where(existence_mask == 0)[0]
    if len(idx_disappear) > 0:
        idx_goal = idx_disappear[0] - 1
        if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
            goal_pos = gt_traj_data[idx_goal, :2]

    if not np.isfinite(goal_pos).all():
        return None

    return goal_pos



def get_gt_action(env, veh_id: int, t: int, veh=None) -> Optional[Tuple[float, float]]:
    """
    Get vehicle's action from GT trajectory data at time step t.

    See: ctrl-sim policy_evaluator.py apply_gt_action()
    """
    if veh_id not in env._gt_data_dict:
        return None

    gt_traj = _get_gt_traj_array(env, veh_id)
    if gt_traj is None:
        return None

    gt_action_target_cache = getattr(env, "_gt_action_target_cache", None)
    if gt_action_target_cache is None:
        build_episode_gt_action_cache(env)
        gt_action_target_cache = env._gt_action_target_cache
    target = gt_action_target_cache.get(veh_id)

    # Check if time step is valid
    if t < 0 or t >= len(gt_traj) - 1:
        return (0.0, 0.0)

    # Check if vehicle exists in current and next time step.
    # In replay mode, opponent may not control this vehicle and adapter can return None;
    # fall back to GT existence in that case.
    # gt trajectory: [pos_x, pos_y, heading, speed, existence, goal_x, goal_y, length]
    if target is None:
        gt_exists = bool(gt_traj[t, 4] and gt_traj[t + 1, 4])
    else:
        gt_exists = bool(target["curr_exists"][t] and target["next_exists"][t])
    if veh_id in env.opponent_vehicle_ids and env.opponent is not None:
        exists = env.opponent.get_opponent_vehicle_exists(veh_id)
        veh_exists = int(gt_exists if exists is None else bool(exists))
    else:
        veh_exists = int(gt_exists)
    # Once missing, remain missing (align ctrl-sim evaluator)
    ego_data = env.opponent.get_vehicle_data(veh_id) if env.opponent else None
    if t > 0 and ego_data:
        existence_history = ego_data.get("existence")
        if existence_history and existence_history[-1] == 0:
            veh_exists = 0

    if not veh_exists or veh is None:
        return (0.0, 0.0)

    pos = veh.getPosition()
    heading = float(veh.getHeading())
    speed = float(veh.getSpeed())
    runtime_cache = getattr(env, "_gt_action_runtime_cache", None)
    if runtime_cache is None:
        runtime_cache = {}
        env._gt_action_runtime_cache = runtime_cache
    cache_key = (
        int(veh_id),
        int(t),
        float(pos.x),
        float(pos.y),
        heading,
        speed,
    )
    cached_action = runtime_cache.get(cache_key)
    if cached_action is not None:
        return cached_action

    if target is None:
        next_pos = gt_traj[t + 1, :2]
        next_heading = float(gt_traj[t + 1, 2])
        next_speed = float(gt_traj[t + 1, 3])
        wheel_base = float(gt_traj[t + 1, -1])
    else:
        next_pos = target["next_pos"][t]
        next_heading = float(target["next_heading"][t])
        next_speed = float(target["next_speed"][t])
        wheel_base = float(target["wheel_base"][t])

    accel, steer = safe_backward_action_from_states(
        prev_pos=(float(pos.x), float(pos.y)),
        prev_theta=heading,
        prev_vel=speed,
        curr_pos=(float(next_pos[0]), float(next_pos[1])),
        curr_theta=next_heading,
        curr_vel=next_speed,
        wheel_base=wheel_base,
        dt=env.dt,
    )

    action = (float(accel), float(steer))
    runtime_cache[cache_key] = action
    return action


def is_ego_position_reached(env) -> bool:
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return False

    ego_pos = env.ego_vehicle.getPosition()
    ego_pos_arr = np.array([ego_pos.x, ego_pos.y])
    goal_pos = env._ego_goal_dict.get('pos')
    if goal_pos is None:
        return False
    dist_to_goal = np.linalg.norm(goal_pos - ego_pos_arr)
    return dist_to_goal < 1.0
