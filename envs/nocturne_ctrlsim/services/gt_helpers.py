"""
GT and episode helper functions for Nocturne CtrlSim adversarial env.
"""

from typing import Optional, Tuple

import numpy as np

from ctrlsim_adapter.opponent_vehicle._opponent_state.gt_helpers import (
    build_gt_action_target_cache,
    compute_goal_dist_normalizer,
    get_gt_traj_array,
    reconstruct_gt_action,
    resolve_goal_state,
    resolve_next_gt_state,
)

from .scenario_helpers import get_vehicle_by_id


def build_episode_gt_action_cache(env) -> None:
    gt_traj_cache = getattr(env, "_gt_traj_cache", {})
    env._gt_action_target_cache = build_gt_action_target_cache(gt_traj_cache)
    env._gt_action_runtime_cache = {}


def _get_gt_traj_array(env, veh_id: int) -> Optional[np.ndarray]:
    return get_gt_traj_array(
        getattr(env, "_gt_data_dict", {}),
        getattr(env, "_gt_traj_cache", None),
        veh_id,
    )


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

    env._ego_goal_dict = resolve_goal_state(
        target_position=np.array(
            [
                env.ego_vehicle.target_position.x,
                env.ego_vehicle.target_position.y,
            ],
            dtype=np.float32,
        ),
        target_heading=env.ego_vehicle.target_heading,
        target_speed=env.ego_vehicle.target_speed,
        gt_traj_data=gt_traj_data,
    )

    # Calculate target distance normalization factor
    ego_pos = env.ego_vehicle.getPosition()
    env._ego_goal_dist_normalizer = compute_goal_dist_normalizer(
        np.array([ego_pos.x, ego_pos.y], dtype=np.float32),
        env._ego_goal_dict["pos"],
    )

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
    goal_dict = resolve_goal_state(
        target_position=np.array(
            [veh.target_position.x, veh.target_position.y],
            dtype=np.float32,
        ),
        target_heading=veh.target_heading,
        target_speed=veh.target_speed,
        gt_traj_data=gt_traj_data,
    )
    goal_pos = goal_dict["pos"]

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

    # Check if time step is valid
    if t < 0 or t >= len(gt_traj) - 1:
        return (0.0, 0.0)

    # Check if vehicle exists in current and next time step.
    # In replay mode, opponent may not control this vehicle and adapter can return None;
    # fall back to GT existence in that case.
    # gt trajectory: [pos_x, pos_y, heading, speed, existence, goal_x, goal_y, length]
    target = gt_action_target_cache.get(veh_id)
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

    next_pos, next_heading, next_speed, wheel_base = resolve_next_gt_state(
        gt_traj,
        gt_action_target_cache,
        veh_id,
        t,
    )
    action = reconstruct_gt_action(
        pos_x=float(pos.x),
        pos_y=float(pos.y),
        heading=heading,
        speed=speed,
        next_pos=next_pos,
        next_heading=next_heading,
        next_speed=next_speed,
        wheel_base=wheel_base,
        dt=env.dt,
    )
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
