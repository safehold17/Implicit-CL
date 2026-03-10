"""
Student reward computation for Nocturne CtrlSim adversarial env.
"""

import numpy as np

from ctrlsim_adapter.ctrlsim_path import ctrlsim_path
from .utils.common import is_valid_world_position

ctrlsim_path()

from utils.sim import compute_reward
from utils.data import compute_distance_to_road_edge


def _angle_diff(a: float, b: float) -> float:
    """Calculate the difference between two angles (handle wraparound)."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def _compute_veh_veh_shaped_reward(env, ego_id: int, ego_pos_arr: np.ndarray) -> float:
    """Compute CtrlSim-style veh-veh shaped reward in [0, 1]."""
    if not getattr(env, 'use_veh_veh_shaped', True):
        return 0.0

    max_veh_veh_distance = float(getattr(env, 'max_veh_veh_distance', 15.0))
    if max_veh_veh_distance <= 0.0:
        return 0.0

    nearest_dist = np.inf
    for veh in getattr(env, 'vehicles', []):
        if veh.getID() == ego_id:
            continue
        if not getattr(veh, 'physics_simulated', True):
            continue
        pos = veh.getPosition()
        if not is_valid_world_position(pos.x, pos.y):
            continue
        dist = float(np.linalg.norm(ego_pos_arr - np.array([pos.x, pos.y])))
        if dist < nearest_dist:
            nearest_dist = dist

    if not np.isfinite(nearest_dist):
        return 0.0
    return float(np.clip(nearest_dist, 0.0, max_veh_veh_distance) / max_veh_veh_distance)


def _extract_road_edge_polylines(env) -> list[np.ndarray]:
    """Extract road-edge polylines from cached road graph."""
    roads_data = getattr(env, '_road_graph_cache', None)
    if not roads_data:
        return []

    polylines = []
    for road in roads_data:
        if road.get('type') != 'road_edge':
            continue
        geometry = road.get('geometry')
        if not isinstance(geometry, list) or len(geometry) < 2:
            continue
        polyline = np.array([[pt['x'], pt['y']] for pt in geometry], dtype=np.float32)
        polylines.append(polyline)
    return polylines


def _compute_veh_edge_shaped_reward(env, ego_pos_arr: np.ndarray) -> float:
    """Compute CtrlSim-style veh-edge shaped reward in [0, 1]."""
    if not getattr(env, 'use_veh_edge_shaped', True):
        return 0.0

    clip_distance = float(getattr(env, 'veh_edge_reward_distance_clip', 5.0))
    if clip_distance <= 0.0:
        return 0.0

    road_edge_polylines = _extract_road_edge_polylines(env)
    if len(road_edge_polylines) == 0:
        return 0.0

    center_x = np.array([[ego_pos_arr[0]]], dtype=np.float32)
    center_y = np.array([[ego_pos_arr[1]]], dtype=np.float32)
    signed_distance = compute_distance_to_road_edge(
        center_x=center_x,
        center_y=center_y,
        road_edge_polylines=road_edge_polylines,
    )
    if signed_distance is None or len(signed_distance) == 0:
        return 0.0

    edge_distance = float(np.abs(signed_distance[0]))
    return float(np.clip(edge_distance, 0.0, clip_distance) / clip_distance)


def compute_student_reward(env) -> float:
    """
    Compute student reward using CtrlSim's compute_reward function.

    Uses CtrlSim compute_reward() to get reward components, then aggregates them
    using target-term style:
      - goal position: position_target_achieved * pos_target_achieved_rew_multiplier
      + pos_goal_shaped (only when use_pos_shaped=True)
      + approaching_goal (only when use_approaching_goal=True)
      - goal heading: heading_target_achieved + heading_goal_shaped
      - goal speed: speed_target_achieved + speed_goal_shaped
      - veh-veh: veh_veh_shaped - veh_veh_collision * veh_veh_collision_rew_multiplier
      - veh-edge: veh_edge_shaped - veh_edge_collision * veh_edge_collision_rew_multiplier
    """
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return 0.0

    ego_id = env.ego_vehicle.getID()

    # Use CtrlSim's reward config
    rew_cfg = {
        'position_target': True,
        'position_target_tolerance': 1.0,
        'speed_target': True,
        'speed_target_tolerance': 1.0,
        'heading_target': True,
        'heading_target_tolerance': 0.3,
        'shaped_goal_distance': getattr(env, 'shaped_goal_reward', True),
        'shaped_goal_distance_scaling': getattr(
            env, 'shaped_goal_distance_scaling', 0.2
        ),
        'reward_scaling': 1.0,
    }

    reward_vector = compute_reward(
        rew_cfg,
        env.ego_vehicle,
        env._ego_goal_dict,
        env._ego_goal_dist_normalizer,
        env._ego_vehicle_data_dict,
        collision_fix=True,
    )

    # Extract components
    position_target_achieved = reward_vector[0]
    heading_target_achieved = reward_vector[1]
    speed_target_achieved = reward_vector[2]
    pos_shaped = reward_vector[3]
    speed_shaped = reward_vector[4]
    heading_shaped = reward_vector[5]
    veh_veh_collision = reward_vector[6]
    veh_edge_collision = reward_vector[7]

    # Check goal achieved using current state (not CtrlSim's persistent logic)
    ego_pos = env.ego_vehicle.getPosition()
    ego_pos_arr = np.array([ego_pos.x, ego_pos.y])
    ego_speed = env.ego_vehicle.getSpeed()
    ego_heading = env.ego_vehicle.getHeading()

    goal_pos = env._ego_goal_dict['pos']
    goal_speed = env._ego_goal_dict['speed']
    goal_heading = env._ego_goal_dict['heading']

    dist_to_goal = np.linalg.norm(goal_pos - ego_pos_arr)

    # best_goal_distance: the closest distance to the goal so far.
    best_goal_distance = float(getattr(env, '_best_goal_distance', np.inf))
    # Keep state in reward logic: initialize once per episode (or when state missing).
    if getattr(env, 'current_step', 0) <= 1 or not np.isfinite(best_goal_distance):
        best_goal_distance = float(dist_to_goal)
    goal_improve = max(0.0, best_goal_distance - float(dist_to_goal))
    env._best_goal_distance = min(best_goal_distance, float(dist_to_goal))

    position_achieved_current = dist_to_goal < 1.0  # position_target_tolerance
    speed_achieved_current = abs(ego_speed - goal_speed) < 1.0  # speed_target_tolerance
    heading_achieved_current = abs(_angle_diff(ego_heading, goal_heading)) < 0.3  # heading_target_tolerance
    reward_history = []
    if ego_id in env._ego_vehicle_data_dict:
        reward_history = env._ego_vehicle_data_dict[ego_id].get('reward', [])

    # Give position bonus only on the first timestep that enters position tolerance.
    already_position_hit = bool(reward_history) and bool(reward_history[-1][0])
    position_reward_once = (
        env.pos_target_achieved_rew_multiplier
        if position_achieved_current and not already_position_hit
        else 0.0
    )
    use_persistent_position_reward = getattr(
        env, 'use_persistent_position_reward', False
    )
    position_reward_term = (
        position_target_achieved * env.pos_target_achieved_rew_multiplier
        if use_persistent_position_reward
        else position_reward_once
    )

    # Update goal state: once reached, stay reached (same as before)
    if env._goal_reached:
        pass  # Keep achieved state
    elif position_achieved_current and speed_achieved_current and heading_achieved_current:
        env._goal_reached = True

    # Update collision states
    if veh_veh_collision:
        env._collision_occurred = True

    if veh_edge_collision:
        env._offroad_occurred = True

    # Store reward vector in vehicle_data_dict (for compute_reward's history check)
    if ego_id in env._ego_vehicle_data_dict:
        env._ego_vehicle_data_dict[ego_id]['reward'].append(reward_vector)

    # Target-term aggregation:
    # goal_pos = persistent or one-time bonus (+ pos_shaped if enabled)
    # goal_heading = achieved + shaped
    # goal_speed = achieved + shaped
    pos_shaped_term = pos_shaped if getattr(env, 'use_pos_shaped', False) else 0.0
    use_approaching_goal = getattr(env, 'use_approaching_goal', True)
    approaching_goal_scaling = getattr(env, 'approaching_goal_scaling', 1.0)
    approaching_goal_term = (
        approaching_goal_scaling * goal_improve if use_approaching_goal else 0.0
    )
    use_speed_shaped = getattr(env, 'use_speed_shaped', True)
    use_heading_shaped = getattr(env, 'use_heading_shaped', True)
    use_speed_heading_target = getattr(env, 'use_speed_heading_target', True)
    heading_target_term = heading_target_achieved if use_speed_heading_target else 0.0
    speed_target_term = speed_target_achieved if use_speed_heading_target else 0.0
    speed_shaped_term = speed_shaped if use_speed_shaped else 0.0
    heading_shaped_term = heading_shaped if use_heading_shaped else 0.0
    veh_veh_shaped_term = _compute_veh_veh_shaped_reward(env, ego_id, ego_pos_arr)  # [0, 1]
    veh_edge_shaped_term = _compute_veh_edge_shaped_reward(env, ego_pos_arr)  #[0, 1]

    scalar_reward = (
            position_reward_term
            + pos_shaped_term          # default once reward
            + approaching_goal_term
            + heading_target_term      # not using
            + heading_shaped_term      # not using
            + speed_target_term        # not using
            + speed_shaped_term        # not using
            + veh_veh_shaped_term      # not using
            + veh_edge_shaped_term     # not using
            - veh_veh_collision * env.veh_veh_collision_rew_multiplier
            - veh_edge_collision * env.veh_edge_collision_rew_multiplier
    )

    env.episode_reward += scalar_reward
    return scalar_reward
