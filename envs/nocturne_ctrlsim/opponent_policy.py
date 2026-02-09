"""
Opponent-policy related helper functions for Nocturne CtrlSim adversarial env.
"""

from typing import List, Optional, Tuple

import numpy as np

import math

from tools.safe_bicycle import safe_backward_action_from_states

from .scenario_helpers import get_vehicle_by_id



def angle_of_rotation(yaw: float) -> float:
    return (math.pi / 2.0) - float(yaw)


def angle_sub(current_angle: float, target_angle: float) -> float:
    diff = (target_angle - current_angle) % (2 * math.pi)
    if diff > math.pi:
        diff = -(2 * math.pi - diff)
    return diff


def to_local(dx: float, dy: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    return local_x, local_y


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

    gt_traj_data = np.array(env._gt_data_dict[ego_id]['traj'])

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

    gt_traj_data = np.array(env._gt_data_dict[veh_id]['traj'])
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

    gt_traj = np.array(env._gt_data_dict[veh_id]['traj'])

    # Check if time step is valid
    if t < 0 or t >= len(gt_traj) - 1:
        return (0.0, 0.0)

    # Check if vehicle exists in current and next time step
    if veh_id in env.opponent_vehicle_ids and env.opponent is not None:
        exists = env.opponent.get_opponent_vehicle_exists(veh_id)
        veh_exists = 1 if exists else 0
    else:
        veh_exists = gt_traj[t, 4] and gt_traj[t + 1, 4]
    # Once missing, remain missing (align ctrl-sim evaluator)
    ego_data = env.opponent.get_vehicle_data(veh_id) if env.opponent else None
    if t > 0 and ego_data and ego_data["existence"][-1] == 0:
        veh_exists = 0

    if not veh_exists or veh is None:
        return (0.0, 0.0)

    accel, steer = safe_backward_action_from_states(
        prev_pos=(veh.getPosition().x, veh.getPosition().y),
        prev_theta=veh.getHeading(),
        prev_vel=veh.getSpeed(),
        curr_pos=(gt_traj[t + 1, 0], gt_traj[t + 1, 1]),
        curr_theta=gt_traj[t + 1, 2],
        curr_vel=gt_traj[t + 1, 3],
        wheel_base=gt_traj[t + 1, -1],
        dt=env.dt,
    )

    return (float(accel), float(steer))



def apply_student_action(env, action: np.ndarray) -> None:
    """
    Apply student action to ego vehicle.

    Args:
        action: [acceleration, steering] normalized to [-1, 1]
    """
    if env.ego_vehicle is None:
        return

    # Convert normalized action to actual values
    accel = action[0] * 10.0  # max acc 10 m/s²
    steer = action[1] * 0.7  # max steer 0.7 rad

    if accel > 0:
        env.ego_vehicle.acceleration = accel
    else:
        env.ego_vehicle.brake(abs(accel))
    env.ego_vehicle.steering = steer



def build_road_graph_obs(env, ego_pos, ego_heading: float) -> List[np.ndarray]:
    """
    Build Road Graph observation (in gpudrive).

    Returns:
        road_graph_states: List of road point features (R 13-dimensional vectors)
    """
    if env._road_graph_cache is None or len(env._road_graph_cache) == 0:
        # No road data, return empty road graph
        return [np.zeros(13, dtype=np.float32) for _ in range(env._top_k_road_points)]

    angle = angle_of_rotation(ego_heading)

    # Extract road point features
    road_points = []

    for road_item in env._road_graph_cache:
        road_type = road_item['type']
        geometry = road_item['geometry']

        # Process different types of geometry data
        if isinstance(geometry, list) and len(geometry) > 0:
            # Road line (multiple points)
            for i, pt in enumerate(geometry):
                # Relative position
                dx = pt['x'] - ego_pos.x
                dy = pt['y'] - ego_pos.y
                rel_x, rel_y = to_local(dx, dy, angle)

                # Calculate road segment length
                if i < len(geometry) - 1:
                    next_pt = geometry[i + 1]
                    seg_length = np.sqrt(
                        (next_pt['x'] - pt['x'])**2 +
                        (next_pt['y'] - pt['y'])**2
                    )
                    # Direction: points to next point
                    orientation = np.arctan2(
                        next_pt['y'] - pt['y'],
                        next_pt['x'] - pt['x']
                    )
                else:
                    seg_length = 1.0  # Default value
                    orientation = 0.0
                orientation = angle_sub(orientation, -angle)

                # Road point scale (default value)
                scale_x = 1.0
                scale_y = 1.0

                # Road type one-hot (7 dimensions)
                type_mapping = {
                    'none': 0,
                    'road_line': 1,
                    'road_edge': 2,
                    'lane': 3,
                    'crosswalk': 4,
                    'speed_bump': 5,
                    'stop_sign': 6,
                    'other': 0,
                }
                type_idx = type_mapping.get(road_type, 0)
                type_onehot = np.zeros(7, dtype=np.float32)
                type_onehot[type_idx] = 1.0

                # Concatenate features (13 dimensions)
                road_feat = np.array([
                    rel_x,
                    rel_y,
                    seg_length,
                    scale_x,
                    scale_y,
                    orientation,
                    *type_onehot,
                ], dtype=np.float32)

                road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))

        elif isinstance(geometry, dict):
            # Static object (e.g. stop_sign)
            dx = geometry['x'] - ego_pos.x
            dy = geometry['y'] - ego_pos.y
            rel_x, rel_y = to_local(dx, dy, angle)

            type_mapping = {
                'stop_sign': 6,
                'crosswalk': 4,
                'speed_bump': 5,
            }
            type_idx = type_mapping.get(road_type, 0)
            type_onehot = np.zeros(7, dtype=np.float32)
            type_onehot[type_idx] = 1.0

            road_feat = np.array([
                rel_x,
                rel_y,
                0.0,  # length
                1.0,  # scale_x
                1.0,  # scale_y
                0.0,  # orientation
                *type_onehot,
            ], dtype=np.float32)

            road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))

    # Sort by distance, select top_k nearest points
    road_points.sort(key=lambda x: x[0])

    road_graph_states = []
    num_valid_points = min(len(road_points), env._top_k_road_points)

    for i in range(num_valid_points):
        road_graph_states.append(road_points[i][1])

    # Fill missing road points
    for _ in range(env._top_k_road_points - num_valid_points):
        road_graph_states.append(np.zeros(13, dtype=np.float32))

    return road_graph_states



def get_student_observation(env) -> np.ndarray:
    """Get student policy observation (consistent with gpudrive)."""
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return np.zeros(env._obs_dim, dtype=np.float32)

    # ========== Ego state (6 dimensions) ==========
    ego_pos = env.ego_vehicle.getPosition()
    ego_heading = env.ego_vehicle.getHeading()
    ego_speed = env.ego_vehicle.getSpeed()

    # Relative target position (in ego coordinate system)
    goal_pos = env._ego_goal_dict['pos']
    angle = angle_of_rotation(ego_heading)
    rel_goal_x, rel_goal_y = to_local(
        goal_pos[0] - ego_pos.x,
        goal_pos[1] - ego_pos.y,
        angle,
    )

    # Collision state
    collision_state = 1.0 if env._collision_occurred else 0.0

    ego_state = np.array([
        ego_speed,
        env.ego_vehicle.getLength(),
        env.ego_vehicle.getWidth(),
        rel_goal_x,
        rel_goal_y,
        collision_state,
    ], dtype=np.float32)

    # ========== Partner state (K*6 dimensions) ==========
    max_neighbors = getattr(env, '_max_observable_agents', 16)
    partner_states = []

    # Select nearest K neighboring vehicles
    num_neighbors = min(len(env.opponent_vehicles), max_neighbors)

    for i in range(num_neighbors):
        veh = env.opponent_vehicles[i]
        veh_pos = veh.getPosition()

        # Relative position to ego
        rel_pos_x, rel_pos_y = to_local(
            veh_pos.x - ego_pos.x,
            veh_pos.y - ego_pos.y,
            angle,
        )

        # Relative orientation
        rel_orientation = angle_sub(veh.getHeading(), -angle)

        partner_state = np.array([
            veh.getSpeed(),
            rel_pos_x,
            rel_pos_y,
            rel_orientation,
            veh.getLength(),
            veh.getWidth(),
        ], dtype=np.float32)
        partner_states.append(partner_state)

    # Fill missing neighbors with zero vector
    for _ in range(max_neighbors - num_neighbors):
        partner_states.append(np.zeros(6, dtype=np.float32))

    # ========== Road Graph (R*13 dimensions) ==========
    road_graph_states = build_road_graph_obs(env, ego_pos, ego_heading)

    # ========== Concatenate all observations ==========
    obs_parts = [ego_state]
    obs_parts.extend(partner_states)
    obs_parts.extend(road_graph_states)

    obs_concat = np.concatenate(obs_parts)

    # Fill or truncate to obs_dim
    if len(obs_concat) < env._obs_dim:
        obs_final = np.zeros(env._obs_dim, dtype=np.float32)
        obs_final[:len(obs_concat)] = obs_concat
    else:
        obs_final = obs_concat[:env._obs_dim]

    return obs_final



def angle_diff(env, a: float, b: float) -> float:
    """Calculate the difference between two angles (handle wraparound)."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff



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



def compute_reward(env) -> float:
    """
    Compute student reward using CtrlSim's compute_reward function.

    Uses the exact same reward calculation as CtrlSim opponents to ensure consistency.
    Returns a scalar reward by summing the shaped reward components and applying
    collision penalties.
    """
    import nocturne

    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return 0.0

    ego_id = env.ego_vehicle.getID()

    # Import compute_reward from ctrl-sim (same as opponent)
    import os
    import sys

    _CTRLSIM_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'third_party',
        'ctrl-sim',
    )
    if _CTRLSIM_PATH not in sys.path:
        sys.path.insert(0, _CTRLSIM_PATH)
    from utils.sim import compute_reward

    # Use CtrlSim's reward config
    rew_cfg = {
        'position_target': True,
        'position_target_tolerance': 1.0,
        'speed_target': True,
        'speed_target_tolerance': 1.0,
        'heading_target': True,
        'heading_target_tolerance': 0.3,
        'shaped_goal_distance': True,
        'shaped_goal_distance_scaling': 0.2,
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
    position_achieved_current = dist_to_goal < 1.0  # position_target_tolerance
    speed_achieved_current = abs(ego_speed - goal_speed) < 1.0  # speed_target_tolerance
    heading_achieved_current = abs(angle_diff(env, ego_heading, goal_heading)) < 0.3  # heading_target_tolerance

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

    # Convert to scalar: sum shaped rewards + collision penalties
    scalar_reward = pos_shaped + speed_shaped + heading_shaped
    scalar_reward += -veh_veh_collision * env.veh_veh_collision_rew_multiplier
    scalar_reward += -veh_edge_collision * env.veh_edge_collision_rew_multiplier

    env.episode_reward += scalar_reward
    return scalar_reward
