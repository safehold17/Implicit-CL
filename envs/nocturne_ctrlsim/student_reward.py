"""
Student reward computation for Nocturne CtrlSim adversarial env.
"""

import os
import sys
import numpy as np


_CTRLSIM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "third_party",
    "ctrl-sim",
)
if _CTRLSIM_PATH not in sys.path:
    sys.path.insert(0, _CTRLSIM_PATH)
from utils.sim import compute_reward


def _angle_diff(a: float, b: float) -> float:
    """Calculate the difference between two angles (handle wraparound)."""
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def compute_student_reward(env) -> float:
    """
    Compute student reward using CtrlSim's compute_reward function.

    Uses CtrlSim compute_reward() to get reward components, then aggregates them
    using target-term style:
      - goal position: position_target_achieved * pos_target_achieved_rew_multiplier
      + pos_goal_shaped (only when use_pos_shaped=True)
      - goal heading: heading_target_achieved + heading_goal_shaped
      - goal speed: speed_target_achieved + speed_goal_shaped
    Collision penalties keep existing multipliers from env config.
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
        'shaped_goal_distance': getattr(env, 'shaped_goal_distance', True),
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
    position_achieved_current = dist_to_goal < 1.0  # position_target_tolerance
    speed_achieved_current = abs(ego_speed - goal_speed) < 1.0  # speed_target_tolerance
    heading_achieved_current = abs(_angle_diff(ego_heading, goal_heading)) < 0.3  # heading_target_tolerance

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
    # goal_pos = achieved * multiplier (+ pos_shaped if enabled)
    # goal_heading = achieved + shaped
    # goal_speed = achieved + shaped
    pos_shaped_term = pos_shaped if getattr(env, 'use_pos_shaped', False) else 0.0

    scalar_reward = (
        position_target_achieved * env.pos_target_achieved_rew_multiplier
        + pos_shaped_term
        + heading_target_achieved
        + heading_shaped
        + speed_target_achieved
        + speed_shaped
    )
    scalar_reward += -veh_veh_collision * env.veh_veh_collision_rew_multiplier
    scalar_reward += -veh_edge_collision * env.veh_edge_collision_rew_multiplier

    env.episode_reward += scalar_reward
    return scalar_reward
