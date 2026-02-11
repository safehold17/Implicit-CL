"""Student-policy related helper functions for Nocturne CtrlSim adversarial env."""

from typing import List

import math
import numpy as np


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


def apply_student_action(env, action: np.ndarray) -> None:
    """
    Apply student action to ego vehicle.

    Args:
        action: [acceleration, steering] normalized to [-1, 1]
    """
    if env.ego_vehicle is None:
        return

    # Convert normalized action to actual values
    accel = action[0] * 10.0  # max acc 10 m/s^2
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
        road_type = road_item["type"]
        geometry = road_item["geometry"]

        # Process different types of geometry data
        if isinstance(geometry, list) and len(geometry) > 0:
            # Road line (multiple points)
            for i, pt in enumerate(geometry):
                # Relative position
                dx = pt["x"] - ego_pos.x
                dy = pt["y"] - ego_pos.y
                rel_x, rel_y = to_local(dx, dy, angle)

                # Calculate road segment length
                if i < len(geometry) - 1:
                    next_pt = geometry[i + 1]
                    seg_length = np.sqrt(
                        (next_pt["x"] - pt["x"]) ** 2 + (next_pt["y"] - pt["y"]) ** 2
                    )
                    # Direction: points to next point
                    orientation = np.arctan2(
                        next_pt["y"] - pt["y"], next_pt["x"] - pt["x"]
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
                    "none": 0,
                    "road_line": 1,
                    "road_edge": 2,
                    "lane": 3,
                    "crosswalk": 4,
                    "speed_bump": 5,
                    "stop_sign": 6,
                    "other": 0,
                }
                type_idx = type_mapping.get(road_type, 0)
                type_onehot = np.zeros(7, dtype=np.float32)
                type_onehot[type_idx] = 1.0

                # Concatenate features (13 dimensions)
                road_feat = np.array(
                    [
                        rel_x,
                        rel_y,
                        seg_length,
                        scale_x,
                        scale_y,
                        orientation,
                        *type_onehot,
                    ],
                    dtype=np.float32,
                )

                road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))

        elif isinstance(geometry, dict):
            # Static object (e.g. stop_sign)
            dx = geometry["x"] - ego_pos.x
            dy = geometry["y"] - ego_pos.y
            rel_x, rel_y = to_local(dx, dy, angle)

            type_mapping = {
                "stop_sign": 6,
                "crosswalk": 4,
                "speed_bump": 5,
            }
            type_idx = type_mapping.get(road_type, 0)
            type_onehot = np.zeros(7, dtype=np.float32)
            type_onehot[type_idx] = 1.0

            road_feat = np.array(
                [
                    rel_x,
                    rel_y,
                    0.0,  # length
                    1.0,  # scale_x
                    1.0,  # scale_y
                    0.0,  # orientation
                    *type_onehot,
                ],
                dtype=np.float32,
            )

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
    goal_pos = env._ego_goal_dict["pos"]
    angle = angle_of_rotation(ego_heading)
    rel_goal_x, rel_goal_y = to_local(
        goal_pos[0] - ego_pos.x,
        goal_pos[1] - ego_pos.y,
        angle,
    )

    # Collision state
    collision_state = 1.0 if env._collision_occurred else 0.0

    ego_state = np.array(
        [
            ego_speed,
            env.ego_vehicle.getLength(),
            env.ego_vehicle.getWidth(),
            rel_goal_x,
            rel_goal_y,
            collision_state,
        ],
        dtype=np.float32,
    )

    # ========== Partner state (K*6 dimensions) ==========
    max_neighbors = getattr(env, "_max_observable_agents", 16)
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

        partner_state = np.array(
            [
                veh.getSpeed(),
                rel_pos_x,
                rel_pos_y,
                rel_orientation,
                veh.getLength(),
                veh.getWidth(),
            ],
            dtype=np.float32,
        )
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
        obs_final[: len(obs_concat)] = obs_concat
    else:
        obs_final = obs_concat[: env._obs_dim]

    return obs_final
