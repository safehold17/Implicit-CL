"""Student-policy related helper functions for Nocturne CtrlSim adversarial env."""

from dataclasses import dataclass
from typing import List

import heapq
import math
import numpy as np

from .utils.common import angle_of_rotation, angle_sub, to_local

EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_FEATURE_DIM = 13
ROAD_TYPE_DIM = 7
ROAD_TYPE_MAPPING = {
    "none": 0,
    "road_line": 1,
    "road_edge": 2,
    "lane": 3,
    "crosswalk": 4,
    "speed_bump": 5,
    "stop_sign": 6,
    "other": 0,
}
STATIC_ROAD_TYPE_MAPPING = {
    "stop_sign": 6,
    "crosswalk": 4,
    "speed_bump": 5,
}
ROAD_TYPE_ONEHOT = np.eye(ROAD_TYPE_DIM, dtype=np.float32)


@dataclass(frozen=True)
class StudentObservationConfig:
    """Centralized student observation layout."""

    max_neighbors: int = 16
    top_k_road_points: int = 64
    ego_feat_dim: int = EGO_FEAT_DIM
    partner_feat_dim: int = PARTNER_FEAT_DIM
    road_graph_feat_dim: int = ROAD_FEATURE_DIM

    @property
    def max_controlled_agents(self) -> int:
        return self.max_neighbors + 1


def build_student_observation_config(
    max_neighbors: int = 16,
    top_k_road_points: int = 64,
) -> StudentObservationConfig:
    """Build the canonical student observation config."""
    return StudentObservationConfig(
        max_neighbors=int(max_neighbors),
        top_k_road_points=int(top_k_road_points),
    )


def get_student_obs_dim(config: StudentObservationConfig) -> int:
    """Return the flattened student observation dimension."""
    return (
        config.ego_feat_dim
        + config.max_neighbors * config.partner_feat_dim
        + config.top_k_road_points * config.road_graph_feat_dim
    )


def get_env_student_observation_config(env) -> StudentObservationConfig:
    """Build the student observation config from env state."""
    return build_student_observation_config(
        max_neighbors=getattr(env, "_max_observable_agents", 16),
        top_k_road_points=getattr(env, "_top_k_road_points", 64),
    )


def split_student_observation(obs_flat, config: StudentObservationConfig):
    """Split flattened observations using the centralized observation spec."""
    expected_obs_dim = get_student_obs_dim(config)
    actual_obs_dim = int(obs_flat.shape[-1])
    if actual_obs_dim != expected_obs_dim:
        raise ValueError(
            "LateFusion observation dimension mismatch: "
            f"expected {expected_obs_dim} = "
            f"{config.ego_feat_dim} + {config.max_neighbors}*{config.partner_feat_dim} + "
            f"{config.top_k_road_points}*{config.road_graph_feat_dim}, "
            f"got {actual_obs_dim}. "
            "Please keep student observation config consistent between env and model."
        )

    ego_end = config.ego_feat_dim
    partner_end = ego_end + config.max_neighbors * config.partner_feat_dim

    ego_state = obs_flat[:, :ego_end]
    partner_obs = obs_flat[:, ego_end:partner_end]
    road_graph_obs = obs_flat[:, partner_end:]

    road_objects = partner_obs.reshape(
        -1, config.max_neighbors, config.partner_feat_dim
    )
    road_graph = road_graph_obs.reshape(
        -1, config.top_k_road_points, config.road_graph_feat_dim
    )
    return ego_state, road_objects, road_graph


def apply_student_action(env, action: np.ndarray) -> None:
    """
    Apply student action to ego vehicle.

    Args:
        action: Discrete student action id.
    """
    if env.ego_vehicle is None:
        return

    accel_bins = int(env.student_accel_discretization)
    steer_bins = int(env.student_steer_discretization)
    num_actions = int(accel_bins * steer_bins)

    action_id = int(np.asarray(action).reshape(-1)[0])
    action_id = int(np.clip(action_id, 0, num_actions - 1))

    accel_idx = action_id // steer_bins
    steer_idx = action_id % steer_bins

    accel_norm = (2.0 * accel_idx / (accel_bins - 1)) - 1.0
    steer_norm = (2.0 * steer_idx / (steer_bins - 1)) - 1.0

    # Convert normalized action to actual values
    accel = accel_norm * 10.0  # max acc 10 m/s^2
    steer = steer_norm * 0.7  # max steer 0.7 rad

    if accel > 0:
        env.ego_vehicle.acceleration = accel
    else:
        env.ego_vehicle.brake(abs(accel))
    env.ego_vehicle.steering = steer


def _build_road_feature(
    rel_x: float,
    rel_y: float,
    seg_length: float,
    orientation: float,
    type_idx: int,
) -> np.ndarray:
    road_feat = np.empty(ROAD_FEATURE_DIM, dtype=np.float32)
    road_feat[0] = rel_x
    road_feat[1] = rel_y
    road_feat[2] = seg_length
    road_feat[3] = 1.0
    road_feat[4] = 1.0
    road_feat[5] = orientation
    road_feat[6:] = ROAD_TYPE_ONEHOT[type_idx]
    return road_feat


def build_road_graph_obs(env, ego_pos, ego_heading: float) -> List[np.ndarray]:
    """
    Build Road Graph observation (in gpudrive).

    Returns:
        road_graph_states: List of road point features (R 13-dimensional vectors)
    """
    top_k = env._top_k_road_points
    if top_k <= 0:
        return []

    if env._road_graph_cache is None or len(env._road_graph_cache) == 0:
        # No road data, return empty road graph
        return [np.zeros(ROAD_FEATURE_DIM, dtype=np.float32) for _ in range(top_k)]

    angle = angle_of_rotation(ego_heading)

    # Keep only nearest top_k points while scanning all road geometries.
    # Heap item: (-dist_sq, -point_index, rel_x, rel_y, seg_length, orientation, type_idx)
    topk_heap = []
    point_index = 0

    for road_item in env._road_graph_cache:
        road_type = road_item["type"]
        geometry = road_item["geometry"]

        # Process different types of geometry data
        if isinstance(geometry, list) and len(geometry) > 0:
            # Road line (multiple points)
            type_idx = ROAD_TYPE_MAPPING.get(road_type, 0)
            last_idx = len(geometry) - 1
            for i, pt in enumerate(geometry):
                # Relative position
                dx = pt["x"] - ego_pos.x
                dy = pt["y"] - ego_pos.y
                rel_x, rel_y = to_local(dx, dy, angle)
                dist_sq = rel_x * rel_x + rel_y * rel_y

                if len(topk_heap) == top_k and dist_sq >= -topk_heap[0][0]:
                    point_index += 1
                    continue

                # Calculate road segment length
                if i < last_idx:
                    next_pt = geometry[i + 1]
                    seg_length = math.sqrt(
                        (next_pt["x"] - pt["x"]) ** 2 + (next_pt["y"] - pt["y"]) ** 2
                    )
                    # Direction: points to next point
                    orientation = math.atan2(
                        next_pt["y"] - pt["y"], next_pt["x"] - pt["x"]
                    )
                else:
                    seg_length = 1.0  # Default value
                    orientation = 0.0
                orientation = angle_sub(orientation, -angle)

                heap_item = (
                    -dist_sq,
                    -point_index,
                    rel_x,
                    rel_y,
                    seg_length,
                    orientation,
                    type_idx,
                )
                if len(topk_heap) < top_k:
                    heapq.heappush(topk_heap, heap_item)
                else:
                    heapq.heapreplace(topk_heap, heap_item)
                point_index += 1

        elif isinstance(geometry, dict):
            # Static object (e.g. stop_sign)
            dx = geometry["x"] - ego_pos.x
            dy = geometry["y"] - ego_pos.y
            rel_x, rel_y = to_local(dx, dy, angle)
            dist_sq = rel_x * rel_x + rel_y * rel_y

            if len(topk_heap) == top_k and dist_sq >= -topk_heap[0][0]:
                point_index += 1
                continue

            type_idx = STATIC_ROAD_TYPE_MAPPING.get(road_type, 0)
            heap_item = (
                -dist_sq,
                -point_index,
                rel_x,
                rel_y,
                0.0,
                0.0,
                type_idx,
            )
            if len(topk_heap) < top_k:
                heapq.heappush(topk_heap, heap_item)
            else:
                heapq.heapreplace(topk_heap, heap_item)
            point_index += 1

    # Convert selected top_k points to features, ordered by distance asc.
    selected_points = sorted(topk_heap, key=lambda item: (-item[0], -item[1]))
    road_graph_states = [
        _build_road_feature(
            rel_x=item[2],
            rel_y=item[3],
            seg_length=item[4],
            orientation=item[5],
            type_idx=item[6],
        )
        for item in selected_points
    ]

    # Fill missing road points
    for _ in range(top_k - len(road_graph_states)):
        road_graph_states.append(np.zeros(ROAD_FEATURE_DIM, dtype=np.float32))

    return road_graph_states


def get_student_observation(env) -> np.ndarray:
    """Get student policy observation (consistent with gpudrive)."""
    obs_dim = get_student_obs_dim(get_env_student_observation_config(env))
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return np.zeros(obs_dim, dtype=np.float32)

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
    if len(obs_concat) < obs_dim:
        obs_final = np.zeros(obs_dim, dtype=np.float32)
        obs_final[: len(obs_concat)] = obs_concat
    else:
        obs_final = obs_concat[:obs_dim]

    return obs_final
