"""Student observation and action helpers for the Nocturne CtrlSim environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np

from ..utils.common import (
    angle_of_rotation,
    is_valid_world_position,
    to_local,
)

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
        """Return the total number of controlled agents including ego."""
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


def apply_student_action(env, action: np.ndarray):
    """Apply a discrete student action to the ego vehicle."""
    if env.ego_vehicle is None:
        return None

    accel_bins = int(env.student_accel_discretization)
    steer_bins = int(env.student_steer_discretization)
    num_actions = int(accel_bins * steer_bins)

    action_id = int(np.asarray(action).reshape(-1)[0])
    action_id = int(np.clip(action_id, 0, num_actions - 1))

    accel_idx = action_id // steer_bins
    steer_idx = action_id % steer_bins

    accel_norm = (2.0 * accel_idx / (accel_bins - 1)) - 1.0
    steer_norm = (2.0 * steer_idx / (steer_bins - 1)) - 1.0

    accel = accel_norm * 10.0
    steer = steer_norm * 0.7

    if accel > 0:
        env.ego_vehicle.acceleration = accel
    else:
        env.ego_vehicle.brake(abs(accel))
    env.ego_vehicle.steering = steer
    return float(accel), float(steer)


def _to_local_array(
    dx: np.ndarray,
    dy: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-frame offsets into ego-local offsets in batch."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    return local_x.astype(np.float32), local_y.astype(np.float32)


def _normalize_angle_to_target(
    current_angles: np.ndarray,
    target_angle: float,
) -> np.ndarray:
    """Return wrapped angle deltas from current angles to a shared target."""
    diff = (float(target_angle) - current_angles + np.pi) % (2.0 * np.pi) - np.pi
    return diff.astype(np.float32)


def build_student_road_cache(
    road_graph_data: list[dict[str, Any]] | None,
) -> dict[str, np.ndarray] | None:
    """Flatten static road graph data into arrays reused by every step."""
    if not road_graph_data:
        return None

    points: list[tuple[float, float]] = []
    segment_lengths: list[float] = []
    orientations: list[float] = []
    type_indices: list[int] = []

    for road_item in road_graph_data:
        road_type = road_item["type"]
        geometry = road_item["geometry"]

        if isinstance(geometry, list) and geometry:
            type_idx = ROAD_TYPE_MAPPING.get(road_type, 0)
            last_idx = len(geometry) - 1
            for point_idx, point in enumerate(geometry):
                points.append((point["x"], point["y"]))
                if point_idx < last_idx:
                    next_point = geometry[point_idx + 1]
                    dx = next_point["x"] - point["x"]
                    dy = next_point["y"] - point["y"]
                    segment_lengths.append(math.hypot(dx, dy))
                    orientations.append(math.atan2(dy, dx))
                else:
                    segment_lengths.append(1.0)
                    orientations.append(0.0)
                type_indices.append(type_idx)
            continue

        if isinstance(geometry, dict):
            points.append((geometry["x"], geometry["y"]))
            segment_lengths.append(0.0)
            orientations.append(0.0)
            type_indices.append(STATIC_ROAD_TYPE_MAPPING.get(road_type, 0))

    if not points:
        return None

    points_array = np.asarray(points, dtype=np.float32)
    type_indices_array = np.asarray(type_indices, dtype=np.int64)
    return {
        "points": points_array,
        "segment_lengths": np.asarray(segment_lengths, dtype=np.float32),
        "orientations": np.asarray(orientations, dtype=np.float32),
        "type_onehot": ROAD_TYPE_ONEHOT[type_indices_array],
        "point_indices": np.arange(points_array.shape[0], dtype=np.int64),
    }


def refresh_student_vehicle_cache(env) -> None:
    """Collect valid per-step vehicle state once for observation and reward."""
    vehicle_ids: list[int] = []
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    speeds: list[float] = []
    lengths: list[float] = []
    widths: list[float] = []

    for veh in getattr(env, "vehicles", []):
        if getattr(veh, "physics_simulated", True) is False:
            continue

        veh_pos = veh.getPosition()
        if not is_valid_world_position(veh_pos.x, veh_pos.y):
            continue

        vehicle_ids.append(int(veh.getID()))
        positions.append((veh_pos.x, veh_pos.y))
        headings.append(float(veh.getHeading()))
        speeds.append(float(veh.getSpeed()))
        lengths.append(float(veh.getLength()))
        widths.append(float(veh.getWidth()))

    if not vehicle_ids:
        env._student_vehicle_cache = {
            "vehicle_ids": np.zeros(0, dtype=np.int64),
            "positions": np.zeros((0, 2), dtype=np.float32),
            "headings": np.zeros(0, dtype=np.float32),
            "speeds": np.zeros(0, dtype=np.float32),
            "lengths": np.zeros(0, dtype=np.float32),
            "widths": np.zeros(0, dtype=np.float32),
        }
        return

    env._student_vehicle_cache = {
        "vehicle_ids": np.asarray(vehicle_ids, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.float32),
        "headings": np.asarray(headings, dtype=np.float32),
        "speeds": np.asarray(speeds, dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.float32),
        "widths": np.asarray(widths, dtype=np.float32),
    }


def _get_student_vehicle_cache(env) -> dict[str, np.ndarray]:
    """Return the current step vehicle cache, building it on demand."""
    vehicle_cache = getattr(env, "_student_vehicle_cache", None)
    if vehicle_cache is None:
        refresh_student_vehicle_cache(env)
        vehicle_cache = env._student_vehicle_cache
    return vehicle_cache


def build_road_graph_obs(env, ego_pos, ego_heading: float) -> np.ndarray:
    """Build the road-graph observation matrix in the Gpudrive layout."""
    top_k = env._top_k_road_points
    if top_k <= 0:
        return np.zeros((0, ROAD_FEATURE_DIM), dtype=np.float32)

    road_cache = getattr(env, "_student_road_cache", None)
    if road_cache is None:
        road_cache = build_student_road_cache(getattr(env, "_road_graph_cache", None))
        env._student_road_cache = road_cache

    road_graph_states = np.zeros((top_k, ROAD_FEATURE_DIM), dtype=np.float32)
    if road_cache is None:
        return road_graph_states

    points = road_cache["points"]
    angle = angle_of_rotation(ego_heading)
    rel_x, rel_y = _to_local_array(
        points[:, 0] - float(ego_pos.x),
        points[:, 1] - float(ego_pos.y),
        angle,
    )
    dist_sq = rel_x * rel_x + rel_y * rel_y

    if points.shape[0] > top_k:
        selected_indices = np.argpartition(dist_sq, top_k - 1)[:top_k]
    else:
        selected_indices = np.arange(points.shape[0], dtype=np.int64)

    sort_order = np.lexsort(
        (
            road_cache["point_indices"][selected_indices],
            dist_sq[selected_indices],
        )
    )
    selected_indices = selected_indices[sort_order]

    num_selected = int(selected_indices.shape[0])
    road_graph_states[:num_selected, 0] = rel_x[selected_indices]
    road_graph_states[:num_selected, 1] = rel_y[selected_indices]
    road_graph_states[:num_selected, 2] = road_cache["segment_lengths"][selected_indices]
    road_graph_states[:num_selected, 3] = 1.0
    road_graph_states[:num_selected, 4] = 1.0
    road_graph_states[:num_selected, 5] = _normalize_angle_to_target(
        road_cache["orientations"][selected_indices],
        -angle,
    )
    road_graph_states[:num_selected, 6:] = road_cache["type_onehot"][selected_indices]
    return road_graph_states


def get_student_observation(env) -> np.ndarray:
    """Build the flattened student observation."""
    obs_dim = get_student_obs_dim(get_env_student_observation_config(env))
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return np.zeros(obs_dim, dtype=np.float32)

    ego_pos = env.ego_vehicle.getPosition()
    ego_heading = env.ego_vehicle.getHeading()
    ego_speed = env.ego_vehicle.getSpeed()

    goal_pos = env._ego_goal_dict["pos"]
    angle = angle_of_rotation(ego_heading)
    rel_goal_x, rel_goal_y = to_local(
        goal_pos[0] - ego_pos.x,
        goal_pos[1] - ego_pos.y,
        angle,
    )
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

    max_neighbors = getattr(env, "_max_observable_agents", 16)
    partner_states = np.zeros((max_neighbors, PARTNER_FEAT_DIM), dtype=np.float32)
    vehicle_cache = _get_student_vehicle_cache(env)
    vehicle_ids = vehicle_cache["vehicle_ids"]
    partner_mask = vehicle_ids != int(env.ego_vehicle.getID())

    if np.any(partner_mask) and max_neighbors > 0:
        partner_positions = vehicle_cache["positions"][partner_mask]
        partner_dx = partner_positions[:, 0] - float(ego_pos.x)
        partner_dy = partner_positions[:, 1] - float(ego_pos.y)
        partner_dist_sq = partner_dx * partner_dx + partner_dy * partner_dy
        sorted_partner_indices = np.argsort(partner_dist_sq, kind="stable")[:max_neighbors]
        selected_count = int(sorted_partner_indices.shape[0])
        rel_pos_x, rel_pos_y = _to_local_array(
            partner_dx[sorted_partner_indices],
            partner_dy[sorted_partner_indices],
            angle,
        )
        partner_headings = vehicle_cache["headings"][partner_mask][sorted_partner_indices]
        partner_states[:selected_count, 0] = vehicle_cache["speeds"][partner_mask][
            sorted_partner_indices
        ]
        partner_states[:selected_count, 1] = rel_pos_x
        partner_states[:selected_count, 2] = rel_pos_y
        partner_states[:selected_count, 3] = _normalize_angle_to_target(
            partner_headings,
            -angle,
        )
        partner_states[:selected_count, 4] = vehicle_cache["lengths"][partner_mask][
            sorted_partner_indices
        ]
        partner_states[:selected_count, 5] = vehicle_cache["widths"][partner_mask][
            sorted_partner_indices
        ]

    road_graph_states = build_road_graph_obs(env, ego_pos, ego_heading)
    obs_concat = np.concatenate(
        [
            ego_state,
            partner_states.reshape(-1),
            road_graph_states.reshape(-1),
        ]
    )
    return obs_concat[:obs_dim]
