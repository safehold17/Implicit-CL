"""Student-policy related helper functions for Nocturne CtrlSim adversarial env."""

from dataclasses import dataclass
from typing import List

import heapq
import math
import numpy as np

from .utils.common import (
    angle_of_rotation,
    angle_sub,
    is_valid_world_position,
    to_local,
)

# Pre-built one-hot matrix for road type indices — shared across all calls.
_ROAD_TYPE_ONEHOT_NP = np.eye(7, dtype=np.float32)

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


def apply_student_action(env, action: np.ndarray):
    """
    Apply student action to ego vehicle.

    Args:
        action: Discrete student action id.

    Returns:
        The decoded continuous `(accel, steer)` action, or `None` when ego is absent.
    """
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

    # Convert normalized action to actual values
    accel = accel_norm * 10.0  # max acc 10 m/s^2
    steer = steer_norm * 0.7  # max steer 0.7 rad

    if accel > 0:
        env.ego_vehicle.acceleration = accel
    else:
        env.ego_vehicle.brake(abs(accel))
    env.ego_vehicle.steering = steer
    return float(accel), float(steer)


def _select_partner_vehicles(env, ego_pos, max_neighbors: int) -> List:
    """Select nearest valid partner vehicles around ego.

    Uses numpy argsort for distance ranking instead of a Python sort,
    avoiding a Python-level comparison loop when there are many vehicles.
    """
    ego_vehicle = env.ego_vehicle
    valid_vehs = []
    xs = []
    ys = []

    for veh in getattr(env, "vehicles", []):
        if veh is ego_vehicle:
            continue
        if getattr(veh, "physics_simulated", True) is False:
            continue
        veh_pos = veh.getPosition()
        if not is_valid_world_position(veh_pos.x, veh_pos.y):
            continue
        valid_vehs.append(veh)
        xs.append(veh_pos.x)
        ys.append(veh_pos.y)

    if not valid_vehs:
        return []

    xs_np = np.array(xs, dtype=np.float32)
    ys_np = np.array(ys, dtype=np.float32)
    dist_sq = (xs_np - ego_pos.x) ** 2 + (ys_np - ego_pos.y) ** 2

    k = min(max_neighbors, len(valid_vehs))
    # argpartition is O(N) vs O(N log N) sort; then sort only the top-k slice
    top_idx = np.argpartition(dist_sq, k - 1)[:k] if k < len(valid_vehs) else np.arange(k)
    top_idx = top_idx[np.argsort(dist_sq[top_idx])]
    return [valid_vehs[i] for i in top_idx]


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


def build_road_graph_obs_np(road_graph_np: dict, ego_pos, ego_heading: float, top_k: int) -> np.ndarray:
    """
    Vectorized road graph observation using pre-flattened numpy arrays.

    Replaces the Python loop + heapq in build_road_graph_obs() with
    fully vectorized numpy ops. Computes top-k nearest road points in
    O(N) via argpartition instead of O(N log N) heapq insertion.

    Args:
        road_graph_np: dict returned by road_data_to_numpy() (populated at
                       scenario reset via env._road_graph_np)
                       keys: pts_xy (N,2), seg_len (N,), seg_orient (N,),
                             type_idx (N,), is_line (N,)
        ego_pos:       Nocturne position object with .x / .y
        ego_heading:   ego heading in radians (world frame)
        top_k:         number of road points to return

    Returns:
        out: float32 array of shape (top_k * ROAD_FEATURE_DIM,) — already flat,
             ready to be written directly into the obs buffer.
    """
    out = np.zeros(top_k * ROAD_FEATURE_DIM, dtype=np.float32)

    pts_xy = road_graph_np["pts_xy"]        # (N, 2)
    N = len(pts_xy)
    if N == 0:
        return out

    # --- Transform all road points to ego-local frame (vectorized) ---
    angle = float(angle_of_rotation(ego_heading))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    ego_xy = np.array([ego_pos.x, ego_pos.y], dtype=np.float32)
    delta = pts_xy - ego_xy                         # (N, 2)
    rel_x =  delta[:, 0] * cos_a + delta[:, 1] * sin_a   # (N,)
    rel_y = -delta[:, 0] * sin_a + delta[:, 1] * cos_a   # (N,)
    dist_sq = rel_x * rel_x + rel_y * rel_y               # (N,)

    # --- Top-k selection via argpartition (O(N) average) ---
    k = min(top_k, N)
    if k < N:
        top_idx = np.argpartition(dist_sq, k - 1)[:k]
    else:
        top_idx = np.arange(N)
    # Sort the top-k slice by ascending distance
    top_idx = top_idx[np.argsort(dist_sq[top_idx])]

    # --- Build feature rows for selected points ---
    seg_len    = road_graph_np["seg_len"][top_idx]     # (k,)
    seg_orient = road_graph_np["seg_orient"][top_idx]  # (k,)
    type_idx   = road_graph_np["type_idx"][top_idx]    # (k,)
    is_line    = road_graph_np["is_line"][top_idx]     # (k,) bool

    # Rotate segment orientation into ego frame for polyline points only.
    # Static objects (stop_sign, crosswalk, speed_bump) always have orientation=0.0
    # in the original code — angle_sub is never applied to them there.
    # Mirrors: polyline → angle_sub(world_orient, -angle); static → 0.0
    seg_orient_local = np.where(
        is_line,
        np.mod(-angle - seg_orient + math.pi, 2 * math.pi) - math.pi,
        0.0,
    )                                                  # (k,)

    # Each row: [rel_x, rel_y, seg_len, 1.0, 1.0, orientation, *one_hot(type)]
    # ROAD_FEATURE_DIM = 13: indices 0-5 scalar, 6-12 one-hot (7 classes)
    rows = np.zeros((k, ROAD_FEATURE_DIM), dtype=np.float32)
    rows[:, 0] = rel_x[top_idx]
    rows[:, 1] = rel_y[top_idx]
    rows[:, 2] = seg_len
    rows[:, 3] = 1.0
    rows[:, 4] = 1.0
    rows[:, 5] = seg_orient_local
    rows[:, 6:] = _ROAD_TYPE_ONEHOT_NP[type_idx]  # (k, 7) via advanced indexing

    out[:k * ROAD_FEATURE_DIM] = rows.ravel()
    return out


def get_student_observation(env) -> np.ndarray:
    """Get student policy observation (consistent with gpudrive).

    Uses a pre-allocated per-env buffer (env._obs_buffer) to avoid
    allocating a new array each step.  Falls back to allocating when
    the buffer is absent (e.g. first call before reset).

    Road graph uses the vectorized numpy path (build_road_graph_obs_np)
    when env._road_graph_np is available (set at scenario reset), and
    falls back to the original Python/heapq path otherwise.
    """
    config = get_env_student_observation_config(env)
    obs_dim = get_student_obs_dim(config)

    # --- Use or create a pre-allocated output buffer (opt 4) ---
    obs = getattr(env, "_obs_buffer", None)
    if obs is None or obs.shape[0] != obs_dim:
        obs = np.zeros(obs_dim, dtype=np.float32)
        env._obs_buffer = obs
    else:
        obs[:] = 0.0

    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return obs

    # ========== Ego state (6 dimensions) ==========
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

    obs[0] = ego_speed
    obs[1] = env.ego_vehicle.getLength()
    obs[2] = env.ego_vehicle.getWidth()
    obs[3] = rel_goal_x
    obs[4] = rel_goal_y
    obs[5] = collision_state

    # ========== Partner state (K*6 dimensions) ==========
    max_neighbors = getattr(env, "_max_observable_agents", 16)
    selected_vehicles = _select_partner_vehicles(env, ego_pos, max_neighbors)

    partner_start = EGO_FEAT_DIM
    for slot, veh in enumerate(selected_vehicles):
        veh_pos = veh.getPosition()
        rel_pos_x, rel_pos_y = to_local(
            veh_pos.x - ego_pos.x,
            veh_pos.y - ego_pos.y,
            angle,
        )
        rel_orientation = angle_sub(veh.getHeading(), -angle)
        base = partner_start + slot * PARTNER_FEAT_DIM
        obs[base + 0] = veh.getSpeed()
        obs[base + 1] = rel_pos_x
        obs[base + 2] = rel_pos_y
        obs[base + 3] = rel_orientation
        obs[base + 4] = veh.getLength()
        obs[base + 5] = veh.getWidth()
    # unfilled partner slots stay zero (buffer was zeroed above)

    # ========== Road Graph (R*13 dimensions) — vectorized path ==========
    top_k = getattr(env, "_top_k_road_points", 64)
    road_start = EGO_FEAT_DIM + max_neighbors * PARTNER_FEAT_DIM
    road_graph_np = getattr(env, "_road_graph_np", None)
    if road_graph_np is not None and len(road_graph_np["pts_xy"]) > 0:
        road_flat = build_road_graph_obs_np(road_graph_np, ego_pos, ego_heading, top_k)
        end = road_start + len(road_flat)
        obs[road_start:end] = road_flat
    else:
        # Fallback: original Python/heapq path
        road_graph_states = build_road_graph_obs(env, ego_pos, ego_heading)
        for i, feat in enumerate(road_graph_states):
            base = road_start + i * ROAD_FEATURE_DIM
            obs[base: base + ROAD_FEATURE_DIM] = feat

    return obs
