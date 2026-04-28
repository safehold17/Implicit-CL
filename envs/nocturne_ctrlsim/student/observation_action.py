"""Student observation and action helpers for the Nocturne CtrlSim environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ctrlsim_adapter._data_bridge.scenario import road_data_to_numpy
from ..utils.common import (
    angle_of_rotation,
    is_valid_world_position,
    to_local,
)

EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_FEATURE_DIM = 13
ROAD_TYPE_DIM = 7
ROAD_TYPE_ONEHOT = np.eye(ROAD_TYPE_DIM, dtype=np.float32)

# taken from GPUDriving
MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_REL_COORD = 1000.0
MAX_ORIENTATION_RAD = float(2.0 * np.pi)
MAX_ROAD_LINE_SEGMENT_LEN = 100.0


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


def get_student_obs_dim(config: StudentObservationConfig) -> int:
    """Return the flattened student observation dimension."""
    return (
        config.ego_feat_dim
        + config.max_neighbors * config.partner_feat_dim
        + config.top_k_road_points * config.road_graph_feat_dim
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

    action_id = int(np.asarray(action).reshape(-1).item())
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


def _normalize_by_scale(values, scale: float):
    """Normalize values by a fixed scale without clipping."""
    return np.asarray(values, dtype=np.float32) / np.float32(scale)


def _normalize_relative_coord(values):
    """Normalize relative coordinates into [-1, 1]."""
    return _normalize_by_scale(values, MAX_REL_COORD)


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


def select_partner_vehicles(
    env,
    ego_pos_x: float,
    ego_pos_y: float,
    angle: float,
    max_neighbors: int,
) -> dict[str, np.ndarray | int]:
    """Select the nearest partner vehicles from the cached per-step state."""
    vehicle_cache = _get_student_vehicle_cache(env)
    vehicle_ids = vehicle_cache["vehicle_ids"]
    partner_mask = vehicle_ids != int(env.ego_vehicle.getID())

    if not np.any(partner_mask) or max_neighbors <= 0:
        return {
            "count": 0,
            "speeds": np.zeros(0, dtype=np.float32),
            "rel_pos_x": np.zeros(0, dtype=np.float32),
            "rel_pos_y": np.zeros(0, dtype=np.float32),
            "rel_heading": np.zeros(0, dtype=np.float32),
            "lengths": np.zeros(0, dtype=np.float32),
            "widths": np.zeros(0, dtype=np.float32),
        }

    partner_positions = vehicle_cache["positions"][partner_mask]
    partner_dx = partner_positions[:, 0] - ego_pos_x
    partner_dy = partner_positions[:, 1] - ego_pos_y
    partner_dist_sq = partner_dx * partner_dx + partner_dy * partner_dy
    selected_indices = np.argsort(partner_dist_sq, kind="stable")[:max_neighbors]
    selected_count = int(selected_indices.shape[0])

    rel_pos_x, rel_pos_y = _to_local_array(
        partner_dx[selected_indices],
        partner_dy[selected_indices],
        angle,
    )
    selected_headings = vehicle_cache["headings"][partner_mask][selected_indices]

    return {
        "count": selected_count,
        "speeds": vehicle_cache["speeds"][partner_mask][selected_indices],
        "rel_pos_x": rel_pos_x,
        "rel_pos_y": rel_pos_y,
        "rel_heading": _normalize_angle_to_target(selected_headings, -angle),
        "lengths": vehicle_cache["lengths"][partner_mask][selected_indices],
        "widths": vehicle_cache["widths"][partner_mask][selected_indices],
    }


def build_road_graph_obs_np(
    road_graph_np: dict[str, np.ndarray] | None,
    ego_pos_x: float,
    ego_pos_y: float,
    angle: float,
    top_k: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build a flat road-graph observation segment from ``_road_graph_np``.

    ``angle`` is the pre-computed ego-frame rotation (``angle_of_rotation``).
    Accepting it as a parameter avoids recomputing it when the caller has
    already evaluated it (e.g. ``get_student_observation``).

    When ``out`` is provided it must have shape ``(top_k * ROAD_FEATURE_DIM,)``
    and is assumed to be pre-zeroed by the caller (the obs buffer is zeroed at
    the start of ``get_student_observation``). Providing ``out`` lets the
    caller write the road features directly into the observation buffer,
    eliminating a separate flat-array allocation and the subsequent copy.

    ``point_indices`` is used only as a deterministic tie-breaker after
    distance-based top-k selection. When multiple road points are exactly the
    same distance from ego, preserving the original scenario parsing order
    keeps the emitted observation stable across runs.
    """
    if top_k <= 0:
        return np.zeros(0, dtype=np.float32)

    if out is None:
        out = np.zeros(top_k * ROAD_FEATURE_DIM, dtype=np.float32)
    # When ``out`` is provided the caller guarantees it is already zeroed,
    # so padding slots require no additional work.

    if road_graph_np is None:
        return out

    points = road_graph_np["pts_xy"]
    if points.shape[0] == 0:
        return out

    rel_x, rel_y = _to_local_array(
        points[:, 0] - ego_pos_x,
        points[:, 1] - ego_pos_y,
        angle,
    )
    dist_sq = rel_x * rel_x + rel_y * rel_y

    num_points = points.shape[0]
    if num_points > top_k:
        selected_indices = np.argpartition(dist_sq, top_k - 1)[:top_k]
    else:
        selected_indices = np.arange(num_points, dtype=np.int64)

    sort_order = np.lexsort(
        (
            road_graph_np["point_indices"][selected_indices],
            dist_sq[selected_indices],
        )
    )
    selected_indices = selected_indices[sort_order]

    num_selected = int(selected_indices.shape[0])
    # Write directly into a reshaped view of ``out`` — eliminates a separate
    # (num_selected, ROAD_FEATURE_DIM) allocation and the final flatten+copy.
    rows = out[: num_selected * ROAD_FEATURE_DIM].reshape(num_selected, ROAD_FEATURE_DIM)
    rows[:, 0] = _normalize_relative_coord(rel_x[selected_indices])
    rows[:, 1] = _normalize_relative_coord(rel_y[selected_indices])
    rows[:, 2] = _normalize_by_scale(
        road_graph_np["seg_len"][selected_indices],
        MAX_ROAD_LINE_SEGMENT_LEN,
    )

    # Legacy constant channels for compatibility with current 13-dim road feature layout.
    rows[:, 3] = 1.0
    rows[:, 4] = 1.0

    selected_is_line = road_graph_np["is_line"][selected_indices]
    selected_orientations = road_graph_np["seg_orient"][selected_indices]

    # Static road objects keep a zero heading feature. Only polyline points
    # have a segment direction that should be rotated into the ego frame.
    if np.any(selected_is_line):
        rows[selected_is_line, 5] = _normalize_angle_to_target(
            selected_orientations[selected_is_line],
            -angle,
        ) / np.float32(MAX_ORIENTATION_RAD)

    rows[:, 6:] = ROAD_TYPE_ONEHOT[road_graph_np["type_idx"][selected_indices]]
    return out


def get_student_observation(env) -> np.ndarray:
    """Build the flattened student observation.

    Dispatches to the CtRL-Sim aligned builder when one is attached to the env
    (``env._ctrlsim_obs_builder`` is set), otherwise falls back to the legacy
    LateFusion observation.
    """
    ctrlsim_builder = getattr(env, "_ctrlsim_obs_builder", None)
    if ctrlsim_builder is not None:
        return ctrlsim_builder.step(env)

    # ------------------------------------------------------------------ legacy
    config = env.student_observation_config
    obs_dim = get_student_obs_dim(config)

    obs = getattr(env, "_obs_buffer", None)
    if obs is None or obs.shape[0] != obs_dim:
        obs = np.zeros(obs_dim, dtype=np.float32)
        env._obs_buffer = obs
    else:
        obs[:] = 0.0

    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return obs

    # Read from the step-scoped cache populated once per step by
    # _update_step_ego_cache() to avoid redundant C++ boundary crossings.
    step_ego_pos_arr = getattr(env, "_step_ego_pos_arr", None)
    if step_ego_pos_arr is not None:
        ego_pos_x = float(step_ego_pos_arr[0])
        ego_pos_y = float(step_ego_pos_arr[1])
        ego_heading = env._step_ego_heading
        ego_speed = env._step_ego_speed
    else:
        _pos = env.ego_vehicle.getPosition()
        ego_pos_x = _pos.x
        ego_pos_y = _pos.y
        ego_heading = float(env.ego_vehicle.getHeading())
        ego_speed = float(env.ego_vehicle.getSpeed())

    goal_pos = env._ego_goal_dict["pos"]
    angle = angle_of_rotation(ego_heading)
    rel_goal_x, rel_goal_y = to_local(
        goal_pos[0] - ego_pos_x,
        goal_pos[1] - ego_pos_y,
        angle,
    )
    collision_state = 1.0 if env._collision_occurred else 0.0

    obs[0] = _normalize_by_scale(ego_speed, MAX_SPEED)
    # Use episode-cached ego geometry (constant for the whole episode).
    obs[1] = _normalize_by_scale(
        getattr(env, "_ego_length", env.ego_vehicle.getLength()),
        MAX_VEH_LEN,
    )
    obs[2] = _normalize_by_scale(
        getattr(env, "_ego_width", env.ego_vehicle.getWidth()),
        MAX_VEH_WIDTH,
    )
    obs[3] = _normalize_relative_coord(rel_goal_x)
    obs[4] = _normalize_relative_coord(rel_goal_y)
    obs[5] = collision_state

    max_neighbors = getattr(env, "_max_observable_agents", 16)
    selected_partners = select_partner_vehicles(
        env=env,
        ego_pos_x=ego_pos_x,
        ego_pos_y=ego_pos_y,
        angle=angle,
        max_neighbors=max_neighbors,
    )
    partner_start = EGO_FEAT_DIM
    count = int(selected_partners["count"])
    if count > 0:
        # Vectorized slice assignment into a reshaped view — replaces a Python
        # for-loop over ``count`` partners.
        obs_partner_view = obs[
            partner_start : partner_start + count * PARTNER_FEAT_DIM
        ].reshape(count, PARTNER_FEAT_DIM)
        obs_partner_view[:, 0] = _normalize_by_scale(
            selected_partners["speeds"],
            MAX_SPEED,
        )
        obs_partner_view[:, 1] = _normalize_relative_coord(
            selected_partners["rel_pos_x"]
        )
        obs_partner_view[:, 2] = _normalize_relative_coord(
            selected_partners["rel_pos_y"]
        )
        obs_partner_view[:, 3] = _normalize_by_scale(
            selected_partners["rel_heading"],
            MAX_ORIENTATION_RAD,
        )
        obs_partner_view[:, 4] = _normalize_by_scale(
            selected_partners["lengths"],
            MAX_VEH_LEN,
        )
        obs_partner_view[:, 5] = _normalize_by_scale(
            selected_partners["widths"],
            MAX_VEH_WIDTH,
        )

    road_start = EGO_FEAT_DIM + max_neighbors * PARTNER_FEAT_DIM
    road_graph_np = getattr(env, "_road_graph_np", None)
    if road_graph_np is None:
        road_graph_cache = getattr(env, "_road_graph_cache", None)
        if road_graph_cache is not None:
            road_graph_np = road_data_to_numpy(road_graph_cache)
            env._road_graph_np = road_graph_np

    top_k = getattr(env, "_top_k_road_points", 64)
    # Pass a view of the obs buffer so build_road_graph_obs_np writes the road
    # features directly into place — no separate flat array + copy needed.
    # The obs buffer is already zeroed above so padding slots are correct.
    # ``angle`` is pre-computed above and passed through to avoid a duplicate
    # angle_of_rotation() call inside build_road_graph_obs_np.
    build_road_graph_obs_np(
        road_graph_np,
        ego_pos_x,
        ego_pos_y,
        angle,
        top_k,
        out=obs[road_start : road_start + top_k * ROAD_FEATURE_DIM],
    )
    return obs
