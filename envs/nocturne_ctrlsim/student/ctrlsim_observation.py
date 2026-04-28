"""CtRL-Sim aligned student observation builder.

Observation layout (all ego-relative, SE(2) transformed, flat float32):
  agent_states : (num_agents, seq_len, 8)   - px,py,vx,vy,heading,length,width,existence
  agent_types  : (num_agents, 5)             - one-hot vehicle type
  goals        : (num_agents, 5)             - goal_px,py,vx,vy,heading in ego frame
  road_points  : (max_roads, max_pts, 3)     - ego-relative xy + valid flag
  road_types   : (max_roads, road_type_dim)  - one-hot road category
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CtrlSimObsConfig:
    """Layout config for the CtRL-Sim aligned student observation."""

    num_agents: int = 9           # ego + 8 nearest neighbours
    seq_len: int = 10             # steps of history per agent
    max_roads: int = 64           # number of road polylines
    max_pts_per_road: int = 20    # points per polyline
    state_dim: int = 8            # px,py,vx,vy,heading,length,width,existence
    agent_type_dim: int = 5       # one-hot vehicle type
    goal_dim: int = 5             # goal_px,py,vx,vy,heading
    road_pt_dim: int = 3          # x,y,valid
    road_type_dim: int = 8        # one-hot road type

    @property
    def num_neighbors(self) -> int:
        return self.num_agents - 1

    @property
    def obs_dim(self) -> int:
        return (
            self.num_agents * self.seq_len * self.state_dim
            + self.num_agents * self.agent_type_dim
            + self.num_agents * self.goal_dim
            + self.max_roads * self.max_pts_per_road * self.road_pt_dim
            + self.max_roads * self.road_type_dim
        )


# ---------------------------------------------------------------------------
# SE(2) helpers  (matches focal_input.py transform)
# ---------------------------------------------------------------------------

def _angle_of_rotation(yaw: float) -> float:
    """Return the rotation angle that maps world-frame to ego-frame."""
    return (math.pi / 2.0) - float(yaw)


def _rotate_xy_batch(xy: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a (..., 2) array of (x, y) pairs by *angle* radians."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] * cos_a + xy[..., 1] * sin_a
    out[..., 1] = -xy[..., 0] * sin_a + xy[..., 1] * cos_a
    return out


def _angle_sub(current: np.ndarray, target: float) -> np.ndarray:
    """Wrap angular difference into (-pi, pi]."""
    diff = (float(target) - current + np.pi) % (2.0 * np.pi) - np.pi
    return diff.astype(np.float32)


# ---------------------------------------------------------------------------
# CtrlSimObsBuilder
# ---------------------------------------------------------------------------

class CtrlSimObsBuilder:
    """Builds CtRL-Sim aligned observations for the student agent each step.

    One instance lives on the environment for the lifetime of a training run.
    Call ``reset(env)`` at the start of each episode, then ``step(env)`` at
    each simulation step to get the flat numpy observation.
    """

    def __init__(self, cfg: CtrlSimObsConfig) -> None:
        self.cfg = cfg
        # Pre-allocate flat output buffer (reused each step to avoid GC pressure).
        self._obs_buf = np.zeros(cfg.obs_dim, dtype=np.float32)
        # History deque filled during an episode.
        self._state_history: deque = deque(maxlen=cfg.seq_len)
        # Cached per-episode data (set in reset).
        self._road_points_world: Optional[np.ndarray] = None   # (R, P, 3)
        self._road_types: Optional[np.ndarray] = None           # (R, road_type_dim)
        self._vehicle_goals_world: Dict[int, np.ndarray] = {}  # veh_id → (goal_px, goal_py)
        # Agent type one-hot: all vehicles → [1, 0, 0, 0, 0].
        self._agent_types = np.zeros(
            (cfg.num_agents, cfg.agent_type_dim), dtype=np.float32
        )
        self._agent_types[:, 0] = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env: Any) -> None:
        """Cache episode-scoped road/goal data and clear history."""
        self._state_history.clear()

        # Road polylines from preprocessed data (world frame).
        preproc = getattr(env, "_preproc_data", None)
        if preproc is not None and "road_points" in preproc and "road_types" in preproc:
            rp = np.asarray(preproc["road_points"], dtype=np.float32)
            rt = np.asarray(preproc["road_types"], dtype=np.float32)
            # Truncate / pad to (max_roads, max_pts, 3) and (max_roads, road_type_dim).
            self._road_points_world = self._fit_road_points(rp)
            self._road_types = self._fit_road_types(rt)
        else:
            self._road_points_world = np.zeros(
                (self.cfg.max_roads, self.cfg.max_pts_per_road, 3), dtype=np.float32
            )
            self._road_types = np.zeros(
                (self.cfg.max_roads, self.cfg.road_type_dim), dtype=np.float32
            )

        # Per-vehicle goal positions from GT trajectories.
        self._vehicle_goals_world = {}
        gt = getattr(env, "_gt_data_dict", {})
        for veh_id, data in gt.items():
            if not isinstance(data, dict) or "traj" not in data:
                continue
            traj = np.asarray(data["traj"], dtype=np.float32)
            # traj columns: px, py, heading, speed, existence, goal_x, goal_y, length
            if traj.shape[1] >= 7:
                self._vehicle_goals_world[int(veh_id)] = traj[0, 5:7].copy()

    def step(self, env: Any) -> np.ndarray:
        """Build and return the flat student observation for the current step."""
        if env.ego_vehicle is None:
            return np.zeros(self.cfg.obs_dim, dtype=np.float32)

        # Current ego kinematics from the step-scoped cache.
        step_pos = getattr(env, "_step_ego_pos_arr", None)
        if step_pos is not None:
            ego_px = float(step_pos[0])
            ego_py = float(step_pos[1])
            ego_heading = float(env._step_ego_heading)
            ego_speed = float(env._step_ego_speed)
        else:
            _p = env.ego_vehicle.getPosition()
            ego_px, ego_py = float(_p.x), float(_p.y)
            ego_heading = float(env.ego_vehicle.getHeading())
            ego_speed = float(env.ego_vehicle.getSpeed())

        angle = _angle_of_rotation(ego_heading)

        # Collect world-frame states for all agents this step.
        world_states = self._collect_world_states(env, ego_px, ego_py, ego_speed, ego_heading)
        self._state_history.append(world_states)  # (num_agents, 8)

        # Build padded history tensor: (seq_len, num_agents, 8).
        history = self._build_padded_history()

        # Transform history to ego frame.
        ego_history = self._apply_se2_to_history(history, ego_px, ego_py, angle)
        # Transpose to (num_agents, seq_len, 8).
        agent_states = ego_history.transpose(1, 0, 2)

        # Goals in ego frame.
        goals = self._build_goals(env, ego_px, ego_py, angle)

        # Road polylines in ego frame.
        road_pts_ego = self._transform_road_points(ego_px, ego_py, angle)

        return self._pack(agent_states, goals, road_pts_ego)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_world_states(
        self,
        env: Any,
        ego_px: float,
        ego_py: float,
        ego_speed: float,
        ego_heading: float,
    ) -> np.ndarray:
        """Build (num_agents, 8) state array in world frame for current step.

        Slot 0 is always ego; slots 1..N are nearest neighbours sorted by
        distance. Missing slots are zero-padded (existence=0).
        """
        cfg = self.cfg
        out = np.zeros((cfg.num_agents, cfg.state_dim), dtype=np.float32)

        # Ego slot.
        ego_vx = ego_speed * math.cos(ego_heading)
        ego_vy = ego_speed * math.sin(ego_heading)
        ego_len = float(getattr(env, "_ego_length", env.ego_vehicle.getLength()))
        ego_wid = float(getattr(env, "_ego_width", env.ego_vehicle.getWidth()))
        out[0] = [ego_px, ego_py, ego_vx, ego_vy, ego_heading, ego_len, ego_wid, 1.0]

        # Neighbour slots.
        if cfg.num_neighbors <= 0:
            return out

        ego_id = int(env.ego_vehicle.getID())
        cache = getattr(env, "_student_vehicle_cache", None)
        if cache is None or cache["vehicle_ids"].size == 0:
            return out

        vids = cache["vehicle_ids"]
        mask = vids != ego_id
        if not np.any(mask):
            return out

        pos = cache["positions"][mask]
        dx = pos[:, 0] - ego_px
        dy = pos[:, 1] - ego_py
        dist_sq = dx * dx + dy * dy
        k = min(cfg.num_neighbors, int(dist_sq.shape[0]))
        sel = np.argsort(dist_sq, kind="stable")[:k]

        headings = cache["headings"][mask][sel]
        speeds = cache["speeds"][mask][sel]
        lengths = cache["lengths"][mask][sel]
        widths = cache["widths"][mask][sel]

        n_vx = speeds * np.cos(headings)
        n_vy = speeds * np.sin(headings)

        for i, s in enumerate(sel):
            out[1 + i] = [
                pos[s, 0], pos[s, 1],
                n_vx[i], n_vy[i],
                headings[i],
                lengths[i], widths[i],
                1.0,
            ]

        return out

    def _build_padded_history(self) -> np.ndarray:
        """Return (seq_len, num_agents, 8) with zero-padding for early steps."""
        cfg = self.cfg
        out = np.zeros((cfg.seq_len, cfg.num_agents, cfg.state_dim), dtype=np.float32)
        hist = list(self._state_history)  # oldest → newest
        offset = cfg.seq_len - len(hist)
        for i, frame in enumerate(hist):
            out[offset + i] = frame
        return out

    def _apply_se2_to_history(
        self,
        history: np.ndarray,   # (seq_len, num_agents, 8)
        ego_px: float,
        ego_py: float,
        angle: float,
    ) -> np.ndarray:
        """Transform all agent states to ego-relative frame.

        Positions: translate then rotate.
        Velocities: rotate only (no translation).
        Headings: angle subtraction.
        length, width, existence: unchanged.
        """
        out = history.copy()

        # Positions.
        out[..., 0] -= ego_px
        out[..., 1] -= ego_py
        pos_xy = out[..., :2].copy()
        out[..., :2] = _rotate_xy_batch(pos_xy, angle)

        # Velocities (rotate only).
        vel_xy = out[..., 2:4].copy()
        out[..., 2:4] = _rotate_xy_batch(vel_xy, angle)

        # Headings.
        out[..., 4] = _angle_sub(out[..., 4], -angle)

        return out.astype(np.float32)

    def _build_goals(
        self,
        env: Any,
        ego_px: float,
        ego_py: float,
        angle: float,
    ) -> np.ndarray:
        """Return (num_agents, 5) goal tensor in ego frame.

        Goal features: [goal_px, goal_py, goal_vx, goal_vy, goal_heading].
        goal_vx = goal_vy = 0; goal_heading = 0 (unknown for neighbours).
        """
        cfg = self.cfg
        goals = np.zeros((cfg.num_agents, cfg.goal_dim), dtype=np.float32)

        # Ego goal.
        ego_goal = getattr(env, "_ego_goal_dict", None)
        if ego_goal is not None:
            gx, gy = float(ego_goal["pos"][0]) - ego_px, float(ego_goal["pos"][1]) - ego_py
            gxy = np.array([[gx, gy]], dtype=np.float32)
            gxy_rot = _rotate_xy_batch(gxy, angle)[0]
            goals[0, :2] = gxy_rot

        # Neighbour goals (use GT scenario goals when available).
        cache = getattr(env, "_student_vehicle_cache", None)
        if cache is None or cfg.num_neighbors <= 0:
            return goals

        ego_id = int(env.ego_vehicle.getID())
        vids = cache["vehicle_ids"]
        mask = vids != ego_id
        if not np.any(mask):
            return goals

        pos = cache["positions"][mask]
        dx = pos[:, 0] - ego_px
        dy = pos[:, 1] - ego_py
        dist_sq = dx * dx + dy * dy
        k = min(cfg.num_neighbors, int(dist_sq.shape[0]))
        sel_idx = np.argsort(dist_sq, kind="stable")[:k]
        sel_vids = vids[mask][sel_idx]

        for slot, vid in enumerate(sel_vids):
            gp = self._vehicle_goals_world.get(int(vid))
            if gp is None:
                continue
            gx, gy = float(gp[0]) - ego_px, float(gp[1]) - ego_py
            gxy = np.array([[gx, gy]], dtype=np.float32)
            gxy_rot = _rotate_xy_batch(gxy, angle)[0]
            goals[1 + slot, :2] = gxy_rot

        return goals

    def _transform_road_points(
        self,
        ego_px: float,
        ego_py: float,
        angle: float,
    ) -> np.ndarray:
        """Return road_points in ego frame: (max_roads, max_pts, 3)."""
        rp = self._road_points_world
        if rp is None:
            return np.zeros(
                (self.cfg.max_roads, self.cfg.max_pts_per_road, 3), dtype=np.float32
            )
        out = rp.copy()
        # Translate then rotate xy.
        valid = out[..., 2] > 0.5   # boolean mask of valid points
        out[..., 0] -= ego_px
        out[..., 1] -= ego_py
        xy = out[..., :2].copy()
        out[..., :2] = _rotate_xy_batch(xy, angle)
        # Zero out invalid point coordinates so they don't pollute attention.
        out[~valid, :2] = 0.0
        return out.astype(np.float32)

    def _pack(
        self,
        agent_states: np.ndarray,   # (num_agents, seq_len, 8)
        goals: np.ndarray,           # (num_agents, 5)
        road_points: np.ndarray,     # (max_roads, max_pts, 3)
    ) -> np.ndarray:
        """Flatten all tensors into the output buffer."""
        buf = self._obs_buf
        buf[:] = 0.0
        cfg = self.cfg
        idx = 0

        n = cfg.num_agents * cfg.seq_len * cfg.state_dim
        buf[idx:idx + n] = agent_states.ravel()
        idx += n

        n = cfg.num_agents * cfg.agent_type_dim
        buf[idx:idx + n] = self._agent_types.ravel()
        idx += n

        n = cfg.num_agents * cfg.goal_dim
        buf[idx:idx + n] = goals.ravel()
        idx += n

        n = cfg.max_roads * cfg.max_pts_per_road * cfg.road_pt_dim
        buf[idx:idx + n] = road_points.ravel()
        idx += n

        n = cfg.max_roads * cfg.road_type_dim
        if self._road_types is not None:
            buf[idx:idx + n] = self._road_types.ravel()
        idx += n

        return buf

    # ------------------------------------------------------------------
    # Road shape helpers
    # ------------------------------------------------------------------

    def _fit_road_points(self, rp: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad road_points to (max_roads, max_pts, 3)."""
        R, P = self.cfg.max_roads, self.cfg.max_pts_per_road
        out = np.zeros((R, P, 3), dtype=np.float32)
        r_in = min(rp.shape[0], R)
        p_in = min(rp.shape[1], P)
        d_in = min(rp.shape[2], 3)
        out[:r_in, :p_in, :d_in] = rp[:r_in, :p_in, :d_in]
        return out

    def _fit_road_types(self, rt: np.ndarray) -> np.ndarray:
        """Truncate or zero-pad road_types to (max_roads, road_type_dim)."""
        R, D = self.cfg.max_roads, self.cfg.road_type_dim
        out = np.zeros((R, D), dtype=np.float32)
        r_in = min(rt.shape[0], R)
        d_in = min(rt.shape[1], D)
        out[:r_in, :d_in] = rt[:r_in, :d_in]
        return out


# ---------------------------------------------------------------------------
# Unpack helper used by the model
# ---------------------------------------------------------------------------

class _AgentData:
    __slots__ = ("agent_states", "agent_types", "goals")

class _MapData:
    __slots__ = ("road_points", "road_types")

class _SceneData:
    __slots__ = ("agent", "map")

def unpack_obs(flat_obs: "torch.Tensor", cfg: CtrlSimObsConfig) -> "_SceneData":
    """Reshape a (batch, obs_dim) tensor into a CtRL-Sim compatible data object.

    Returns a lightweight namespace ``data`` with:
      data.agent.agent_states : (batch, num_agents, seq_len, 8)
      data.agent.agent_types  : (batch, num_agents, 5)
      data.agent.goals        : (batch, num_agents, 5)
      data.map.road_points    : (batch, max_roads, max_pts, 3)
      data.map.road_types     : (batch, max_roads, road_type_dim)
    """
    B = flat_obs.shape[0]
    idx = 0

    n = cfg.num_agents * cfg.seq_len * cfg.state_dim
    agent_states = flat_obs[:, idx:idx + n].reshape(B, cfg.num_agents, cfg.seq_len, cfg.state_dim)
    idx += n

    n = cfg.num_agents * cfg.agent_type_dim
    agent_types = flat_obs[:, idx:idx + n].reshape(B, cfg.num_agents, cfg.agent_type_dim)
    idx += n

    n = cfg.num_agents * cfg.goal_dim
    goals = flat_obs[:, idx:idx + n].reshape(B, cfg.num_agents, cfg.goal_dim)
    idx += n

    n = cfg.max_roads * cfg.max_pts_per_road * cfg.road_pt_dim
    road_points = flat_obs[:, idx:idx + n].reshape(
        B, cfg.max_roads, cfg.max_pts_per_road, cfg.road_pt_dim
    )
    idx += n

    n = cfg.max_roads * cfg.road_type_dim
    road_types = flat_obs[:, idx:idx + n].reshape(B, cfg.max_roads, cfg.road_type_dim)

    agent = _AgentData()
    agent.agent_states = agent_states
    agent.agent_types = agent_types
    agent.goals = goals

    map_data = _MapData()
    map_data.road_points = road_points
    map_data.road_types = road_types

    data = _SceneData()
    data.agent = agent
    data.map = map_data
    return data
