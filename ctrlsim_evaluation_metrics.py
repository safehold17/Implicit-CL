"""CtrlSim-style trajectory metrics for evaluation outputs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


CTRLSIM_EGO_METRIC_FIELDS = (
    "fde",
    "ade",
    "lin_speed_jsd",
    "ang_speed_jsd",
    "accel_jsd",
    "nearest_dist_jsd",
    "meta_jsd",
)


def _zero_metrics() -> dict[str, float]:
    """Return a zero-valued CtrlSim ego metric row."""
    return {field: 0.0 for field in CTRLSIM_EGO_METRIC_FIELDS}


def _get_nested_value(source: Any, path: Iterable[str], default: Any) -> Any:
    """Read nested dict/attribute values from configs."""
    value = source
    for key in path:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
        if value is default:
            return default
    return value


def _position_array(entries: list[Any], limit: int) -> np.ndarray:
    """Convert recorded position entries to an ``N x 2`` array."""
    positions = []
    for entry in entries[:limit]:
        if isinstance(entry, dict):
            positions.append([float(entry["x"]), float(entry["y"])])
        else:
            arr = np.asarray(entry, dtype=np.float32)
            positions.append([float(arr[0]), float(arr[1])])
    return np.asarray(positions, dtype=np.float32)


def _speed_array(entries: list[Any], limit: int) -> np.ndarray:
    """Convert recorded velocity entries to scalar speeds."""
    velocities = []
    for entry in entries[:limit]:
        if isinstance(entry, dict):
            velocities.append([float(entry["x"]), float(entry["y"])])
        else:
            arr = np.asarray(entry, dtype=np.float32)
            velocities.append([float(arr[0]), float(arr[1])])
    if not velocities:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(np.asarray(velocities, dtype=np.float32), axis=1)


def _probabilities(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Build histogram probabilities for Jensen-Shannon distance."""
    counts = np.histogram(values, bins=bins)[0].astype(np.float64)
    total = float(counts.sum())
    if total == 0.0:
        return np.zeros_like(counts)
    return counts / total


def _jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return SciPy-compatible Jensen-Shannon distance."""
    mixture = 0.5 * (left + right)
    left_mask = left > 0.0
    right_mask = right > 0.0
    divergence = 0.0
    if np.any(left_mask):
        divergence += 0.5 * float(
            np.sum(left[left_mask] * np.log(left[left_mask] / mixture[left_mask]))
        )
    if np.any(right_mask):
        divergence += 0.5 * float(
            np.sum(right[right_mask] * np.log(right[right_mask] / mixture[right_mask]))
        )
    return float(np.sqrt(max(divergence, 0.0)))


def _histogram_jsd(
    sim_values: np.ndarray,
    gt_values: np.ndarray,
    bins: np.ndarray,
) -> float:
    """Compute Jensen-Shannon distance between two value arrays."""
    if sim_values.size == 0 or gt_values.size == 0:
        return 0.0
    return _jensen_shannon_distance(
        _probabilities(sim_values, bins),
        _probabilities(gt_values, bins),
    )


def _central_gt_acceleration(gt_speeds: np.ndarray, dt: float) -> np.ndarray:
    """Compute CtrlSim's central-difference GT acceleration approximation."""
    gt_accels = np.zeros_like(gt_speeds, dtype=np.float32)
    if len(gt_speeds) > 2:
        gt_accels[1:-1] = (gt_speeds[2:] - gt_speeds[:-2]) / (2.0 * dt)
    return gt_accels


def _gt_nearest_distances(
    gt_data_dict: dict[int, Any],
    ego_id: int,
    limit: int,
) -> np.ndarray:
    """Compute nearest GT vehicle distance for the ego at each timestep."""
    ego_traj = np.asarray(gt_data_dict[ego_id]["traj"], dtype=np.float32)
    distances = []
    for t in range(limit):
        if t >= len(ego_traj) or not bool(ego_traj[t, 4]):
            distances.append(0.0)
            continue
        ego_pos = ego_traj[t, :2]
        step_distances = []
        for veh_id, data in gt_data_dict.items():
            if int(veh_id) == int(ego_id):
                continue
            traj = np.asarray(data["traj"], dtype=np.float32)
            if t >= len(traj) or not bool(traj[t, 4]):
                continue
            step_distances.append(float(np.linalg.norm(ego_pos - traj[t, :2])))
        distances.append(min(step_distances) if step_distances else 0.0)
    return np.asarray(distances, dtype=np.float32)


def _get_vehicle_data(adapter: Any, ego_id: int) -> dict[str, Any] | None:
    """Read one vehicle's recorded data from a CtrlSim adapter."""
    if hasattr(adapter, "get_vehicle_data"):
        return adapter.get_vehicle_data(ego_id)
    return getattr(adapter, "_vehicle_data_dict", {}).get(ego_id)


def _get_gt_traj(adapter: Any, ego_id: int) -> np.ndarray | None:
    """Read one vehicle's GT trajectory from a CtrlSim adapter."""
    gt_traj_by_id = getattr(adapter, "_gt_traj_by_id", {})
    if ego_id in gt_traj_by_id:
        return np.asarray(gt_traj_by_id[ego_id], dtype=np.float32)
    gt_data = getattr(adapter, "_gt_data_dict", {}).get(ego_id)
    if isinstance(gt_data, dict) and "traj" in gt_data:
        return np.asarray(gt_data["traj"], dtype=np.float32)
    return None


def compute_ctrlsim_ego_metrics_from_adapter(
    adapter: Any,
    ego_id: int | None,
) -> dict[str, float]:
    """Compute CtrlSim ADE/FDE/JSD metrics for one ego episode."""
    if adapter is None or ego_id is None:
        return _zero_metrics()

    ego_id = int(ego_id)
    vehicle_data = _get_vehicle_data(adapter, ego_id)
    gt_traj = _get_gt_traj(adapter, ego_id)
    gt_data_dict = getattr(adapter, "_gt_data_dict", {})
    if not vehicle_data or gt_traj is None or ego_id not in gt_data_dict:
        return _zero_metrics()

    positions = vehicle_data.get("position", [])
    existence = vehicle_data.get("existence", [])
    limit = min(len(positions), len(existence), len(gt_traj))
    if limit == 0:
        return _zero_metrics()

    sim_positions = _position_array(positions, limit)
    gt_positions = np.asarray(gt_traj[:limit, :2], dtype=np.float32)
    mask = np.asarray(existence[:limit], dtype=bool) & np.asarray(gt_traj[:limit, 4], dtype=bool)
    history_steps = int(getattr(adapter, "history_steps", 10))
    if history_steps > 0:
        mask[:history_steps] = False
    if not np.any(mask):
        return _zero_metrics()

    position_errors = np.linalg.norm(sim_positions[mask] - gt_positions[mask], axis=1)
    last_position = np.where(mask)[0][-1]

    dt = float(getattr(adapter, "dt", 1.0))
    cfg = getattr(adapter, "cfg", None)
    waymo_cfg = _get_nested_value(cfg, ("dataset", "waymo"), None)
    min_accel = float(_get_nested_value(waymo_cfg, ("min_accel",), -10.0))
    max_accel = float(_get_nested_value(waymo_cfg, ("max_accel",), 10.0))
    accel_discretization = int(
        _get_nested_value(waymo_cfg, ("accel_discretization",), 20)
    )

    sim_speeds = _speed_array(vehicle_data.get("velocity", []), limit)[mask]
    gt_speeds = np.asarray(gt_traj[:limit, 3], dtype=np.float32)[mask]
    lin_speed_bins = np.arange(201, dtype=np.float32) * 0.5 * (100.0 / 30.0)

    sim_ang_speeds = (
        np.asarray(vehicle_data.get("heading", [])[:limit], dtype=np.float32)[mask] / dt
    )
    gt_ang_speeds = np.asarray(gt_traj[:limit, 2], dtype=np.float32)[mask] / dt
    ang_speed_bins = np.arange(201, dtype=np.float32) * 0.5 - 50.0

    sim_accels = np.asarray(vehicle_data.get("acceleration", [])[:limit], dtype=np.float32)[mask]
    gt_accels = _central_gt_acceleration(np.asarray(gt_traj[:limit, 3], dtype=np.float32), dt)[mask]
    if sim_accels.size > 2 and gt_accels.size > 2:
        sim_accels = sim_accels[1:-1]
        gt_accels = gt_accels[1:-1]
    gt_accels = np.clip(gt_accels, a_min=min_accel, a_max=max_accel)
    gt_accels = (gt_accels - min_accel) / (max_accel - min_accel)
    gt_accels = np.round(gt_accels * (accel_discretization - 1))
    gt_accels = gt_accels / (accel_discretization - 1)
    gt_accels = gt_accels * (max_accel - min_accel) + min_accel
    accel_bins = np.arange(accel_discretization + 1, dtype=np.float32) * 2.0 - accel_discretization

    sim_nearest = np.asarray(vehicle_data.get("nearest_dist", [])[:limit], dtype=np.float32)[mask]
    gt_nearest = _gt_nearest_distances(gt_data_dict, ego_id, limit)[mask]
    nearest_bins = np.arange(201, dtype=np.float32) * 0.5 * (100.0 / 40.0)

    lin_speed_jsd = _histogram_jsd(
        np.clip(sim_speeds, 0.0, 30.0),
        np.clip(gt_speeds, 0.0, 30.0),
        lin_speed_bins,
    )
    ang_speed_jsd = _histogram_jsd(
        np.clip(sim_ang_speeds, -50.0, 50.0),
        np.clip(gt_ang_speeds, -50.0, 50.0),
        ang_speed_bins,
    )
    accel_jsd = _histogram_jsd(sim_accels, gt_accels, accel_bins)
    nearest_dist_jsd = _histogram_jsd(
        np.clip(sim_nearest, 0.0, 40.0),
        np.clip(gt_nearest, 0.0, 40.0),
        nearest_bins,
    )

    return {
        "fde": float(np.linalg.norm(sim_positions[last_position] - gt_positions[last_position])),
        "ade": float(position_errors.mean()),
        "lin_speed_jsd": lin_speed_jsd,
        "ang_speed_jsd": ang_speed_jsd,
        "accel_jsd": accel_jsd,
        "nearest_dist_jsd": nearest_dist_jsd,
        "meta_jsd": float(
            np.mean([lin_speed_jsd, ang_speed_jsd, accel_jsd, nearest_dist_jsd])
        ),
    }
