"""Nocturne rollout logging and aggregate stats helpers."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

import numpy as np


def build_nocturne_tilting_columns(
    info: Mapping[str, Any],
    *,
    tilting_mode: str = "per_vehicle",
) -> dict[str, float]:
    """Build per-vehicle and aggregate tilting columns from Nocturne info."""
    opp_count = max(0, int(info.get("opponent_k", 7)))

    tilts = []
    ego_goal_tilt = float(info.get("ego_goal_tilt", 0.0))
    ego_veh_veh_tilt = float(info.get("ego_veh_veh_tilt", 0.0))
    ego_veh_edge_tilt = float(info.get("ego_veh_edge_tilt", 0.0))

    if tilting_mode == "per_vehicle":
        for i in range(opp_count):
            g = float(info.get(f"per_vehicle_goal_tilt_{i}", 0.0))
            v = float(info.get(f"per_vehicle_veh_tilt_{i}", 0.0))
            e = float(info.get(f"per_vehicle_edge_tilt_{i}", 0.0))
            tilts.append((g, v, e))
    elif tilting_mode == "global":
        g = float(info.get("goal_tilt", 0.0))
        v = float(info.get("veh_veh_tilt", 0.0))
        e = float(info.get("veh_edge_tilt", 0.0))
        for _ in range(opp_count):
            tilts.append((g, v, e))
    else:
        for _ in range(opp_count):
            tilts.append((0.0, 0.0, 0.0))

    tilts_for_avg = list(tilts)
    tilts_for_avg.append((ego_goal_tilt, ego_veh_veh_tilt, ego_veh_edge_tilt))

    goal_sum = 0.0
    goal_valid = 0
    veh_veh_sum = 0.0
    veh_veh_valid = 0
    veh_edge_sum = 0.0
    veh_edge_valid = 0

    for g, v, e in tilts_for_avg:
        if g != 0.0:
            goal_valid += 1
            goal_sum += g
        if v != 0.0:
            veh_veh_valid += 1
            veh_veh_sum += v
        if e != 0.0:
            veh_edge_valid += 1
            veh_edge_sum += e

    columns = {}
    for i, (g, v, e) in enumerate(tilts):
        columns[f"opp{i}_goal_tilt"] = g
        columns[f"opp{i}_veh_veh_tilt"] = v
        columns[f"opp{i}_veh_edge_tilt"] = e

    columns["veh_goal_avg"] = (
        float(round(goal_sum / goal_valid, 2)) if goal_valid > 0 else 0.0
    )
    columns["veh_veh_avg"] = (
        float(round(veh_veh_sum / veh_veh_valid, 2))
        if veh_veh_valid > 0
        else 0.0
    )
    columns["veh_edge_avg"] = (
        float(round(veh_edge_sum / veh_edge_valid, 2))
        if veh_edge_valid > 0
        else 0.0
    )
    columns["ego_goal_tilt"] = ego_goal_tilt
    columns["ego_veh_veh_tilt"] = ego_veh_veh_tilt
    columns["ego_veh_edge_tilt"] = ego_veh_edge_tilt

    return columns


def filter_nocturne_process_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Remove Nocturne process fields that are expanded or logged separately."""
    filtered = {}
    for k, v in info.items():
        if k in ("opponent_k", "scenario_pool_size"):
            continue
        if k.startswith("per_vehicle_"):
            continue
        if k.endswith("_progress") and k not in ("max_progress", "plr_progress"):
            continue
        filtered[k] = v
    return filtered


def _prepare_nocturne_stats_row(
    info: Mapping[str, Any],
    *,
    tilting_mode: str,
) -> dict[str, Any]:
    """Prepare one Nocturne info row for aggregate stats and process logging."""
    stats_row = filter_nocturne_process_info(info)
    stats_row.update(build_nocturne_tilting_columns(info, tilting_mode=tilting_mode))
    return stats_row


def _resolve_nocturne_process_infos(
    *,
    infos: Sequence[Mapping[str, Any]] | None,
    infos_by_process: Sequence[Sequence[Mapping[str, Any]]] | None,
    venv: Any | None,
    tilting_mode: str,
) -> tuple[list[Mapping[str, Any]], bool]:
    """Resolve Nocturne process infos and whether they were pre-aggregated."""
    if infos_by_process is None and infos is None:
        try:
            infos = venv.get_complexity_info()
        except AttributeError:
            return [], False

    if infos_by_process is not None:
        return [
            _average_nocturne_infos(
                process_done_infos,
                tilting_mode=tilting_mode,
            )
            for process_done_infos in infos_by_process
        ], True

    return list(infos or []), False


def _average_nocturne_infos(
    infos: Sequence[Mapping[str, Any]],
    *,
    tilting_mode: str,
) -> dict[str, Any]:
    """Average completed episodes for one process into one stats row."""
    if not infos:
        return {}

    stats_rows = [
        _prepare_nocturne_stats_row(info, tilting_mode=tilting_mode)
        for info in infos
    ]
    averaged_info = {
        k: v
        for k, v in stats_rows[0].items()
        if k != "seed" and (not isinstance(v, (int, float)) or np.isnan(v))
    }
    averaged_info["seed"] = None
    sums = defaultdict(float)
    counts = defaultdict(int)

    for stats_row in stats_rows:
        for key, value in stats_row.items():
            if key == "seed":
                continue
            if isinstance(value, (int, float)) and not np.isnan(value):
                sums[key] += float(value)
                counts[key] += 1

    for key, total in sums.items():
        averaged_info[key] = total / counts[key]

    return averaged_info


def _pop_averaged_nocturne_tilting_columns(
    info: dict[str, Any],
) -> dict[str, Any]:
    """Pop only the derived Nocturne tilting columns from one averaged row."""
    return {
        key: info.pop(key)
        for key in list(info)
        if _is_averaged_nocturne_tilting_key(key)
    }


def _is_averaged_nocturne_tilting_key(key: str) -> bool:
    """Return whether one averaged key belongs to derived tilting columns."""
    tilt_keys = {
        "veh_goal_avg",
        "veh_veh_avg",
        "veh_edge_avg",
        "ego_goal_tilt",
        "ego_veh_veh_tilt",
        "ego_veh_edge_tilt",
    }
    if key in tilt_keys:
        return True
    return bool(
        re.fullmatch(r"opp\d+_(goal_tilt|veh_veh_tilt|veh_edge_tilt)", key)
    )


def build_nocturne_process_stats(
    *,
    infos: Sequence[Mapping[str, Any]] | None = None,
    infos_by_process: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    venv: Any | None = None,
    tilting_mode: str = "per_vehicle",
    log_replay_complexity: bool = False,
) -> list[dict[str, Any]]:
    """Build per-process Nocturne logging rows."""
    process_stats = []
    process_infos, infos_are_averaged = _resolve_nocturne_process_infos(
        infos=infos,
        infos_by_process=infos_by_process,
        venv=venv,
        tilting_mode=tilting_mode,
    )

    for process_idx, info in enumerate(process_infos):
        process_log = {"process_idx": process_idx}
        if infos_are_averaged:
            base_info = dict(info)
            tilting_columns = _pop_averaged_nocturne_tilting_columns(base_info)
        else:
            base_info = filter_nocturne_process_info(info)
            tilting_columns = build_nocturne_tilting_columns(
                info,
                tilting_mode=tilting_mode,
            )
        process_log.update(base_info)
        max_progress = base_info.get("max_progress")
        opponent_vehicle_num = process_log.get("opponent_vehicle_num")
        if log_replay_complexity:
            process_log["max_progress"] = None
            process_log["plr_progress"] = max_progress
            process_log["opponent_vehicle_num"] = None
            process_log["plr_opponent_vehicle_num"] = opponent_vehicle_num
        else:
            process_log["plr_progress"] = None
            process_log["plr_opponent_vehicle_num"] = None
        if log_replay_complexity:
            process_log.update({k: None for k in tilting_columns})
            process_log.update({f"plr_{k}": v for k, v in tilting_columns.items()})
        else:
            process_log.update(tilting_columns)
        process_stats.append(process_log)

    return process_stats


def compute_nocturne_env_stats(
    *,
    agent_info: Mapping[str, Any],
    infos: Sequence[Mapping[str, Any]] | None = None,
    venv: Any | None = None,
    tilting_mode: str = "per_vehicle",
) -> dict[str, float]:
    """Aggregate Nocturne process info into runner-level environment stats."""
    if infos is None:
        try:
            infos = venv.get_complexity_info()
        except AttributeError:
            return {}

    stats_rows = [
        _prepare_nocturne_stats_row(info, tilting_mode=tilting_mode)
        for info in infos
    ]

    if not stats_rows:
        return {}

    sums = defaultdict(float)
    counts = defaultdict(int)
    for stats_row in stats_rows:
        for k, v in stats_row.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k in {"opponent_k", "scenario_pool_size", "max_progress", "seed"}:
                    continue
                if k.endswith("_occurred"):
                    continue
                sums[k] += v
                counts[k] += 1

    stats = {}
    for k, total in sums.items():
        avg_value = total / counts[k]
        if k.startswith("opp") or k.startswith("veh_") or k.startswith("ego_"):
            stats[k] = avg_value
        else:
            stats["scenario_" + k] = avg_value

    if "episode_return" in agent_info:
        episode_returns = agent_info["episode_return"]
        if len(episode_returns) > 0:
            stats["mean_episode_return"] = float(np.mean(episode_returns))

    return stats
