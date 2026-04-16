"""Nocturne rollout logging and aggregate stats helpers."""

from __future__ import annotations

from collections import defaultdict
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


def build_nocturne_process_stats(
    *,
    infos: Sequence[Mapping[str, Any]] | None = None,
    venv: Any | None = None,
    tilting_mode: str = "per_vehicle",
    log_replay_complexity: bool = False,
) -> list[dict[str, Any]]:
    """Build per-process Nocturne logging rows."""
    if infos is None:
        try:
            infos = venv.get_complexity_info()
        except AttributeError:
            return []

    process_stats = []
    for process_idx, info in enumerate(infos):
        process_log = {"process_idx": process_idx}
        process_log.update(filter_nocturne_process_info(info))
        max_progress = process_log.pop("max_progress", None)
        if max_progress is not None:
            process_log["max_progress"] = max_progress
        opponent_vehicle_num = process_log.get("opponent_vehicle_num")
        if log_replay_complexity:
            process_log["max_progress"] = None
            process_log["plr_progress"] = max_progress
            process_log["opponent_vehicle_num"] = None
            process_log["plr_opponent_vehicle_num"] = opponent_vehicle_num
        else:
            process_log["plr_progress"] = None
            process_log["plr_opponent_vehicle_num"] = None
        tilting_columns = build_nocturne_tilting_columns(
            info,
            tilting_mode=tilting_mode,
        )
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

    if len(infos) == 0:
        return {}

    sums = defaultdict(float)
    counts = defaultdict(int)
    for info in infos:
        merged_info = filter_nocturne_process_info(info)
        merged_info.update(
            build_nocturne_tilting_columns(info, tilting_mode=tilting_mode)
        )
        for k, v in merged_info.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if k in ("opponent_k", "scenario_pool_size"):
                    continue
                if k == "max_progress":
                    continue
                if k == "seed":
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
