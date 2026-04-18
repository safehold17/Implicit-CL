"""Metrics and CSV helpers shared by solvability evaluation scripts."""

from __future__ import annotations

import csv
import os
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.evaluation_common import extract_episode_metrics as extract_base_episode_metrics

SOLVABILITY_TILTING_FIELDS = (
    "opp0_goal_tilt",
    "opp0_veh_veh_tilt",
    "opp0_veh_edge_tilt",
    "opp1_goal_tilt",
    "opp1_veh_veh_tilt",
    "opp1_veh_edge_tilt",
    "opp2_goal_tilt",
    "opp2_veh_veh_tilt",
    "opp2_veh_edge_tilt",
    "opp3_goal_tilt",
    "opp3_veh_veh_tilt",
    "opp3_veh_edge_tilt",
    "opp4_goal_tilt",
    "opp4_veh_veh_tilt",
    "opp4_veh_edge_tilt",
    "opp5_goal_tilt",
    "opp5_veh_veh_tilt",
    "opp5_veh_edge_tilt",
    "opp6_goal_tilt",
    "opp6_veh_veh_tilt",
    "opp6_veh_edge_tilt",
    "veh_goal_avg",
    "veh_veh_avg",
    "veh_edge_avg",
    "ego_goal_tilt",
    "ego_veh_veh_tilt",
    "ego_veh_edge_tilt",
)

SOLVABILITY_METRIC_FIELDS = (
    "episode",
    "scenario_id",
    "seed",
    "test_returns",
    "solved",
    "collision",
    "goal_reached",
    "position_reached",
    "offroad",
    "progress",
    *SOLVABILITY_TILTING_FIELDS,
)


def build_solvability_tilting_columns(
    info: Mapping[str, Any],
    *,
    tilting_mode: str,
) -> dict[str, float]:
    """Extract solvability tilting columns using the legacy CSV semantics."""
    opp_count = 7
    tilts = []
    if tilting_mode == "per_vehicle":
        for index in range(opp_count):
            goal_tilt = float(info.get(f"per_vehicle_goal_tilt_{index}", 0.0))
            veh_veh_tilt = float(info.get(f"per_vehicle_veh_tilt_{index}", 0.0))
            veh_edge_tilt = float(info.get(f"per_vehicle_edge_tilt_{index}", 0.0))
            tilts.append((goal_tilt, veh_veh_tilt, veh_edge_tilt))
    elif tilting_mode == "global":
        goal_tilt = float(info.get("goal_tilt", 0.0))
        veh_veh_tilt = float(info.get("veh_veh_tilt", 0.0))
        veh_edge_tilt = float(info.get("veh_edge_tilt", 0.0))
        tilts = [(goal_tilt, veh_veh_tilt, veh_edge_tilt)] * opp_count
    else:
        tilts = [(0.0, 0.0, 0.0)] * opp_count

    goal_values = [goal_tilt for goal_tilt, _, _ in tilts if goal_tilt != 0.0]
    veh_veh_values = [veh_veh_tilt for _, veh_veh_tilt, _ in tilts if veh_veh_tilt != 0.0]
    veh_edge_values = [veh_edge_tilt for _, _, veh_edge_tilt in tilts if veh_edge_tilt != 0.0]

    columns = {}
    for index, (goal_tilt, veh_veh_tilt, veh_edge_tilt) in enumerate(tilts):
        columns[f"opp{index}_goal_tilt"] = goal_tilt
        columns[f"opp{index}_veh_veh_tilt"] = veh_veh_tilt
        columns[f"opp{index}_veh_edge_tilt"] = veh_edge_tilt
    columns["veh_goal_avg"] = (
        float(round(sum(goal_values) / len(goal_values), 2)) if goal_values else 0.0
    )
    columns["veh_veh_avg"] = (
        float(round(sum(veh_veh_values) / len(veh_veh_values), 2))
        if veh_veh_values
        else 0.0
    )
    columns["veh_edge_avg"] = (
        float(round(sum(veh_edge_values) / len(veh_edge_values), 2))
        if veh_edge_values
        else 0.0
    )
    columns["ego_goal_tilt"] = 0.0
    columns["ego_veh_veh_tilt"] = 0.0
    columns["ego_veh_edge_tilt"] = 0.0
    return columns


def extract_solvability_episode_metrics(
    info: Mapping[str, Any],
    *,
    progress_threshold: float,
    tilting_mode: str,
) -> dict[str, float | str]:
    """Build one solvability metrics row from completed episode info."""
    base_metrics = extract_base_episode_metrics(dict(info))
    collision = float(base_metrics["collision"])
    offroad = float(base_metrics["offroad"])
    progress = float(base_metrics["progress"])
    metrics = {
        "scenario_id": base_metrics["scenario_id"],
        "seed": info.get("seed", ""),
        "test_returns": float(base_metrics["total_episode_reward"]),
        "solved": (
            1.0
            if progress > float(progress_threshold)
            and collision == 0.0
            and offroad == 0.0
            else 0.0
        ),
        "collision": collision,
        "goal_reached": float(
            info.get("goal_reached_occurred", info.get("goal_reached", 0.0))
        ),
        "position_reached": float(base_metrics["position_reached"]),
        "offroad": offroad,
        "progress": progress,
    }
    metrics.update(
        build_solvability_tilting_columns(info, tilting_mode=tilting_mode)
    )
    return metrics


def _build_mean_metric_row(
    episode_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, float | str]:
    """Build the final mean row for solvability CSV output."""
    mean_row: dict[str, float | str] = {"episode": "avg", "scenario_id": "", "seed": ""}
    for field in SOLVABILITY_METRIC_FIELDS[3:]:
        values = [float(row[field]) for row in episode_metrics]
        mean_row[field] = f"{float(np.mean(values)):.2f}" if values else "0.00"
    return mean_row


def write_solvability_metrics_csv(
    output_dir: str,
    xpid: str,
    episode_metrics: Sequence[Mapping[str, Any]],
) -> str:
    """Write solvability metrics and a final mean row to CSV."""
    out_dir = os.path.join(output_dir, xpid)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOLVABILITY_METRIC_FIELDS)
        writer.writeheader()
        for index, metrics in enumerate(episode_metrics):
            row = {"episode": index}
            row.update(metrics)
            writer.writerow(row)
        if episode_metrics:
            writer.writerow(_build_mean_metric_row(episode_metrics))

    return csv_path
