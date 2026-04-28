#!/usr/bin/env python3
"""Build pass/fail empirical distributions for solvability analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CTRLSIM_IPC_USE_SHM", "0")

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from evaluation.ctrlsim_evaluation_runner import (
    build_zero_action_batch,
    build_ctrlsim_evaluator,
    build_ctrlsim_external_teacher,
    run_batched_ctrlsim_step,
)
from evaluation.evaluation_common import compute_solved_flag


@dataclass(frozen=True)
class RtgWeights:
    """Weights used to combine the three mean RTG components."""

    goal: float = 1.0
    veh_veh: float = 1.0
    veh_edge: float = 1.0


@dataclass(frozen=True)
class LevelTask:
    """One flattened solvability evaluation task."""

    scenario_index: int
    scenario_id: str
    level_index_in_scenario: int
    global_level_index: int
    tilt_group: str
    level: ScenarioLevel


@dataclass
class EnvSlotState:
    """Track one active env slot in the flattened scheduler."""

    task: LevelTask | None
    rtg_records: list[np.ndarray]
    parking_level: ScenarioLevel | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for solvability distribution generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample scenario levels, evaluate CtrlSim solvability, and fit "
            "global pass/fail empirical distributions of mean_rtg_total."
        )
    )
    parser.add_argument("--scenario_index_path", type=str, required=True)
    parser.add_argument("--scenario_data_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument(
        "--vehicle_map_path",
        type=str,
        default="data/vehicle_map_valid.json",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--student_accel_discretization",
        type=int,
        default=20,
        help="Student acceleration discretization used by Nocturne-CtrlSim env.",
    )
    parser.add_argument(
        "--student_steer_discretization",
        type=int,
        default=50,
        help="Student steering discretization used by Nocturne-CtrlSim env.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. If omitted, a random seed is generated.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=90,
        help="Maximum episode steps.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel environments to run.",
    )
    parser.add_argument(
        "--levels_per_scenario",
        type=int,
        default=10,
        help="How many levels to sample for each scenario.",
    )
    parser.add_argument(
        "--progress_threshold",
        type=float,
        default=0.85,
        help=(
            "Pass criterion threshold: pass if max progress exceeds this value "
            "and no collision/offroad occurred."
        ),
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["global", "per_vehicle", "none"],
        default="global",
        help=(
            "Tilting mode for the environment. The script samples one level-wise "
            "tilt triple; in per_vehicle mode it repeats the same triple for all "
            "controlled opponents."
        ),
    )
    parser.add_argument(
        "--positive_ratio",
        type=float,
        default=0.5,
        help="Probability of sampling a positive level.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--show_vehicle_ids", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    parser.add_argument(
        "--action_repeat_frequency",
        type=int,
        default=2,
        help="Cycle length N for opponent action repeat.",
    )
    parser.add_argument(
        "--kl_loss_computation_frequency",
        type=int,
        default=2,
        help="Cycle length N for ego KL loss computation.",
    )
    parser.add_argument(
        "--sparse_inference_action_repeat",
        action="store_true",
        help="Repeat the previous action on the last step in each action-repeat cycle.",
    )
    parser.add_argument(
        "--inference_precision",
        type=str,
        choices=["fp32", "amp_fp16", "amp_bf16"],
        default="fp32",
        help="Inference precision for CtrlSim teacher inference.",
    )
    parser.add_argument(
        "--rtg_weight_goal",
        type=float,
        default=1.0,
        help="Weight for mean_rtg_goal in mean_rtg_total.",
    )
    parser.add_argument(
        "--rtg_weight_veh_veh",
        type=float,
        default=1.0,
        help="Weight for mean_rtg_veh_veh in mean_rtg_total.",
    )
    parser.add_argument(
        "--rtg_weight_veh_edge",
        type=float,
        default=1.0,
        help="Weight for mean_rtg_veh_edge in mean_rtg_total.",
    )
    parser.add_argument(
        "--fit_every_n_scenarios",
        type=int,
        default=100,
        help="Refit empirical distributions every N scenarios.",
    )
    parser.add_argument(
        "--num_distribution_bins",
        type=int,
        default=100,
        help="Number of bins used for histogram-based empirical distributions.",
    )
    return parser.parse_args()


def _load_scenario_ids(index_path: str) -> list[str]:
    """Load scenario IDs from a scenario index JSON."""
    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenario_ids = data.get("scenario_ids")
    if not isinstance(scenario_ids, list):
        raise ValueError(
            f"Invalid scenario index format in {index_path}: missing 'scenario_ids' list"
        )
    return [str(scenario_id) for scenario_id in scenario_ids]


def _sample_level_seed(rng: np.random.Generator) -> int:
    """Sample a non-negative int32-compatible level seed."""
    return int(rng.integers(0, np.iinfo(np.int32).max, endpoint=False))


def _sample_tilt_value(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample one inclusive integer tilt value."""
    return int(rng.integers(low, high + 1))


def _sample_level_tilts(
    rng: np.random.Generator,
    positive_ratio: float,
) -> tuple[str, int, int, int]:
    """Sample one level-wise tilt triple under the positive/negative split."""
    is_positive = bool(rng.random() < positive_ratio)
    tilt_group = "positive" if is_positive else "negative"
    low, high = (0, 25) if is_positive else (-25, 0)
    return (
        tilt_group,
        _sample_tilt_value(rng, low, high),
        _sample_tilt_value(rng, low, high),
        _sample_tilt_value(rng, low, high),
    )


def _build_level(
    scenario_id: str,
    seed: int,
    goal_tilt: int,
    veh_veh_tilt: int,
    veh_edge_tilt: int,
    tilting_mode: str,
    opponent_k: int,
) -> ScenarioLevel:
    """Build one scenario level from the sampled tilt triple."""
    per_vehicle_tilting: tuple[int, ...] = ()
    if tilting_mode == "per_vehicle":
        per_vehicle_tilting = tuple(
            value
            for _ in range(opponent_k)
            for value in (goal_tilt, veh_veh_tilt, veh_edge_tilt)
        )
    if tilting_mode == "none":
        goal_tilt = 0
        veh_veh_tilt = 0
        veh_edge_tilt = 0

    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=goal_tilt,
        veh_veh_tilt=veh_veh_tilt,
        veh_edge_tilt=veh_edge_tilt,
        per_vehicle_tilting=per_vehicle_tilting,
    )


def _build_task_queue(
    scenario_ids: Sequence[str],
    levels_per_scenario: int,
    tilting_mode: str,
    positive_ratio: float,
    rng: np.random.Generator,
    opponent_k: int,
) -> list[LevelTask]:
    """Flatten all `(scenario, level)` jobs into one global task list."""
    tasks: list[LevelTask] = []
    global_level_index = 0
    for scenario_index, scenario_id in enumerate(scenario_ids, start=1):
        for level_index_in_scenario in range(levels_per_scenario):
            global_level_index += 1
            tilt_group, goal_tilt, veh_veh_tilt, veh_edge_tilt = _sample_level_tilts(
                rng,
                positive_ratio,
            )
            level = _build_level(
                scenario_id=scenario_id,
                seed=_sample_level_seed(rng),
                goal_tilt=goal_tilt,
                veh_veh_tilt=veh_veh_tilt,
                veh_edge_tilt=veh_edge_tilt,
                tilting_mode=tilting_mode,
                opponent_k=opponent_k,
            )
            tasks.append(
                LevelTask(
                    scenario_index=scenario_index,
                    scenario_id=scenario_id,
                    level_index_in_scenario=level_index_in_scenario,
                    global_level_index=global_level_index,
                    tilt_group=tilt_group,
                    level=level,
                )
            )
    return tasks


def _mean_or_zero(values: list[np.ndarray]) -> np.ndarray:
    """Return the mean across rows or a zero 3-vector when empty."""
    if not values:
        return np.zeros(3, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    return array.mean(axis=0, dtype=np.float32)


def _compute_weighted_mean_rtg_total(
    mean_rtg: Sequence[float],
    weights: RtgWeights,
) -> float:
    """Combine the three mean RTG components into one scalar score."""
    return (
        float(weights.goal) * float(mean_rtg[0])
        + float(weights.veh_veh) * float(mean_rtg[1])
        + float(weights.veh_edge) * float(mean_rtg[2])
    )


def _level_metric_fieldnames() -> list[str]:
    """Return CSV field order for per-level metrics."""
    return [
        "scenario_index",
        "scenario_id",
        "level_index_in_scenario",
        "global_level_index",
        "seed",
        "tilt_group",
        "goal_tilt",
        "veh_veh_tilt",
        "veh_edge_tilt",
        "pass",
        "fail",
        "collision",
        "offroad",
        "max_progress",
        "goal_reached",
        "position_reached",
        "episode_length",
        "pass_step",
        "episode_return",
        "mean_rtg_goal",
        "mean_rtg_veh_veh",
        "mean_rtg_veh_edge",
        "mean_rtg_total",
    ]


def _distribution_edges(
    pass_samples: Sequence[float],
    fail_samples: Sequence[float],
    num_bins: int,
) -> np.ndarray:
    """Build shared histogram bin edges across both pass/fail samples."""
    all_values = np.asarray(list(pass_samples) + list(fail_samples), dtype=np.float64)
    if all_values.size == 0:
        return np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)

    value_min = float(all_values.min())
    value_max = float(all_values.max())
    if np.isclose(value_min, value_max):
        delta = 1.0 if np.isclose(value_min, 0.0) else abs(value_min) * 0.05
        value_min -= delta
        value_max += delta
    return np.linspace(value_min, value_max, num_bins + 1, dtype=np.float64)


def _quantile_or_empty(values: np.ndarray, q: float) -> float | str:
    """Return one quantile for non-empty samples, otherwise an empty string."""
    if values.size == 0:
        return ""
    return float(np.quantile(values, q))


def _distribution_rows_for_label(
    label: str,
    values: Sequence[float],
    edges: np.ndarray,
    scenario_count: int,
    level_count: int,
) -> list[dict[str, object]]:
    """Build histogram and empirical-CDF rows for one label."""
    value_array = np.asarray(values, dtype=np.float64)
    rows: list[dict[str, object]] = []
    bin_widths = np.diff(edges)
    if value_array.size == 0:
        for bin_index in range(len(edges) - 1):
            rows.append(
                {
                    "scenario_count": scenario_count,
                    "level_count": level_count,
                    "label": label,
                    "sample_count": 0,
                    "mean": "",
                    "std": "",
                    "q05": "",
                    "q25": "",
                    "q50": "",
                    "q75": "",
                    "q95": "",
                    "bin_index": bin_index,
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "bin_width": float(bin_widths[bin_index]),
                    "bin_count": 0,
                    "density": 0.0,
                    "cdf_right": 0.0,
                }
            )
        return rows

    counts, _ = np.histogram(value_array, bins=edges)
    sorted_values = np.sort(value_array)
    sample_count = int(value_array.size)
    mean = float(value_array.mean())
    std = float(value_array.std(ddof=0))
    q05 = _quantile_or_empty(value_array, 0.05)
    q25 = _quantile_or_empty(value_array, 0.25)
    q50 = _quantile_or_empty(value_array, 0.50)
    q75 = _quantile_or_empty(value_array, 0.75)
    q95 = _quantile_or_empty(value_array, 0.95)

    for bin_index, count in enumerate(counts):
        edge_right = float(edges[bin_index + 1])
        cdf_right = float(np.searchsorted(sorted_values, edge_right, side="right")) / float(sample_count)
        density = float(count) / float(sample_count * bin_widths[bin_index])
        rows.append(
            {
                "scenario_count": scenario_count,
                "level_count": level_count,
                "label": label,
                "sample_count": sample_count,
                "mean": mean,
                "std": std,
                "q05": q05,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q95": q95,
                "bin_index": bin_index,
                "bin_left": float(edges[bin_index]),
                "bin_right": edge_right,
                "bin_width": float(bin_widths[bin_index]),
                "bin_count": int(count),
                "density": density,
                "cdf_right": cdf_right,
            }
        )
    return rows


def _write_distribution_fit(
    output_path: Path,
    pass_samples: Sequence[float],
    fail_samples: Sequence[float],
    num_bins: int,
    scenario_count: int,
    level_count: int,
) -> None:
    """Write histogram + empirical-CDF rows for the current global samples."""
    edges = _distribution_edges(pass_samples, fail_samples, num_bins)
    rows = _distribution_rows_for_label(
        label="pass",
        values=pass_samples,
        edges=edges,
        scenario_count=scenario_count,
        level_count=level_count,
    ) + _distribution_rows_for_label(
        label="fail",
        values=fail_samples,
        edges=edges,
        scenario_count=scenario_count,
        level_count=level_count,
    )

    fieldnames = [
        "scenario_count",
        "level_count",
        "label",
        "sample_count",
        "mean",
        "std",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_width",
        "bin_count",
        "density",
        "cdf_right",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_distribution_summary(
    output_path: Path,
    pass_samples: Sequence[float],
    fail_samples: Sequence[float],
    scenario_count: int,
    level_count: int,
) -> None:
    """Write one-row-per-label summary statistics for final samples."""
    fieldnames = [
        "scenario_count",
        "level_count",
        "label",
        "sample_count",
        "mean",
        "std",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    ]
    rows = []
    for label, values in (("pass", pass_samples), ("fail", fail_samples)):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "scenario_count": scenario_count,
                "level_count": level_count,
                "label": label,
                "sample_count": int(array.size),
                "mean": "" if array.size == 0 else float(array.mean()),
                "std": "" if array.size == 0 else float(array.std(ddof=0)),
                "q05": _quantile_or_empty(array, 0.05),
                "q25": _quantile_or_empty(array, 0.25),
                "q50": _quantile_or_empty(array, 0.50),
                "q75": _quantile_or_empty(array, 0.75),
                "q95": _quantile_or_empty(array, 0.95),
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _output_dir(base_dir: str, xpid: str) -> Path:
    """Resolve and create the output directory for one run."""
    path = Path(base_dir).expanduser().resolve() / xpid
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_task_metrics(
    task: LevelTask,
    rtg_records: list[np.ndarray],
    final_info: dict[str, object],
    progress_threshold: float,
    weights: RtgWeights,
) -> dict[str, object]:
    """Build one level row once an env slot finishes its current task."""
    mean_rtg = _mean_or_zero(rtg_records)
    mean_rtg_total = _compute_weighted_mean_rtg_total(mean_rtg, weights)
    collision = float(final_info.get("collision_occurred", final_info.get("collision", 0.0)))
    offroad = float(final_info.get("offroad_occurred", final_info.get("offroad", 0.0)))
    max_progress = float(final_info.get("max_progress", final_info.get("progress", 0.0)))
    passed = bool(
        compute_solved_flag(
            progress=max_progress,
            collision=collision,
            offroad=offroad,
            progress_threshold=progress_threshold,
        )
    )
    episode_length = int(final_info.get("episode_steps", final_info.get("step", 0)))
    episode_return = float(final_info.get("episode", {}).get("r", final_info.get("episode_reward", 0.0)))
    return {
        "scenario_index": task.scenario_index,
        "scenario_id": task.scenario_id,
        "level_index_in_scenario": task.level_index_in_scenario,
        "global_level_index": task.global_level_index,
        "seed": int(task.level.seed),
        "tilt_group": task.tilt_group,
        "goal_tilt": int(task.level.goal_tilt),
        "veh_veh_tilt": int(task.level.veh_veh_tilt),
        "veh_edge_tilt": int(task.level.veh_edge_tilt),
        "pass": 1.0 if passed else 0.0,
        "fail": 0.0 if passed else 1.0,
        "collision": collision,
        "offroad": offroad,
        "max_progress": max_progress,
        "goal_reached": float(final_info.get("goal_reached_occurred", 0.0)),
        "position_reached": float(final_info.get("position_reached_occurred", 0.0)),
        "episode_length": episode_length,
        "pass_step": episode_length if passed else -1,
        "episode_return": episode_return,
        "mean_rtg_goal": float(mean_rtg[0]),
        "mean_rtg_veh_veh": float(mean_rtg[1]),
        "mean_rtg_veh_edge": float(mean_rtg[2]),
        "mean_rtg_total": float(mean_rtg_total),
    }


def _assign_tasks_to_slots(
    *,
    venv,
    slot_states: list[EnvSlotState | None],
    tasks: Sequence[LevelTask],
    next_task_index: int,
    slot_indices: Sequence[int],
) -> int:
    """Reset selected env slots to new task levels and seed their local state."""
    if not slot_indices:
        return next_task_index
    levels = []
    actual_indices = []
    for slot_index in slot_indices:
        if next_task_index >= len(tasks):
            break
        task = tasks[next_task_index]
        next_task_index += 1
        slot_states[slot_index] = EnvSlotState(task=task, rtg_records=[])
        levels.append(task.level)
        actual_indices.append(slot_index)
    if actual_indices:
        venv.reset_to_level_indices(levels, actual_indices)
    return next_task_index


def _assign_parking_levels(
    *,
    venv,
    slot_states: list[EnvSlotState | None],
    slot_indices: Sequence[int],
) -> None:
    """Keep idle env slots on ignored parking levels until real work finishes."""
    if not slot_indices:
        return
    levels = []
    actual_indices = []
    for slot_index in slot_indices:
        state = slot_states[slot_index]
        if state is None or state.parking_level is None:
            continue
        levels.append(state.parking_level)
        actual_indices.append(slot_index)
    if actual_indices:
        venv.reset_to_level_indices(levels, actual_indices)


def _count_active_real_slots(slot_states: Sequence[EnvSlotState | None]) -> int:
    """Return how many env slots currently carry real tasks."""
    return sum(
        1
        for state in slot_states
        if state is not None and state.task is not None
    )


def main() -> None:
    """Run solvability sampling and empirical-distribution fitting."""
    args = parse_args()
    if not (0.0 <= args.positive_ratio <= 1.0):
        raise ValueError("--positive_ratio must be in [0, 1]")
    if args.levels_per_scenario <= 0:
        raise ValueError("--levels_per_scenario must be positive")
    if args.num_processes <= 0:
        raise ValueError("--num_processes must be positive")
    if args.fit_every_n_scenarios <= 0:
        raise ValueError("--fit_every_n_scenarios must be positive")
    if args.num_distribution_bins <= 0:
        raise ValueError("--num_distribution_bins must be positive")
    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")

    base_seed = (
        int(args.seed)
        if args.seed is not None
        else int.from_bytes(os.urandom(4), byteorder="little")
    )
    rng = np.random.default_rng(base_seed)
    scenario_ids = _load_scenario_ids(args.scenario_index_path)
    output_dir = _output_dir(args.output_dir, args.xpid)
    weights = RtgWeights(
        goal=float(args.rtg_weight_goal),
        veh_veh=float(args.rtg_weight_veh_veh),
        veh_edge=float(args.rtg_weight_veh_edge),
    )

    print(f"Loaded {len(scenario_ids)} scenarios from: {args.scenario_index_path}")
    print(f"Levels per scenario: {args.levels_per_scenario}")
    print(f"Positive ratio: {args.positive_ratio}")
    print(f"Parallel envs: {args.num_processes}")
    print(
        "RTG weights: "
        f"goal={weights.goal}, veh_veh={weights.veh_veh}, veh_edge={weights.veh_edge}"
    )
    print(f"Base seed: {base_seed}")
    print(f"Output dir: {output_dir}")
    print("CTRLSIM_IPC_USE_SHM=0 for this run to avoid shared-memory tracker warnings.")

    total_tasks = len(scenario_ids) * int(args.levels_per_scenario)
    if total_tasks == 0:
        raise ValueError("No solvability tasks were generated.")
    num_processes = min(int(args.num_processes), total_tasks)

    evaluator = build_ctrlsim_evaluator(
        args,
        base_seed=base_seed,
        num_processes=num_processes,
        tilt_range=(-25.0, 25.0),
        collect_ego_ctrlsim_rtg=True,
    )
    venv = evaluator.venv["Nocturne-CtrlSim-v0"]
    venv.reset_random()
    external_teacher = build_ctrlsim_external_teacher(args, base_seed=base_seed)
    tasks = _build_task_queue(
        scenario_ids=scenario_ids,
        levels_per_scenario=int(args.levels_per_scenario),
        tilting_mode=args.tilting_mode,
        positive_ratio=float(args.positive_ratio),
        rng=rng,
        opponent_k=7,
    )
    action_batch = build_zero_action_batch(venv, num_processes)
    slot_states: list[EnvSlotState | None] = [None for _ in range(num_processes)]
    completed_level_count = 0
    completed_by_scenario = [0 for _ in range(len(scenario_ids) + 1)]
    scenario_pass_samples = [[] for _ in range(len(scenario_ids) + 1)]
    scenario_fail_samples = [[] for _ in range(len(scenario_ids) + 1)]
    pass_samples_emitted: list[float] = []
    fail_samples_emitted: list[float] = []
    contiguous_completed_scenarios = 0
    aggregated_until_scenario = 0
    next_fit_scenario = int(args.fit_every_n_scenarios)
    next_task_index = _assign_tasks_to_slots(
        venv=venv,
        slot_states=slot_states,
        tasks=tasks,
        next_task_index=0,
        slot_indices=list(range(num_processes)),
    )

    level_metrics_path = output_dir / "level_metrics.csv"
    progress_bar = tqdm(total=total_tasks, disable=not args.verbose, desc="Levels")
    try:
        with open(level_metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_level_metric_fieldnames())
            writer.writeheader()

            while _count_active_real_slots(slot_states) > 0:
                _, _, dones, infos = run_batched_ctrlsim_step(
                    venv=venv,
                    action=action_batch,
                    external_teacher=external_teacher,
                    reset_random=False,
                    auto_reset_on_done=False,
                    collect_ego_ctrlsim_rtg=True,
                )

                real_reset_indices = []
                real_reset_tasks = []
                parking_reset_indices = []
                for slot_index, (done, info) in enumerate(zip(dones, infos)):
                    state = slot_states[slot_index]
                    if state is None:
                        continue
                    rtg = info.get("ego_ctrlsim_pred_rtg")
                    if state.task is not None and rtg is not None:
                        state.rtg_records.append(
                            np.asarray(rtg, dtype=np.float32).reshape(3,)
                        )
                    if not done:
                        continue

                    if state.task is not None:
                        metrics = _finalize_task_metrics(
                            task=state.task,
                            rtg_records=state.rtg_records,
                            final_info=dict(info),
                            progress_threshold=float(args.progress_threshold),
                            weights=weights,
                        )
                        writer.writerow(metrics)
                        handle.flush()
                        completed_level_count += 1
                        progress_bar.update(1)

                        scenario_idx = int(metrics["scenario_index"])
                        completed_by_scenario[scenario_idx] += 1
                        if float(metrics["pass"]) > 0.0:
                            scenario_pass_samples[scenario_idx].append(
                                float(metrics["mean_rtg_total"])
                            )
                        else:
                            scenario_fail_samples[scenario_idx].append(
                                float(metrics["mean_rtg_total"])
                            )

                        state.parking_level = state.task.level
                        state.task = None
                        state.rtg_records = []

                        if next_task_index < len(tasks):
                            real_reset_indices.append(slot_index)
                            real_reset_tasks.append(tasks[next_task_index])
                            next_task_index += 1
                        elif _count_active_real_slots(slot_states) > 0:
                            parking_reset_indices.append(slot_index)
                    else:
                        if _count_active_real_slots(slot_states) > 0:
                            parking_reset_indices.append(slot_index)
                        else:
                            slot_states[slot_index] = None

                if real_reset_indices:
                    for slot_index, task in zip(real_reset_indices, real_reset_tasks):
                        slot_states[slot_index] = EnvSlotState(task=task, rtg_records=[])
                    venv.reset_to_level_indices(
                        [task.level for task in real_reset_tasks],
                        real_reset_indices,
                    )
                if parking_reset_indices:
                    _assign_parking_levels(
                        venv=venv,
                        slot_states=slot_states,
                        slot_indices=parking_reset_indices,
                    )

                while (
                    contiguous_completed_scenarios + 1 <= len(scenario_ids)
                    and completed_by_scenario[contiguous_completed_scenarios + 1]
                    >= int(args.levels_per_scenario)
                ):
                    contiguous_completed_scenarios += 1

                while next_fit_scenario <= contiguous_completed_scenarios:
                    for scenario_idx in range(
                        aggregated_until_scenario + 1,
                        next_fit_scenario + 1,
                    ):
                        pass_samples_emitted.extend(scenario_pass_samples[scenario_idx])
                        fail_samples_emitted.extend(scenario_fail_samples[scenario_idx])
                    aggregated_until_scenario = next_fit_scenario
                    checkpoint_path = (
                        output_dir
                        / f"distribution_fit_{next_fit_scenario:05d}_scenarios.csv"
                    )
                    _write_distribution_fit(
                        output_path=checkpoint_path,
                        pass_samples=pass_samples_emitted,
                        fail_samples=fail_samples_emitted,
                        num_bins=int(args.num_distribution_bins),
                        scenario_count=next_fit_scenario,
                        level_count=next_fit_scenario * int(args.levels_per_scenario),
                    )
                    next_fit_scenario += int(args.fit_every_n_scenarios)

            for scenario_idx in range(aggregated_until_scenario + 1, len(scenario_ids) + 1):
                pass_samples_emitted.extend(scenario_pass_samples[scenario_idx])
                fail_samples_emitted.extend(scenario_fail_samples[scenario_idx])

            final_fit_path = output_dir / "distribution_fit_final.csv"
            _write_distribution_fit(
                output_path=final_fit_path,
                pass_samples=pass_samples_emitted,
                fail_samples=fail_samples_emitted,
                num_bins=int(args.num_distribution_bins),
                scenario_count=len(scenario_ids),
                level_count=completed_level_count,
            )
            summary_path = output_dir / "distribution_summary.csv"
            _write_distribution_summary(
                output_path=summary_path,
                pass_samples=pass_samples_emitted,
                fail_samples=fail_samples_emitted,
                scenario_count=len(scenario_ids),
                level_count=completed_level_count,
            )

        print(f"Level metrics saved to: {level_metrics_path}")
        print(f"Final distribution fit saved to: {final_fit_path}")
        print(f"Distribution summary saved to: {summary_path}")
    finally:
        progress_bar.close()
        evaluator.close()


if __name__ == "__main__":
    main()
