"""Evaluate a CtrlSim policy with replay or teacher-controlled opponents."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctrlsim_evaluation_metrics import (
    CTRLSIM_EGO_METRIC_FIELDS,
)
from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from evaluation.evaluation_common import (
    build_metrics_mean_row,
    compute_solved_flag,
    extract_ctrlsim_episode_metrics,
    resolve_csv_output_path,
    write_metrics_csv,
)
from evaluation.ctrlsim_evaluation_runner import (
    CtrlSimEvaluator,
    build_ctrlsim_external_teacher,
    build_zero_action_batch,
    run_batched_ctrlsim_step,
)
from util import str2bool


TEACHER_EVAL_METRIC_FIELDS = (
    "number",
    "scenario_id",
    "collision",
    "offroad",
    "position_reached",
    "progress",
    "solved",
    "total_episode_reward",
    *CTRLSIM_EGO_METRIC_FIELDS,
)

TEACHER_EVAL_SUMMARY_FIELDS = (
    "mode",
    "episodes",
    "solved_rate",
    "collision_rate",
    "offroad_rate",
    "position_reached_rate",
    "avg_progress",
    "avg_return",
    "avg_fde",
    "avg_ade",
    "avg_lin_speed_jsd",
    "avg_ang_speed_jsd",
    "avg_accel_jsd",
    "avg_nearest_dist_jsd",
    "avg_meta_jsd",
)


@dataclass(frozen=True)
class EpisodeTemplate:
    """Fixed scenario/seed pair used for deterministic evaluation episodes."""

    scenario_id: str
    level_seed: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CtrlSim policy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate CtrlSim teacher ego with replay or teacher opponents."
    )
    parser.add_argument("--scenario_index_path", type=str, required=True)
    parser.add_argument(
        "--scenario_csv_path",
        type=str,
        default=None,
        help="Optional CSV whose scenario_id rows define the exact evaluation order.",
    )
    parser.add_argument("--scenario_data_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--vehicle_map_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, default="ctrlsim-policy-eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=90)
    parser.add_argument(
        "--opponent_eval_mode",
        type=str,
        choices=["replay", "teacher", "joint_teacher", "both"],
        default="both",
        help="Opponent control protocol to evaluate.",
    )
    parser.add_argument(
        "--progress_threshold",
        type=float,
        default=0.85,
        help="Solved if progress exceeds this threshold with no collision.",
    )
    parser.add_argument("--student_accel_discretization", type=int, default=20)
    parser.add_argument("--student_steer_discretization", type=int, default=50)
    parser.add_argument("--action_repeat_frequency", type=int, default=2)
    parser.add_argument("--kl_loss_computation_frequency", type=int, default=2)
    parser.add_argument(
        "--sparse_inference_action_repeat",
        action="store_true",
        help="Repeat CtrlSim actions on sparse inference repeat steps.",
    )
    parser.add_argument(
        "--inference_precision",
        type=str,
        choices=["fp32", "amp_fp16", "amp_bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["none", "global", "per_vehicle"],
        default="none",
        help="Kept for level metadata; replay opponents are not teacher-tilted.",
    )
    parser.add_argument("--ego_goal_tilt", type=int, default=0)
    parser.add_argument("--ego_veh_veh_tilt", type=int, default=0)
    parser.add_argument("--ego_veh_edge_tilt", type=int, default=0)
    parser.add_argument("--tilt_range_min", type=float, default=0.0)
    parser.add_argument("--tilt_range_max", type=float, default=0.0)
    parser.add_argument(
        "--enable_goal_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether random opponent tilting may modify goal tilt.",
    )
    parser.add_argument(
        "--enable_veh_veh_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether random opponent tilting may modify vehicle-vehicle tilt.",
    )
    parser.add_argument(
        "--enable_veh_edge_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether random opponent tilting may modify vehicle-edge tilt.",
    )
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--show_vehicle_ids", action="store_true")
    return parser.parse_args()


def read_scenario_ids_from_csv(scenario_csv_path: str) -> list[str]:
    """Read ordered scenario IDs from an evaluation CSV file."""
    with open(scenario_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "scenario_id" not in fieldnames:
            raise ValueError(
                f"{scenario_csv_path} must contain a 'scenario_id' column"
            )
        return [str(row["scenario_id"]) for row in reader]


def build_episode_templates(
    *,
    scenario_ids: Sequence[str],
    seed: int,
) -> list[EpisodeTemplate]:
    """Generate deterministic per-row level seeds for a fixed scenario list."""
    rng = random.Random(seed)
    return [
        EpisodeTemplate(
            scenario_id=scenario_id,
            level_seed=rng.randrange(0, 2**31),
        )
        for scenario_id in scenario_ids
    ]


def build_scenario_level(template: EpisodeTemplate) -> ScenarioLevel:
    """Create the zero-tilt ScenarioLevel used by fixed-scenario evaluation."""
    return ScenarioLevel(
        scenario_id=template.scenario_id,
        seed=template.level_seed,
        goal_tilt=0,
        veh_veh_tilt=0,
        veh_edge_tilt=0,
    )


def evaluate_fixed_episode_templates(
    *,
    args: argparse.Namespace,
    venv,
    external_teacher,
) -> list[dict[str, float | str]]:
    """Evaluate explicit scenario templates without random environment resets."""
    episode_templates = build_episode_templates(
        scenario_ids=read_scenario_ids_from_csv(args.scenario_csv_path),
        seed=args.seed,
    )
    if not episode_templates:
        return []

    action_batch = build_zero_action_batch(venv, args.num_processes)
    episode_metrics: list[dict[str, float | str]] = []
    next_template_idx = 0
    active_templates: list[EpisodeTemplate | None] = []
    fallback_level = build_scenario_level(episode_templates[0])
    initial_levels = []

    for _ in range(args.num_processes):
        if next_template_idx < len(episode_templates):
            template = episode_templates[next_template_idx]
            active_templates.append(template)
            initial_levels.append(build_scenario_level(template))
            next_template_idx += 1
        else:
            active_templates.append(None)
            initial_levels.append(fallback_level)

    venv.reset_to_level_batch(initial_levels)
    pbar = tqdm(total=len(episode_templates))

    while len(episode_metrics) < len(episode_templates):
        _, _, done, infos = run_batched_ctrlsim_step(
            venv=venv,
            action=action_batch,
            external_teacher=external_teacher,
            reset_random=False,
            auto_reset_on_done=False,
            collect_ego_ctrlsim_rtg=False,
        )
        reset_levels = []
        reset_indices = []
        for idx, info in enumerate(infos):
            if not done[idx]:
                continue
            template = active_templates[idx]
            if template is not None and "episode" in info:
                metrics = extract_ctrlsim_episode_metrics(
                    info,
                    offroad_progress_threshold=args.progress_threshold,
                )
                metrics["solved"] = compute_solved_flag(
                    progress=float(metrics["progress"]),
                    collision=float(metrics["collision"]),
                    offroad=float(metrics["offroad"]),
                    progress_threshold=args.progress_threshold,
                )
                episode_metrics.append(metrics)
                pbar.update(1)
                if len(episode_metrics) >= len(episode_templates):
                    break

            if next_template_idx < len(episode_templates):
                next_template = episode_templates[next_template_idx]
                active_templates[idx] = next_template
                reset_levels.append(build_scenario_level(next_template))
                next_template_idx += 1
            else:
                active_templates[idx] = None
                reset_levels.append(fallback_level)
            reset_indices.append(idx)

        if reset_indices and len(episode_metrics) < len(episode_templates):
            venv.reset_to_level_indices(reset_levels, reset_indices)

    pbar.close()
    return episode_metrics

def evaluate_teacher_mode(
    args: argparse.Namespace,
    *,
    opponent_mode: str,
) -> list[dict[str, float | str]]:
    """Evaluate one CtrlSim checkpoint under one opponent protocol."""
    teacher_control_mode = "joint" if opponent_mode == "joint_teacher" else "split"
    ego_tilt_override = None
    if teacher_control_mode == "joint":
        ego_tilt_override = (
            int(getattr(args, "ego_goal_tilt", 0)),
            int(getattr(args, "ego_veh_veh_tilt", 0)),
            int(getattr(args, "ego_veh_edge_tilt", 0)),
        )
    evaluator = CtrlSimEvaluator(
        env_names=["Nocturne-CtrlSim-v0"],
        num_processes=args.num_processes,
        num_episodes=args.num_episodes,
        device=args.device,
        seed=args.seed,
        scenario_index_path=args.scenario_index_path,
        opponent_checkpoint=args.checkpoint_path,
        scenario_data_dir=args.scenario_data_dir,
        preprocess_dir=args.preprocess_dir,
        vehicle_map_path=args.vehicle_map_path,
        max_episode_steps=args.num_steps,
        opponent_k=7,
        tilting_mode=args.tilting_mode,
        tilt_range=(float(args.tilt_range_min), float(args.tilt_range_max)),
        enable_goal_tilt=args.enable_goal_tilt,
        enable_veh_veh_tilt=args.enable_veh_veh_tilt,
        enable_veh_edge_tilt=args.enable_veh_edge_tilt,
        show_level_log=args.show_level_log,
        record_video=args.record_video,
        show_vehicle_ids=args.show_vehicle_ids,
        output_dir=args.output_dir,
        xpid=args.xpid,
        inference_precision=args.inference_precision,
        action_repeat_frequency=args.action_repeat_frequency,
        kl_loss_computation_frequency=args.kl_loss_computation_frequency,
        sparse_inference_action_repeat=args.sparse_inference_action_repeat,
        student_accel_discretization=args.student_accel_discretization,
        student_steer_discretization=args.student_steer_discretization,
        collect_ego_ctrlsim_rtg=False,
        opponent_runtime_mode=(
            "normal"
            if opponent_mode in {"teacher", "joint_teacher"}
            else "replay"
        ),
        teacher_control_mode=teacher_control_mode,
        ego_tilt_override=ego_tilt_override,
    )
    external_teacher = build_ctrlsim_external_teacher(args, base_seed=args.seed)
    try:
        env_name = evaluator.env_names[0]
        venv = evaluator.venv[env_name]
        if args.scenario_csv_path:
            return evaluate_fixed_episode_templates(
                args=args,
                venv=venv,
                external_teacher=external_teacher,
            )

        venv.reset_random()
        action_batch = build_zero_action_batch(venv, args.num_processes)
        episode_metrics: list[dict[str, float | str]] = []
        pbar = tqdm(total=args.num_episodes)

        while len(episode_metrics) < args.num_episodes:
            _, _, _, infos = run_batched_ctrlsim_step(
                venv=venv,
                action=action_batch,
                external_teacher=external_teacher,
                reset_random=True,
                auto_reset_on_done=True,
                collect_ego_ctrlsim_rtg=False,
            )
            for info in infos:
                if "episode" not in info:
                    continue
                metrics = extract_ctrlsim_episode_metrics(
                    info,
                    offroad_progress_threshold=args.progress_threshold,
                )
                metrics["solved"] = compute_solved_flag(
                    progress=float(metrics["progress"]),
                    collision=float(metrics["collision"]),
                    offroad=float(metrics["offroad"]),
                    progress_threshold=args.progress_threshold,
                )
                episode_metrics.append(metrics)
                if pbar is not None:
                    pbar.update(1)
                if len(episode_metrics) >= args.num_episodes:
                    break

        if pbar is not None:
            pbar.close()
        return episode_metrics
    finally:
        evaluator.close()


def main() -> None:
    """Run CtrlSim policy evaluation and write metrics CSV."""
    args = parse_args()
    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    opponent_modes = (
        ["replay", "teacher"]
        if args.opponent_eval_mode == "both"
        else [args.opponent_eval_mode]
    )
    for opponent_mode in opponent_modes:
        episode_metrics = evaluate_teacher_mode(args, opponent_mode=opponent_mode)
        output_path = resolve_csv_output_path(args.output_dir, f"{args.xpid}-{opponent_mode}")
        mean_row = None
        if episode_metrics:
            mean_row = build_metrics_mean_row(
                episode_metrics,
                TEACHER_EVAL_METRIC_FIELDS,
                label_field="number",
                label_value="mean",
                empty_fields=("scenario_id",),
                mean_fields=TEACHER_EVAL_METRIC_FIELDS[2:],
            )
        write_metrics_csv(
            output_path,
            TEACHER_EVAL_METRIC_FIELDS,
            episode_metrics,
            index_field="number",
            mean_row=mean_row,
        )
        summary_rows.append(
            {
                "mode": opponent_mode,
                "episodes": len(episode_metrics),
                "solved_rate": float(
                    np.mean([float(row["solved"]) for row in episode_metrics])
                ),
                "collision_rate": float(
                    np.mean([float(row["collision"]) for row in episode_metrics])
                ),
                "offroad_rate": float(
                    np.mean([float(row["offroad"]) for row in episode_metrics])
                ),
                "position_reached_rate": float(
                    np.mean([float(row["position_reached"]) for row in episode_metrics])
                ),
                "avg_progress": float(
                    np.mean([float(row["progress"]) for row in episode_metrics])
                ),
                "avg_return": float(
                    np.mean([float(row["total_episode_reward"]) for row in episode_metrics])
                ),
                "avg_fde": float(np.mean([float(row["fde"]) for row in episode_metrics])),
                "avg_ade": float(np.mean([float(row["ade"]) for row in episode_metrics])),
                "avg_lin_speed_jsd": float(
                    np.mean([float(row["lin_speed_jsd"]) for row in episode_metrics])
                ),
                "avg_ang_speed_jsd": float(
                    np.mean([float(row["ang_speed_jsd"]) for row in episode_metrics])
                ),
                "avg_accel_jsd": float(
                    np.mean([float(row["accel_jsd"]) for row in episode_metrics])
                ),
                "avg_nearest_dist_jsd": float(
                    np.mean([float(row["nearest_dist_jsd"]) for row in episode_metrics])
                ),
                "avg_meta_jsd": float(
                    np.mean([float(row["meta_jsd"]) for row in episode_metrics])
                ),
            }
        )
        print(f"{opponent_mode} metrics saved to: {output_path}")

    if len(summary_rows) > 1:
        summary_path = resolve_csv_output_path(args.output_dir, f"{args.xpid}-summary")
        with open(summary_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TEACHER_EVAL_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
