#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from solvability.ctrlsim_evaluation_runner import (
    build_zero_action_batch,
    build_ctrlsim_evaluator,
    build_ctrlsim_external_teacher,
    run_batched_ctrlsim_step,
)
from solvability.solvability_common import (
    extract_solvability_episode_metrics,
    write_solvability_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval pipeline with CtrlSim ego + selectable opponent tilting."
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
        help="Base random seed. If omitted, a random seed is generated for each run.",
    )
    parser.add_argument("--num_steps", type=int, default=90)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--progress_threshold",
        type=float,
        default=0.85,
        help=(
            "Progress threshold used in solved metric: solved if max progress "
            "> threshold and no collision/offroad occurred."
        ),
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["global", "per_vehicle", "none"],
        default="per_vehicle",
    )
    parser.add_argument(
        "--tilt_range_min",
        type=float,
        default=-25.0,
        help="Minimum tilt sampling value for Nocturne.",
    )
    parser.add_argument(
        "--tilt_range_max",
        type=float,
        default=25.0,
        help="Maximum tilt sampling value for Nocturne.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--show_vehicle_ids", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    parser.add_argument(
        "--replay_metrics_csv",
        type=str,
        default=None,
        help="Replay the exact episode list stored in an existing solvability metrics CSV.",
    )
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
        help="Inference precision for ExternalTeacher.",
    )
    return parser.parse_args()


def _parse_int_metric(row: dict[str, str], key: str) -> int:
    """Parse one integer-valued metrics CSV field."""
    value = row.get(key, "")
    if value == "":
        return 0
    return int(round(float(value)))


def _build_level_from_metrics_row(
    row: dict[str, str],
    *,
    tilting_mode: str,
) -> ScenarioLevel:
    """Build one replayable ScenarioLevel from one solvability CSV row."""
    scenario_id = str(row["scenario_id"])
    seed = int(row["seed"])
    if tilting_mode == "global":
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=seed,
            goal_tilt=_parse_int_metric(row, "opp0_goal_tilt"),
            veh_veh_tilt=_parse_int_metric(row, "opp0_veh_veh_tilt"),
            veh_edge_tilt=_parse_int_metric(row, "opp0_veh_edge_tilt"),
        )
    if tilting_mode == "per_vehicle":
        per_vehicle_tilting = tuple(
            _parse_int_metric(row, field_name)
            for opponent_index in range(7)
            for field_name in (
                f"opp{opponent_index}_goal_tilt",
                f"opp{opponent_index}_veh_veh_tilt",
                f"opp{opponent_index}_veh_edge_tilt",
            )
        )
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=seed,
            goal_tilt=0,
            veh_veh_tilt=0,
            veh_edge_tilt=0,
            per_vehicle_tilting=per_vehicle_tilting,
        )
    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=0,
        veh_veh_tilt=0,
        veh_edge_tilt=0,
    )


def load_fixed_levels_from_metrics_csv(
    metrics_csv: str,
    *,
    tilting_mode: str,
) -> list[ScenarioLevel]:
    """Load replay levels from a solvability metrics CSV in row order."""
    levels: list[ScenarioLevel] = []
    with open(metrics_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("episode") == "avg" or not row.get("scenario_id"):
                continue
            levels.append(
                _build_level_from_metrics_row(row, tilting_mode=tilting_mode)
            )
    return levels


def _reset_slots_to_levels(
    *,
    venv,
    levels: Sequence[ScenarioLevel],
    indices: Sequence[int],
) -> None:
    """Reset selected vectorized env slots to explicit levels."""
    if not levels:
        return
    venv.reset_to_level_indices(list(levels), list(indices))


@dataclass
class _FixedReplaySlotScheduler:
    """Assign fixed replay levels to vector-env slots in CSV row order."""

    levels: Sequence[ScenarioLevel]
    num_slots: int
    active_levels_by_slot: list[ScenarioLevel | None] = field(init=False)
    parking_levels_by_slot: list[ScenarioLevel | None] = field(init=False)
    active_slots: set[int] = field(init=False, default_factory=set)
    next_level_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize per-slot active and parking state."""
        self.active_levels_by_slot = [None] * self.num_slots
        self.parking_levels_by_slot = [None] * self.num_slots

    def initial_resets(
        self,
    ) -> tuple[list[ScenarioLevel], list[int], list[ScenarioLevel], list[int]]:
        """Return active and parking reset assignments for startup."""
        initial_count = min(self.num_slots, len(self.levels))
        active_levels = list(self.levels[:initial_count])
        active_indices = list(range(initial_count))
        for slot_index, level in enumerate(active_levels):
            self.active_levels_by_slot[slot_index] = level
            self.active_slots.add(slot_index)
        self.next_level_index = initial_count

        parking_count = self.num_slots - initial_count
        if parking_count <= 0:
            return active_levels, active_indices, [], []

        parking_level = self.levels[0]
        parking_indices = list(range(initial_count, self.num_slots))
        for slot_index in parking_indices:
            self.parking_levels_by_slot[slot_index] = parking_level
        return (
            active_levels,
            active_indices,
            [parking_level] * parking_count,
            parking_indices,
        )

    def advance_completed_slots(
        self,
        completed_slots: set[int],
    ) -> tuple[list[ScenarioLevel], list[int], list[ScenarioLevel], list[int]]:
        """Return level resets needed after the given slots complete."""
        new_levels: list[ScenarioLevel] = []
        new_indices: list[int] = []
        parking_levels: list[ScenarioLevel] = []
        parking_indices: list[int] = []
        for slot_index in sorted(completed_slots):
            if self.next_level_index < len(self.levels):
                level = self.levels[self.next_level_index]
                self.next_level_index += 1
                self.active_levels_by_slot[slot_index] = level
                self.active_slots.add(slot_index)
                self.parking_levels_by_slot[slot_index] = None
                new_levels.append(level)
                new_indices.append(slot_index)
                continue

            self.active_slots.discard(slot_index)
            parking_level = self.active_levels_by_slot[slot_index]
            if parking_level is not None:
                self.parking_levels_by_slot[slot_index] = parking_level
            if parking_level is None:
                parking_level = self.parking_levels_by_slot[slot_index]
            self.active_levels_by_slot[slot_index] = None
            if parking_level is None:
                continue
            parking_levels.append(parking_level)
            parking_indices.append(slot_index)
        return new_levels, new_indices, parking_levels, parking_indices


def evaluate_with_metrics(
    evaluator,
    show_progress,
    render,
    tilting_mode,
    progress_threshold,
    external_teacher=None,
    fixed_levels: Sequence[ScenarioLevel] | None = None,
):
    env_name = evaluator.env_names[0]
    venv = evaluator.venv[env_name]
    num_episodes = (
        len(fixed_levels) if fixed_levels is not None else evaluator.num_episodes
    )
    if num_episodes == 0:
        return []

    if fixed_levels is None and env_name.startswith("Nocturne") and hasattr(venv, "reset_random"):
        obs = venv.reset_random()
    else:
        obs = None
        if fixed_levels is None:
            obs = venv.reset()
        elif not hasattr(venv, "reset_to_level_indices"):
            raise ValueError("Fixed replay mode requires venv.reset_to_level_indices().")

    episode_metrics = []
    pbar = tqdm(total=num_episodes) if show_progress else None
    action_batch = build_zero_action_batch(venv, evaluator.num_processes)
    fixed_replay_scheduler = None

    if fixed_levels is not None:
        fixed_replay_scheduler = _FixedReplaySlotScheduler(
            levels=fixed_levels,
            num_slots=evaluator.num_processes,
        )
        (
            initial_levels,
            initial_indices,
            parking_levels,
            parking_indices,
        ) = fixed_replay_scheduler.initial_resets()
        _reset_slots_to_levels(
            venv=venv,
            levels=initial_levels,
            indices=initial_indices,
        )
        if parking_levels:
            _reset_slots_to_levels(
                venv=venv,
                levels=parking_levels,
                indices=parking_indices,
            )

    while len(episode_metrics) < num_episodes:
        if external_teacher is not None:
            obs, reward, done, infos = run_batched_ctrlsim_step(
                venv=venv,
                action=action_batch,
                external_teacher=external_teacher,
                reset_random=fixed_levels is None,
                auto_reset_on_done=fixed_levels is None,
                collect_ego_ctrlsim_rtg=False,
            )
        else:
            obs, reward, done, infos = venv.step(action_batch)

        completed_slots: set[int] = set()
        for slot_index, (slot_done, info) in enumerate(zip(done, infos)):
            if fixed_levels is None:
                if "episode" not in info.keys():
                    continue
                metrics = extract_solvability_episode_metrics(
                    info,
                    progress_threshold=progress_threshold,
                    tilting_mode=tilting_mode,
                )
                episode_metrics.append(metrics)
                if pbar:
                    pbar.update(1)
                if len(episode_metrics) >= num_episodes:
                    break
                continue

            if (
                fixed_replay_scheduler is not None
                and slot_index in fixed_replay_scheduler.active_slots
                and "episode" in info.keys()
            ):
                metrics = extract_solvability_episode_metrics(
                    info,
                    progress_threshold=progress_threshold,
                    tilting_mode=tilting_mode,
                )
                episode_metrics.append(metrics)
                if pbar:
                    pbar.update(1)
                if len(episode_metrics) >= num_episodes:
                    completed_slots.add(slot_index)
                    break

            if slot_done:
                completed_slots.add(slot_index)

        if fixed_replay_scheduler is not None and len(episode_metrics) < num_episodes:
            (
                new_levels,
                new_indices,
                parking_levels,
                parking_indices,
            ) = fixed_replay_scheduler.advance_completed_slots(completed_slots)
            _reset_slots_to_levels(
                venv=venv,
                levels=new_levels,
                indices=new_indices,
            )
            if fixed_replay_scheduler.active_slots and parking_levels:
                _reset_slots_to_levels(
                    venv=venv,
                    levels=parking_levels,
                    indices=parking_indices,
                )

        if render:
            venv.render_to_screen()

    if pbar:
        pbar.close()

    return episode_metrics


def main() -> None:
    args = parse_args()
    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), byteorder="little")
    tilt_range = (float(args.tilt_range_min), float(args.tilt_range_max))
    print(f"Tilting mode: {args.tilting_mode}")
    print(f"Tilt range: [{tilt_range[0]}, {tilt_range[1]}]")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Base seed: {base_seed}")

    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")

    video_dir = os.path.join(args.output_dir, args.xpid)
    if args.record_video:
        print(f"Video output dir: {video_dir}")

    fixed_levels = None
    if args.replay_metrics_csv:
        fixed_levels = load_fixed_levels_from_metrics_csv(
            args.replay_metrics_csv,
            tilting_mode=args.tilting_mode,
        )
        if not fixed_levels:
            raise ValueError(
                f"No replayable episodes found in metrics CSV: {args.replay_metrics_csv}"
            )
        args.num_episodes = len(fixed_levels)
        print(f"Replay metrics CSV: {args.replay_metrics_csv}")
        print(f"Replay episodes: {len(fixed_levels)}")

    evaluator = build_ctrlsim_evaluator(
        args,
        base_seed=base_seed,
        num_processes=args.num_processes,
        tilt_range=tilt_range,
        collect_ego_ctrlsim_rtg=False,
    )

    external_teacher = build_ctrlsim_external_teacher(args, base_seed=base_seed)

    episode_metrics = evaluate_with_metrics(
        evaluator,
        show_progress=args.verbose,
        render=args.render,
        tilting_mode=args.tilting_mode,
        progress_threshold=args.progress_threshold,
        external_teacher=external_teacher,
        fixed_levels=fixed_levels,
    )
    csv_path = write_solvability_metrics_csv(args.output_dir, args.xpid, episode_metrics)
    print(f"Metrics saved to: {csv_path}")

    evaluator.close()


if __name__ == "__main__":
    main()
