"""Evaluate one scenario with student and CtrlSim models and record both videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from evaluation import ego_policy_evaluation
from evaluation.ctrlsim_evaluation_runner import (
    build_ctrlsim_evaluator,
    build_ctrlsim_external_teacher,
    build_zero_action_batch,
    run_batched_ctrlsim_step,
)
from evaluation.evaluation_common import (
    compute_solved_flag,
    extract_ctrlsim_episode_metrics,
)
from util import DotDict

SUMMARY_FIELDS = (
    "model_type",
    "scenario_id",
    "seed",
    "goal_tilt",
    "veh_veh_tilt",
    "veh_edge_tilt",
    "collision",
    "offroad",
    "position_reached",
    "progress",
    "solved",
    "total_episode_reward",
)


@dataclass(frozen=True)
class OutputPaths:
    """Resolved artifact paths for one dual-model evaluation run."""

    run_dir: Path
    summary_csv_path: Path
    student_video_dir: Path
    student_video_path: Path
    ctrlsim_video_dir: Path
    ctrlsim_video_path: Path


@dataclass(frozen=True)
class StudentCheckpointTarget:
    """Base directory and checkpoint stem expected by ego_policy_evaluation."""

    base_path: str
    model_tar: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one-scenario dual-model evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one explicit scenario level with one student model and one "
            "CtrlSim model, and save both rollout videos."
        )
    )
    parser.add_argument("--scenario_id", type=str, required=True)
    parser.add_argument("--scenario_index_path", type=str, required=True)
    parser.add_argument("--scenario_data_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--vehicle_map_path", type=str, required=True)
    parser.add_argument("--student_checkpoint_path", type=str, required=True)
    parser.add_argument("--ctrlsim_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--goal_tilt", type=int, default=0)
    parser.add_argument("--veh_veh_tilt", type=int, default=0)
    parser.add_argument("--veh_edge_tilt", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=90)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--student_accel_discretization", type=int, default=20)
    parser.add_argument("--student_steer_discretization", type=int, default=50)
    parser.add_argument("--action_repeat_frequency", type=int, default=2)
    parser.add_argument("--kl_loss_computation_frequency", type=int, default=2)
    parser.add_argument("--sparse_inference_action_repeat", action="store_true")
    parser.add_argument(
        "--inference_precision",
        type=str,
        choices=["fp32", "amp_fp16", "amp_bf16"],
        default="fp32",
    )
    parser.add_argument("--show_vehicle_ids", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_scenario_ids(scenario_index_path: str) -> list[str]:
    """Load ordered scenario IDs from one scenario-index JSON file."""
    with open(scenario_index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        scenario_ids = payload.get("scenario_ids")
    else:
        scenario_ids = payload
    if not isinstance(scenario_ids, list):
        raise ValueError(
            f"Invalid scenario index format in {scenario_index_path}: missing 'scenario_ids' list"
        )
    return [str(scenario_id) for scenario_id in scenario_ids]


def validate_scenario_id(*, scenario_id: str, scenario_index_path: str) -> None:
    """Raise when the requested scenario is absent from the scenario index."""
    scenario_ids = _load_scenario_ids(scenario_index_path)
    if scenario_id not in scenario_ids:
        raise ValueError(
            f"Scenario '{scenario_id}' is not present in scenario index: {scenario_index_path}"
        )


def validate_num_steps(*, num_steps: int) -> None:
    """Reject any rollout length other than the fixed 90-step protocol."""
    if int(num_steps) != 90:
        raise ValueError(
            "single_scenario_dual_model_video.py only supports num_steps=90, "
            f"got num_steps={num_steps}."
        )


def build_level(
    *,
    scenario_id: str,
    seed: int,
    goal_tilt: int,
    veh_veh_tilt: int,
    veh_edge_tilt: int,
) -> ScenarioLevel:
    """Build one explicit global-tilt scenario level."""
    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=goal_tilt,
        veh_veh_tilt=veh_veh_tilt,
        veh_edge_tilt=veh_edge_tilt,
    )


def build_output_paths(
    *,
    output_dir: str,
    xpid: str,
    scenario_id: str,
    seed: int,
) -> OutputPaths:
    """Return the resolved output layout for student/CtrlSim artifacts."""
    run_dir = Path(output_dir) / xpid
    student_video_dir = run_dir / "student"
    ctrlsim_video_dir = run_dir / "ctrlsim"
    video_suffix = f"{scenario_id}_seed{seed}.mp4"
    return OutputPaths(
        run_dir=run_dir,
        summary_csv_path=run_dir / "summary.csv",
        student_video_dir=student_video_dir,
        student_video_path=student_video_dir / f"student_{video_suffix}",
        ctrlsim_video_dir=ctrlsim_video_dir,
        ctrlsim_video_path=ctrlsim_video_dir / f"ctrlsim_{video_suffix}",
    )


def write_summary_csv(
    *,
    output_path: Path,
    rows: list[Mapping[str, Any]],
) -> None:
    """Write the minimal dual-model comparison summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _resolve_student_checkpoint_target(student_checkpoint_path: str) -> StudentCheckpointTarget:
    """Convert one full student checkpoint path into base_path + model_tar."""
    checkpoint_path = Path(student_checkpoint_path).expanduser().resolve()
    if checkpoint_path.suffix != ".tar":
        raise ValueError(
            "student_checkpoint_path must point to a .tar checkpoint file, "
            f"got: {student_checkpoint_path}"
        )
    return StudentCheckpointTarget(
        base_path=str(checkpoint_path.parent),
        model_tar=checkpoint_path.stem,
    )


def _build_student_cli_args(args: argparse.Namespace) -> DotDict:
    """Build the CLI-style args bundle expected by student evaluation helpers."""
    return DotDict(
        {
            "seed": args.seed,
            "num_processes": 1,
            "num_episodes": 1,
            "deterministic": True,
            "verbose": args.verbose,
            "record_video": True,
            "opponent_eval_mode": "replay",
            "tilting_mode": "none",
            "tilt_range_min": 0.0,
            "tilt_range_max": 0.0,
            "student_accel_discretization": args.student_accel_discretization,
            "student_steer_discretization": args.student_steer_discretization,
            "action_repeat_frequency": args.action_repeat_frequency,
            "kl_loss_computation_frequency": args.kl_loss_computation_frequency,
            "sparse_inference_action_repeat": args.sparse_inference_action_repeat,
            "inference_precision": args.inference_precision,
            "show_vehicle_ids": args.show_vehicle_ids,
            "scenario_index_path": args.scenario_index_path,
            "scenario_data_dir": args.scenario_data_dir,
            "preprocess_dir": args.preprocess_dir,
            "vehicle_map_path": args.vehicle_map_path,
            "max_episode_steps": args.num_steps,
            "opponent_checkpoint": args.ctrlsim_checkpoint_path,
            "enable_goal_tilt": False,
            "enable_veh_veh_tilt": False,
            "enable_veh_edge_tilt": False,
        }
    )


def _load_student_agent(
    *,
    student_checkpoint_path: str,
    env_name: str,
    device: str,
    cli_args: DotDict,
) -> tuple[Any, DotDict, Any, DotDict]:
    """Load one student evaluation agent from a full checkpoint path."""
    checkpoint_target = _resolve_student_checkpoint_target(student_checkpoint_path)
    agent, xpid_flags, nocturne_required, dummy_venv = (
        ego_policy_evaluation.load_agent_from_checkpoint(
            base_path=checkpoint_target.base_path,
            model_tar=checkpoint_target.model_tar,
            env_name=env_name,
            device=device,
            cli_args=cli_args,
        )
    )
    xpid_flags.update(cli_args)
    xpid_flags.update({"use_skip": False})
    return agent, DotDict(xpid_flags), dummy_venv, DotDict(nocturne_required)


def _init_recurrent_state(agent: Any, device: str, num_processes: int) -> Any:
    """Initialize recurrent hidden state storage for one evaluation batch."""
    hidden_state = torch.zeros(
        num_processes,
        agent.algo.actor_critic.recurrent_hidden_state_size,
        device=device,
    )
    if (
        agent.algo.actor_critic.is_recurrent
        and agent.algo.actor_critic.rnn.arch == "lstm"
    ):
        hidden_state = (hidden_state, torch.zeros_like(hidden_state))
    return hidden_state


def _extract_summary_row(
    *,
    model_type: str,
    level: ScenarioLevel,
    episode_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one episode metrics mapping into the summary CSV row format."""
    return {
        "model_type": model_type,
        "scenario_id": level.scenario_id,
        "seed": level.seed,
        "goal_tilt": level.goal_tilt,
        "veh_veh_tilt": level.veh_veh_tilt,
        "veh_edge_tilt": level.veh_edge_tilt,
        "collision": float(episode_metrics["collision"]),
        "offroad": float(episode_metrics["offroad"]),
        "position_reached": float(episode_metrics["position_reached"]),
        "progress": float(episode_metrics["progress"]),
        "solved": float(episode_metrics["solved"]),
        "total_episode_reward": float(episode_metrics["total_episode_reward"]),
    }


def _finalize_recorded_video(*, video_dir: Path, target_path: Path) -> None:
    """Rename the single recorded mp4 in one directory to the target filename."""
    mp4_paths = sorted(video_dir.glob("*.mp4"))
    if len(mp4_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one recorded mp4 in {video_dir}, found {len(mp4_paths)}"
        )
    source_path = mp4_paths[0]
    if source_path == target_path:
        return
    if target_path.exists():
        target_path.unlink()
    source_path.rename(target_path)


def run_student_rollout(
    *,
    args: argparse.Namespace,
    level: ScenarioLevel,
    output_paths: OutputPaths,
) -> dict[str, Any]:
    """Run one student-policy rollout on the explicit scenario level."""
    cli_args = _build_student_cli_args(args)
    agent, xpid_flags, dummy_venv, nocturne_required = _load_student_agent(
        student_checkpoint_path=args.student_checkpoint_path,
        env_name="Nocturne-CtrlSim-v0",
        device=args.device,
        cli_args=cli_args,
    )
    evaluator = None
    try:
        evaluator = ego_policy_evaluation.EgoReplayEvaluator(
            ["Nocturne-CtrlSim-v0"],
            num_processes=1,
            num_episodes=1,
            frame_stack=xpid_flags.frame_stack,
            grayscale=xpid_flags.grayscale,
            use_global_critic=xpid_flags.use_global_critic,
            video_dir=str(output_paths.student_video_dir),
            seed=args.seed,
            opponent_eval_mode="replay",
            **nocturne_required,
            record_video=True,
        )
        venv = evaluator.venv["Nocturne-CtrlSim-v0"]
        obs = venv.reset_to_level_batch([level])
        recurrent_hidden_states = _init_recurrent_state(agent, args.device, 1)
        masks = torch.ones(1, 1, device=args.device)

        while True:
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = agent.act(
                    obs,
                    recurrent_hidden_states,
                    masks,
                    deterministic=True,
                )

            action_np = agent.process_action(action.cpu().numpy())
            _prepared_batch = venv.step_prepare(action_np)
            obs, _reward, done, infos = venv.step_complete(
                [None],
                reset_random=False,
                auto_reset_on_done=False,
            )
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=args.device,
            )
            if not done[0]:
                continue

            info = infos[0]
            if "episode" not in info:
                raise RuntimeError("Student rollout finished without episode metrics.")
            metrics = extract_ctrlsim_episode_metrics(info)
            metrics["solved"] = compute_solved_flag(
                progress=float(metrics["progress"]),
                collision=float(metrics["collision"]),
                offroad=float(metrics["offroad"]),
                progress_threshold=ego_policy_evaluation.SOLVED_PROGRESS_THRESHOLD,
            )
            _finalize_recorded_video(
                video_dir=output_paths.student_video_dir,
                target_path=output_paths.student_video_path,
            )
            return metrics
    finally:
        if evaluator is not None:
            evaluator.close()
        dummy_venv.close()


def _build_ctrlsim_args(args: argparse.Namespace, output_paths: OutputPaths) -> argparse.Namespace:
    """Build the CLI-style args namespace expected by CtrlSim evaluation helpers."""
    return argparse.Namespace(
        scenario_index_path=args.scenario_index_path,
        scenario_data_dir=args.scenario_data_dir,
        preprocess_dir=args.preprocess_dir,
        vehicle_map_path=args.vehicle_map_path,
        checkpoint_path=args.ctrlsim_checkpoint_path,
        output_dir=str(output_paths.ctrlsim_video_dir),
        xpid="",
        device=args.device,
        seed=args.seed,
        num_processes=1,
        num_episodes=1,
        num_steps=args.num_steps,
        progress_threshold=ego_policy_evaluation.SOLVED_PROGRESS_THRESHOLD,
        student_accel_discretization=args.student_accel_discretization,
        student_steer_discretization=args.student_steer_discretization,
        action_repeat_frequency=args.action_repeat_frequency,
        kl_loss_computation_frequency=args.kl_loss_computation_frequency,
        sparse_inference_action_repeat=args.sparse_inference_action_repeat,
        inference_precision=args.inference_precision,
        tilting_mode="none",
        tilt_range_min=0.0,
        tilt_range_max=0.0,
        enable_goal_tilt=False,
        enable_veh_veh_tilt=False,
        enable_veh_edge_tilt=False,
        show_level_log=args.verbose,
        record_video=True,
        show_vehicle_ids=args.show_vehicle_ids,
    )


def run_ctrlsim_rollout(
    *,
    args: argparse.Namespace,
    level: ScenarioLevel,
    output_paths: OutputPaths,
) -> dict[str, Any]:
    """Run one CtrlSim-ego rollout on the explicit scenario level."""
    ctrlsim_args = _build_ctrlsim_args(args, output_paths)
    evaluator = build_ctrlsim_evaluator(
        ctrlsim_args,
        base_seed=args.seed,
        num_processes=1,
        tilt_range=(0.0, 0.0),
        collect_ego_ctrlsim_rtg=False,
    )
    external_teacher = build_ctrlsim_external_teacher(ctrlsim_args, base_seed=args.seed)
    try:
        venv = evaluator.venv["Nocturne-CtrlSim-v0"]
        _obs = venv.reset_to_level_batch([level])
        action_batch = build_zero_action_batch(venv, 1)

        while True:
            _obs, _reward, done, infos = run_batched_ctrlsim_step(
                venv=venv,
                action=action_batch,
                external_teacher=external_teacher,
                reset_random=False,
                auto_reset_on_done=False,
                collect_ego_ctrlsim_rtg=False,
            )
            if not done[0]:
                continue

            info = infos[0]
            if "episode" not in info:
                raise RuntimeError("CtrlSim rollout finished without episode metrics.")
            metrics = extract_ctrlsim_episode_metrics(
                info,
                offroad_progress_threshold=ctrlsim_args.progress_threshold,
            )
            metrics["solved"] = compute_solved_flag(
                progress=float(metrics["progress"]),
                collision=float(metrics["collision"]),
                offroad=float(metrics["offroad"]),
                progress_threshold=ctrlsim_args.progress_threshold,
            )
            _finalize_recorded_video(
                video_dir=output_paths.ctrlsim_video_dir,
                target_path=output_paths.ctrlsim_video_path,
            )
            return metrics
    finally:
        evaluator.close()


def _validate_checkpoint_paths(args: argparse.Namespace) -> None:
    """Validate that both model checkpoint paths exist."""
    missing_paths = [
        path
        for path in (args.student_checkpoint_path, args.ctrlsim_checkpoint_path)
        if not Path(path).expanduser().exists()
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(f"Missing checkpoint path(s): {missing}")


def main() -> None:
    """Run both single-scenario evaluations and write artifacts."""
    args = parse_args()
    validate_scenario_id(
        scenario_id=args.scenario_id,
        scenario_index_path=args.scenario_index_path,
    )
    validate_num_steps(num_steps=args.num_steps)
    _validate_checkpoint_paths(args)
    level = build_level(
        scenario_id=args.scenario_id,
        seed=args.seed,
        goal_tilt=args.goal_tilt,
        veh_veh_tilt=args.veh_veh_tilt,
        veh_edge_tilt=args.veh_edge_tilt,
    )
    output_paths = build_output_paths(
        output_dir=args.output_dir,
        xpid=args.xpid,
        scenario_id=args.scenario_id,
        seed=args.seed,
    )
    output_paths.student_video_dir.mkdir(parents=True, exist_ok=True)
    output_paths.ctrlsim_video_dir.mkdir(parents=True, exist_ok=True)

    student_metrics = run_student_rollout(
        args=args,
        level=level,
        output_paths=output_paths,
    )
    ctrlsim_metrics = run_ctrlsim_rollout(
        args=args,
        level=level,
        output_paths=output_paths,
    )

    rows = [
        _extract_summary_row(
            model_type="student",
            level=level,
            episode_metrics=student_metrics,
        ),
        _extract_summary_row(
            model_type="ctrlsim",
            level=level,
            episode_metrics=ctrlsim_metrics,
        ),
    ]
    write_summary_csv(output_path=output_paths.summary_csv_path, rows=rows)

    print(f"student_video={output_paths.student_video_path}")
    print(f"ctrlsim_video={output_paths.ctrlsim_video_path}")
    print(f"summary_csv={output_paths.summary_csv_path}")


if __name__ == "__main__":
    main()
