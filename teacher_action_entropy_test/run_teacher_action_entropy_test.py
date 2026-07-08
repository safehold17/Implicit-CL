"""Evaluate teacher-driven ego entropy with replay-controlled surrounding vehicles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ctrlsim_evaluation_runner import (  # noqa: E402
    CtrlSimEvaluator,
    build_ctrlsim_external_teacher,
    build_zero_action_batch,
)
from evaluation.ctrlsim_policy_evaluation import (  # noqa: E402
    build_episode_templates,
    build_scenario_level,
    read_scenario_ids_from_csv,
)
from evaluation.evaluation_common import (  # noqa: E402
    build_metrics_mean_row,
    compute_solved_flag,
    extract_ctrlsim_episode_metrics,
    resolve_csv_output_path,
    write_metrics_csv,
)
from util.clearml import upload_clearml_artifact  # noqa: E402
from util import str2bool  # noqa: E402


ACTION_ENTROPY_FIELDS = (
    "action_entropy",
    "action_entropy_std",
    "action_entropy_num_steps",
)

TEACHER_ACTION_ENTROPY_FIELDS = (
    "number",
    "scenario_id",
    "collision",
    "offroad",
    "progress",
    "solved",
    "ade",
    "fde",
    "meta_jsd",
    *ACTION_ENTROPY_FIELDS,
)


@dataclass
class ActiveEpisode:
    """Track one live environment slot's entropy history."""

    entropies: list[float]


@dataclass
class ArtifactBatchWriter:
    """Write per-episode metrics to numbered CSV batches and optionally upload them."""

    output_dir: Path
    clearml_task: Any
    chunk_size: int = 1000
    _buffer: list[dict[str, float | str]] | None = None
    _batch_index: int = 0
    _written_paths: list[Path] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable batch state."""
        self._buffer = []
        self._written_paths = []

    def add_row(self, row: dict[str, float | str]) -> None:
        """Append one row and flush immediately when the chunk is full."""
        self._buffer.append(row)
        if len(self._buffer) >= self.chunk_size:
            self._flush_current_buffer()

    def finalize(self) -> list[Path]:
        """Flush any trailing partial batch and return all written CSV paths."""
        if self._buffer:
            self._flush_current_buffer()
        return list(self._written_paths)

    def _flush_current_buffer(self) -> None:
        """Write and upload the current buffer as one numbered CSV artifact."""
        if not self._buffer:
            return

        self._batch_index += 1
        output_path = self.output_dir / f"action-entropy-{self._batch_index}.csv"
        mean_row = build_metrics_mean_row(
            self._buffer,
            TEACHER_ACTION_ENTROPY_FIELDS,
            label_field="number",
            label_value="mean",
            empty_fields=("scenario_id",),
            mean_fields=TEACHER_ACTION_ENTROPY_FIELDS[2:],
        )
        write_metrics_csv(
            str(output_path),
            TEACHER_ACTION_ENTROPY_FIELDS,
            self._buffer,
            index_field="number",
            mean_row=mean_row,
        )
        upload_clearml_artifact(
            self.clearml_task,
            str(output_path),
            artifact_name=output_path.name,
        )
        self._written_paths.append(output_path)
        self._buffer = []


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the replay-only teacher entropy evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate teacher-driven ego action entropy with replay-controlled "
            "surrounding vehicles."
        )
    )
    parser.add_argument(
        "--scenario_index_path",
        type=str,
        default="preparation_file/scenario_index_10k_train_no_offroad.json",
    )
    parser.add_argument(
        "--scenario_csv_path",
        type=str,
        default=None,
        help="Optional CSV whose scenario_id rows define the exact evaluation order.",
    )
    parser.add_argument(
        "--scenario_data_dir",
        type=str,
        default=(
            "scenario_data/formatted_json_v2_no_tl_train"
        ),
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="compressed_preprocessed_data/train",
    )
    parser.add_argument(
        "--vehicle_map_path",
        type=str,
        default="preparation_file/vehicle_map_10k_train.json",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/model_fp16.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/logs/dcd",
    )
    parser.add_argument(
        "--xpid",
        type=str,
        default="teacher-action-entropy-071814",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--num_steps", type=int, default=90)
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
        default="amp_bf16",
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["none", "global", "per_vehicle"],
        default="none",
    )
    parser.add_argument("--tilt_range_min", type=float, default=0.0)
    parser.add_argument("--tilt_range_max", type=float, default=0.0)
    parser.add_argument(
        "--enable_goal_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--enable_veh_veh_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--enable_veh_edge_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--show_vehicle_ids", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--use_clearml",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Execute evaluation remotely on a ClearML agent.",
    )
    parser.add_argument(
        "--clearml_monitor_only",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable local ClearML monitoring without remote execution.",
    )
    parser.add_argument(
        "--clearml_project",
        type=str,
        default="abouelazm/behavior-curriculum",
    )
    parser.add_argument(
        "--clearml_task",
        type=str,
        default="teacher-action-entropy",
    )
    parser.add_argument(
        "--clearml_dataset_project",
        type=str,
        default="ctrlsim_dataset",
    )
    parser.add_argument(
        "--clearml_dataset_name",
        type=str,
        default="ctrlsim_dataset",
    )
    parser.add_argument("--artifact_chunk_size", type=int, default=1000)
    return parser.parse_args(argv)


def _remap_clearml_dataset_paths(args: argparse.Namespace, dataset_dir: str) -> None:
    """Resolve dataset-backed resource paths on a ClearML worker."""
    default_paths = {
        "scenario_index_path": "preparation_file/scenario_index_10k_train_no_offroad.json",
        "scenario_data_dir": "scenario_data/formatted_json_v2_no_tl_train",
        "preprocess_dir": "compressed_preprocessed_data/train",
        "checkpoint_path": "checkpoints/model_fp16.ckpt",
        "vehicle_map_path": "preparation_file/vehicle_map_10k_train.json",
    }
    for key, default_relative_path in default_paths.items():
        path = getattr(args, key, None)
        if not path:
            continue
        if os.path.isabs(path):
            resolved_path = path if os.path.exists(path) else os.path.join(dataset_dir, default_relative_path)
        else:
            resolved_path = os.path.join(dataset_dir, path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"ClearML dataset missing required path for {key}: {resolved_path}"
            )
        setattr(args, key, resolved_path)


def maybe_create_clearml_task(args: argparse.Namespace) -> Any:
    """Create a ClearML task and remap dataset-backed paths on workers."""
    is_clearml_worker = bool(os.environ.get("CLEARML_TASK_ID"))
    if bool(getattr(args, "use_clearml", False)) and bool(
        getattr(args, "clearml_monitor_only", False)
    ):
        raise ValueError(
            "--use_clearml and --clearml_monitor_only are mutually exclusive"
        )

    clearml_enabled = (
        bool(getattr(args, "use_clearml", False))
        or bool(getattr(args, "clearml_monitor_only", False))
        or is_clearml_worker
    )
    if not clearml_enabled:
        return None

    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError(
            "ClearML support requires the 'clearml' package when --use_clearml is set."
        ) from exc

    task = Task.init(
        project_name=str(args.clearml_project),
        task_name=str(args.clearml_task),
        reuse_last_task_id=False,
        tags=["test run"],
        output_uri="s3://tks-zx.fzi.de:9000/ri928",
        auto_connect_frameworks={"tensorboard": False},
        auto_resource_monitoring=True,
    )
    if is_clearml_worker:
        from util.clearml import download_clearml_dataset

        if args.clearml_dataset_project and args.clearml_dataset_name:
            dataset_dir = download_clearml_dataset(
                args.clearml_dataset_project,
                args.clearml_dataset_name,
            )
            _remap_clearml_dataset_paths(args, dataset_dir)
        return task

    task.connect(vars(args))
    if bool(getattr(args, "use_clearml", False)):
        task.set_base_docker(
            "tks-zx.fzi.de/hu778/dcd-nocturne",
            docker_setup_bash_script=[
                "apt-get install -y libgl1 ffmpeg imagemagick",
            ],
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all --network=host",
        )
        task.execute_remotely("default", clone=False, exit_process=True)
    return task


def compute_action_entropy_from_logits(
    logits: Optional[Sequence[float] | np.ndarray],
) -> Optional[float]:
    """Return categorical entropy for one action-logit vector."""
    if logits is None:
        return None
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    return float(torch.distributions.Categorical(logits=logits_tensor).entropy().item())


def summarize_action_entropy(entropies: Sequence[float]) -> dict[str, float]:
    """Return mean/std/count summary for one episode's entropy sequence."""
    if not entropies:
        return {
            "action_entropy": 0.0,
            "action_entropy_std": 0.0,
            "action_entropy_num_steps": 0.0,
        }

    values = np.asarray(entropies, dtype=np.float32)
    return {
        "action_entropy": float(values.mean()),
        "action_entropy_std": float(values.std()),
        "action_entropy_num_steps": float(values.size),
    }


def flush_artifact_batches(
    *,
    episode_metrics: Sequence[dict[str, float | str]],
    output_dir: Path,
    clearml_task: Any,
    chunk_size: int,
) -> list[Path]:
    """Write numbered CSV artifact chunks for a completed metrics sequence."""
    writer = ArtifactBatchWriter(
        output_dir=output_dir,
        clearml_task=clearml_task,
        chunk_size=chunk_size,
    )
    for row in episode_metrics:
        writer.add_row(dict(row))
    return writer.finalize()


def _split_ctrlsim_eval_prepared_batch(
    per_env_prepared: Sequence[Optional[dict[str, Any]]],
) -> tuple[list[Optional[dict[str, Any]]], list[Optional[dict[str, Any]]], list[Optional[dict[str, Any]]]]:
    """Split the prepared payloads into ego, ego-ctrlsim, and opponent streams."""
    ego_prepared = [item.get("ego") if item else None for item in per_env_prepared]
    ego_ctrlsim_prepared = [
        item.get("ego_ctrlsim") if item else None for item in per_env_prepared
    ]
    opponent_prepared = [
        item.get("opponent") if item else None for item in per_env_prepared
    ]
    return ego_prepared, ego_ctrlsim_prepared, opponent_prepared


def _run_batched_entropy_step(
    *,
    venv: Any,
    action: np.ndarray,
    external_teacher: Any,
    reset_random: bool,
    auto_reset_on_done: bool,
) -> tuple[Any, Any, Any, Sequence[dict[str, Any]], list[Optional[float]]]:
    """Run one batched replay step and return per-env teacher entropy values."""
    per_env_prepared = venv.step_prepare(action)
    ego_prepared, ego_ctrlsim_prepared, opponent_prepared = (
        _split_ctrlsim_eval_prepared_batch(per_env_prepared)
    )

    ego_results = external_teacher.run_batched_forward(ego_prepared)
    opponent_results, ego_logits_by_env, _, _ = (
        external_teacher.run_batched_forward_with_ego_logits(
            opponent_prepared,
            ego_ctrlsim_prepared,
        )
    )
    combined_outputs = [
        {"ego": ego_result, "opponent": opponent_result}
        for ego_result, opponent_result in zip(ego_results, opponent_results)
    ]
    obs, reward, done, infos = venv.step_complete(
        combined_outputs,
        reset_random=reset_random,
        auto_reset_on_done=auto_reset_on_done,
    )
    entropies = [
        compute_action_entropy_from_logits(logits)
        for logits in ego_logits_by_env
    ]
    return obs, reward, done, infos, entropies


def _build_episode_metrics_row(
    *,
    info: Mapping[str, Any],
    entropies: Sequence[float],
    progress_threshold: float,
) -> dict[str, float | str]:
    """Build one CSV row for a completed replay episode."""
    metrics = extract_ctrlsim_episode_metrics(
        dict(info),
        offroad_progress_threshold=progress_threshold,
    )
    row = {
        "scenario_id": metrics["scenario_id"],
        "collision": float(metrics["collision"]),
        "offroad": float(metrics["offroad"]),
        "progress": float(metrics["progress"]),
        "solved": compute_solved_flag(
            progress=float(metrics["progress"]),
            collision=float(metrics["collision"]),
            offroad=float(metrics["offroad"]),
            progress_threshold=progress_threshold,
        ),
        "ade": float(metrics["ade"]),
        "fde": float(metrics["fde"]),
        "meta_jsd": float(metrics["meta_jsd"]),
    }
    row.update(summarize_action_entropy(entropies))
    return row


def evaluate_fixed_episode_templates(
    *,
    args: argparse.Namespace,
    venv: Any,
    external_teacher: Any,
    artifact_batch_writer: ArtifactBatchWriter | None = None,
) -> list[dict[str, float | str]]:
    """Evaluate the exact scenario order from one CSV file."""
    episode_templates = build_episode_templates(
        scenario_ids=read_scenario_ids_from_csv(args.scenario_csv_path),
        seed=args.seed,
    )
    if not episode_templates:
        return []

    action_batch = build_zero_action_batch(venv, args.num_processes)
    episode_metrics: list[dict[str, float | str]] = []
    next_template_idx = 0
    active_templates: list[Any] = []
    active_episodes = [ActiveEpisode(entropies=[]) for _ in range(args.num_processes)]
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
    progress = tqdm(total=len(episode_templates), disable=not args.verbose)

    while len(episode_metrics) < len(episode_templates):
        _, _, done, infos, entropies = _run_batched_entropy_step(
            venv=venv,
            action=action_batch,
            external_teacher=external_teacher,
            reset_random=False,
            auto_reset_on_done=False,
        )
        reset_levels = []
        reset_indices = []
        for idx, info in enumerate(infos):
            entropy = entropies[idx]
            if entropy is not None:
                active_episodes[idx].entropies.append(entropy)

            if not done[idx]:
                continue

            template = active_templates[idx]
            if template is not None and "episode" in info:
                row = _build_episode_metrics_row(
                        info=info,
                        entropies=active_episodes[idx].entropies,
                        progress_threshold=args.progress_threshold,
                )
                episode_metrics.append(row)
                if artifact_batch_writer is not None:
                    artifact_batch_writer.add_row(dict(row))
                progress.update(1)
                if len(episode_metrics) >= len(episode_templates):
                    break

            active_episodes[idx] = ActiveEpisode(entropies=[])
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

    progress.close()
    return episode_metrics


def evaluate_random_episodes(
    *,
    args: argparse.Namespace,
    venv: Any,
    external_teacher: Any,
    artifact_batch_writer: ArtifactBatchWriter | None = None,
) -> list[dict[str, float | str]]:
    """Evaluate replay episodes with random resets."""
    venv.reset_random()
    action_batch = build_zero_action_batch(venv, args.num_processes)
    episode_metrics: list[dict[str, float | str]] = []
    active_episodes = [ActiveEpisode(entropies=[]) for _ in range(args.num_processes)]
    progress = tqdm(total=args.num_episodes, disable=not args.verbose)

    while len(episode_metrics) < args.num_episodes:
        _, _, _, infos, entropies = _run_batched_entropy_step(
            venv=venv,
            action=action_batch,
            external_teacher=external_teacher,
            reset_random=True,
            auto_reset_on_done=True,
        )
        for idx, info in enumerate(infos):
            entropy = entropies[idx]
            if entropy is not None:
                active_episodes[idx].entropies.append(entropy)
            if "episode" not in info:
                continue

            row = _build_episode_metrics_row(
                    info=info,
                    entropies=active_episodes[idx].entropies,
                    progress_threshold=args.progress_threshold,
            )
            episode_metrics.append(row)
            if artifact_batch_writer is not None:
                artifact_batch_writer.add_row(dict(row))
            active_episodes[idx] = ActiveEpisode(entropies=[])
            progress.update(1)
            if len(episode_metrics) >= args.num_episodes:
                break

    progress.close()
    return episode_metrics


def run_evaluation(
    args: argparse.Namespace,
    *,
    artifact_batch_writer: ArtifactBatchWriter | None = None,
) -> list[dict[str, float | str]]:
    """Run the replay-only teacher entropy evaluation and return CSV rows."""
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
        collect_ego_ctrlsim_rtg=True,
        opponent_runtime_mode="replay",
        teacher_control_mode="split",
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
                artifact_batch_writer=artifact_batch_writer,
            )
        return evaluate_random_episodes(
            args=args,
            venv=venv,
            external_teacher=external_teacher,
            artifact_batch_writer=artifact_batch_writer,
        )
    finally:
        evaluator.close()


def main() -> None:
    """Run the replay-only teacher entropy script and write one CSV file."""
    args = parse_args()
    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clearml_task = maybe_create_clearml_task(args)
    artifact_batch_writer = ArtifactBatchWriter(
        output_dir=output_dir,
        clearml_task=clearml_task,
        chunk_size=int(args.artifact_chunk_size),
    )

    episode_metrics = run_evaluation(
        args,
        artifact_batch_writer=artifact_batch_writer,
    )
    artifact_batch_writer.finalize()
    output_path = resolve_csv_output_path(
        str(output_dir),
        f"{args.xpid}-replay-teacher-action-entropy",
    )
    mean_row = build_metrics_mean_row(
        episode_metrics,
        TEACHER_ACTION_ENTROPY_FIELDS,
        label_field="number",
        label_value="mean",
        empty_fields=("scenario_id",),
        mean_fields=TEACHER_ACTION_ENTROPY_FIELDS[2:],
    ) if episode_metrics else None
    write_metrics_csv(
        output_path,
        TEACHER_ACTION_ENTROPY_FIELDS,
        episode_metrics,
        index_field="number",
        mean_row=mean_row,
    )
    print(f"Replay teacher entropy metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
