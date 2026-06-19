"""Random-scenario Nocturne evaluation for ego policy with policy opponents."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_common import (
    build_metrics_mean_row,
    compute_solved_flag,
    resolve_csv_output_path,
    start_headless_display,
    stop_headless_display,
    validate_nocturne_env_names,
    write_metrics_csv,
)
from evaluation.ego_policy_evaluation import (
    EgoReplayEvaluator,
    MULTI_CHECKPOINT_METRIC_FIELDS,
    SOLVED_PROGRESS_THRESHOLD,
    build_policy_opponent_teacher,
    find_checkpoint_targets,
    load_agent_from_checkpoint,
)
from util import DotDict, str2bool

DEFAULT_OPPONENT_CHECKPOINT = (
    "/media/chen/Dataset/ctrlsim_dataset/checkpoints/model_fp16.ckpt"
)
INT32_MAX = 2**31 - 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for random-scenario ego policy evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ego policy in Nocturne-CtrlSim with random scenario "
            "sampling and policy-controlled opponents."
        )
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="~/logs/dcd",
        help="Directory containing model checkpoint and meta.json.",
    )
    parser.add_argument(
        "--env_names",
        type=str,
        default="Nocturne-CtrlSim-v0",
        help="CSV string of Nocturne evaluation environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional base seed. When omitted, generate a fresh random seed.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=10,
        help="Number of CPU processes to use.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=300,
        help="Number of evaluation episodes per environment.",
    )
    parser.add_argument(
        "--model_tar",
        type=str,
        default="model",
        help="Name of .tar to evaluate.",
    )
    parser.add_argument(
        "--all_checkpoints",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Evaluate every .tar checkpoint under base_path.",
    )
    parser.add_argument(
        "--opponent_eval_mode",
        type=str,
        choices=["replay", "policy"],
        default="policy",
        help="Use GT replay or CtrlSim policy control for opponent vehicles.",
    )
    parser.add_argument(
        "--opponent_checkpoint",
        type=str,
        default=DEFAULT_OPPONENT_CHECKPOINT,
        help="CtrlSim checkpoint path for policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["none", "global", "per_vehicle"],
        default="per_vehicle",
        help="Tilt mode for CtrlSim policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilt_range_min",
        type=float,
        default=-20,
        help="Minimum sampled tilt value for policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilt_range_max",
        type=float,
        default=0,
        help="Maximum sampled tilt value for policy-controlled opponents.",
    )
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
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Evaluate policy greedily.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Show logging messages in stdout.",
    )
    parser.add_argument(
        "--record_video",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Record video of first environment evaluation process.",
    )
    return parser.parse_args()


def build_solved_metrics(
    episode_metrics: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Append solved flags to per-episode metrics."""
    solved_metrics = []
    for metrics in episode_metrics:
        row = dict(metrics)
        row["solved"] = compute_solved_flag(
            progress=float(row["progress"]),
            collision=float(row["collision"]),
            offroad=float(row["offroad"]),
            progress_threshold=SOLVED_PROGRESS_THRESHOLD,
        )
        solved_metrics.append(row)
    return solved_metrics


def resolve_runtime_seed(cli_args: DotDict, xpid_flags: DotDict) -> int:
    """Return the per-run seed used to diversify worker sampling."""
    cli_seed = cli_args.get("seed")
    if cli_seed is not None:
        return int(cli_seed)

    return 1 + secrets.randbelow(INT32_MAX - 1)


def run_single_random_checkpoint_evaluation(
    *,
    base_path: str,
    model_tar: str,
    env_names: list[str],
    device: str,
    cli_args: DotDict,
    video_dir: str,
    progress_position: int = 0,
) -> list[dict[str, float | str]]:
    """Run one checkpoint evaluation with random scenario sampling."""
    agent, xpid_flags, nocturne_required, dummy_venv = load_agent_from_checkpoint(
        base_path=base_path,
        model_tar=model_tar,
        env_name=env_names[0],
        device=device,
        cli_args=cli_args,
    )
    runtime_seed = resolve_runtime_seed(cli_args=cli_args, xpid_flags=xpid_flags)
    xpid_flags.update(cli_args)
    xpid_flags["seed"] = runtime_seed
    xpid_flags.update({"use_skip": False})

    opponent_eval_mode = str(cli_args.get("opponent_eval_mode", "policy"))
    external_teacher = None
    if opponent_eval_mode == "policy":
        external_teacher = build_policy_opponent_teacher(
            checkpoint_path=str(nocturne_required["opponent_checkpoint"]),
            device=device,
            cli_args=xpid_flags,
        )

    evaluator = None
    try:
        evaluator = EgoReplayEvaluator(
            env_names,
            num_processes=cli_args.num_processes,
            num_episodes=cli_args.num_episodes,
            frame_stack=xpid_flags.frame_stack,
            grayscale=xpid_flags.grayscale,
            use_global_critic=xpid_flags.use_global_critic,
            video_dir=video_dir,
            seed=runtime_seed,
            opponent_eval_mode=opponent_eval_mode,
            **nocturne_required,
            record_video=cli_args.record_video,
        )
        return evaluator.evaluate(
            agent,
            deterministic=cli_args.deterministic,
            show_progress=cli_args.verbose,
            progress_position=progress_position,
            accumulator=None,
            external_teacher=external_teacher,
            episode_templates=None,
        )
    finally:
        if evaluator is not None:
            evaluator.close()
        dummy_venv.close()


def run_all_random_checkpoint_evaluations(
    *,
    base_path: str,
    env_names: list[str],
    device: str,
    cli_args: DotDict,
) -> None:
    """Evaluate every checkpoint under base_path with random scenario sampling."""
    checkpoint_targets = find_checkpoint_targets(base_path)
    if not checkpoint_targets:
        raise FileNotFoundError(f"No .tar checkpoints found under {base_path}")

    pending_targets = []
    for target in checkpoint_targets:
        target_result_path = os.path.join(target.checkpoint_dir, "evaluation")
        result_file = os.path.join(target_result_path, f"eval-{target.model_tar}.csv")
        if os.path.exists(result_file):
            continue
        pending_targets.append((target, target_result_path, result_file))

    cli_args.record_video = False
    pbar = tqdm(total=len(pending_targets), position=0) if cli_args.verbose else None
    try:
        for target, target_result_path, result_file in pending_targets:
            os.makedirs(target_result_path, exist_ok=True)
            if pbar is not None:
                relative_dir = os.path.relpath(target.checkpoint_dir, base_path)
                label = (
                    target.model_tar
                    if relative_dir == "."
                    else f"{relative_dir}/{target.model_tar}"
                )
                pbar.set_description_str(f"Evaluating {label}")

            episode_metrics = run_single_random_checkpoint_evaluation(
                base_path=target.checkpoint_dir,
                model_tar=target.model_tar,
                env_names=env_names,
                device=device,
                cli_args=cli_args,
                video_dir=target_result_path,
                progress_position=1,
            )
            solved_metrics = build_solved_metrics(episode_metrics)
            mean_row = build_metrics_mean_row(
                solved_metrics,
                MULTI_CHECKPOINT_METRIC_FIELDS,
                label_field="number",
                label_value="mean",
                empty_fields=("scenario_id",),
                mean_fields=MULTI_CHECKPOINT_METRIC_FIELDS[2:],
            )
            write_metrics_csv(
                result_file,
                MULTI_CHECKPOINT_METRIC_FIELDS,
                solved_metrics,
                index_field="number",
                mean_row=mean_row,
            )
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()


def main() -> None:
    """Run random-scenario ego policy evaluation."""
    os.environ["OMP_NUM_THREADS"] = "1"
    args = DotDict(vars(parse_args()))
    args.num_processes = min(args.num_processes, args.num_episodes)

    display = start_headless_display()
    try:
        device = "cuda"
        base_path = os.path.expandvars(os.path.expanduser(args.base_path))
        result_path = os.path.join(base_path, "evaluation")
        video_dir = result_path

        os.makedirs(result_path, exist_ok=True)
        if args.record_video:
            os.makedirs(video_dir, exist_ok=True)

        env_names = args.env_names.split(",")
        validate_nocturne_env_names(env_names)

        if args.all_checkpoints:
            run_all_random_checkpoint_evaluations(
                base_path=base_path,
                env_names=env_names,
                device=device,
                cli_args=args,
            )
            return

        result_fname = f"ego-policy-random-eval-{args.model_tar}"
        if args.record_video:
            if len(env_names) != 1:
                raise ValueError("--record_video requires exactly one env_name")
            args.num_processes = 1

        episode_metrics = run_single_random_checkpoint_evaluation(
            base_path=base_path,
            model_tar=args.model_tar,
            env_names=env_names,
            device=device,
            cli_args=args,
            video_dir=video_dir,
        )
        solved_metrics = build_solved_metrics(episode_metrics)
        mean_row = build_metrics_mean_row(
            solved_metrics,
            MULTI_CHECKPOINT_METRIC_FIELDS,
            label_field="number",
            label_value="mean",
            empty_fields=("scenario_id",),
            mean_fields=MULTI_CHECKPOINT_METRIC_FIELDS[2:],
        )
        result_fpath = resolve_csv_output_path(result_path, result_fname)
        write_metrics_csv(
            result_fpath,
            MULTI_CHECKPOINT_METRIC_FIELDS,
            solved_metrics,
            index_field="number",
            mean_row=mean_row,
        )
    finally:
        stop_headless_display(display)


if __name__ == "__main__":
    main()
