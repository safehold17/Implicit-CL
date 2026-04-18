#!/usr/bin/env python3
import argparse
import os
import sys

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def evaluate_with_metrics(
    evaluator,
    show_progress,
    render,
    tilting_mode,
    progress_threshold,
    external_teacher=None,
):
    env_name = evaluator.env_names[0]
    venv = evaluator.venv[env_name]
    num_episodes = evaluator.num_episodes
    if env_name.startswith("Nocturne") and hasattr(venv, "reset_random"):
        obs = venv.reset_random()
    else:
        obs = venv.reset()

    episode_metrics = []
    pbar = tqdm(total=num_episodes) if show_progress else None
    action_batch = build_zero_action_batch(venv, evaluator.num_processes)

    while len(episode_metrics) < num_episodes:
        if external_teacher is not None:
            obs, reward, done, infos = run_batched_ctrlsim_step(
                venv=venv,
                action=action_batch,
                external_teacher=external_teacher,
                reset_random=True,
                auto_reset_on_done=True,
                collect_ego_ctrlsim_rtg=False,
            )
        else:
            obs, reward, done, infos = venv.step(action_batch)

        for info in infos:
            if "episode" in info.keys():
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
    )
    csv_path = write_solvability_metrics_csv(args.output_dir, args.xpid, episode_metrics)
    print(f"Metrics saved to: {csv_path}")

    evaluator.close()


if __name__ == "__main__":
    main()
