"""Run ego-policy evaluation for every numbered model checkpoint in a run dir."""

import argparse
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_common import (
    start_headless_display,
    stop_headless_display,
    validate_nocturne_env_names,
    write_episode_metrics_csv,
)
from util import DotDict, ignore_warning, str2bool

from evaluation.ego_policy_evaluation import (
    run_single_checkpoint_evaluation,
)

ignore_warning.configure_subprocess_env()

MODEL_TAR_PATTERN = re.compile(r"^model_(\d+)\.tar$")


def parse_args():
    """Parse CLI arguments for multi-checkpoint ego policy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate all numbered model checkpoints with GT-replay opponents."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="~/logs/dcd",
        help="Directory containing model checkpoints and meta.json.",
    )
    parser.add_argument(
        "--env_names",
        type=str,
        default="Nocturne-CtrlSim-v0",
        help="CSV string of Nocturne evaluation environments.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of CPU processes to use.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per environment.",
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
        default=False,
        help="Show logging messages in stdout.",
    )
    return parser.parse_args()


def find_model_checkpoints(base_path: str) -> list[str]:
    """Return sorted checkpoint stems matching model_<step>.tar."""
    matched = []
    for filename in os.listdir(base_path):
        match = MODEL_TAR_PATTERN.match(filename)
        if match is None:
            continue
        matched.append((int(match.group(1)), filename[:-4]))
    matched.sort(key=lambda item: item[0])
    return [stem for _, stem in matched]

def main():
    """Evaluate all numbered checkpoints under one base path."""
    os.environ["OMP_NUM_THREADS"] = "1"
    args = DotDict(vars(parse_args()))
    args.num_processes = min(args.num_processes, args.num_episodes)
    args.record_video = False

    display = start_headless_display()
    try:
        device = "cuda"
        base_path = os.path.expandvars(os.path.expanduser(args.base_path))
        result_path = os.path.join(base_path, "evaluation")
        os.makedirs(result_path, exist_ok=True)

        env_names = args.env_names.split(",")
        validate_nocturne_env_names(env_names)

        model_tars = find_model_checkpoints(base_path)
        if not model_tars:
            raise FileNotFoundError(
                f"No numbered checkpoints found under {base_path}: expected model_<step>.tar"
            )

        eval_args = DotDict(dict(args))
        pbar = tqdm(total=len(model_tars), position=0) if args.verbose else None
        try:
            for model_tar in model_tars:
                if pbar is not None:
                    pbar.set_description_str(f"Evaluating {model_tar}")
                episode_metrics = run_single_checkpoint_evaluation(
                    base_path=base_path,
                    model_tar=model_tar,
                    env_names=env_names,
                    device=device,
                    cli_args=eval_args,
                    video_dir=result_path,
                    progress_position=1,
                )
                write_episode_metrics_csv(
                    os.path.join(result_path, f"eval-{model_tar}.csv"),
                    episode_metrics,
                )
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
    finally:
        stop_headless_display(display)


if __name__ == "__main__":
    main()
