"""Specialized Nocturne evaluation for ego policy with GT-replay opponents."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.wrappers import ParallelAdversarialVecEnv
from evaluation.evaluation_common import (
    build_replay_nocturne_env,
    collect_replay_nocturne_args,
    extract_episode_metrics,
    resolve_csv_output_path,
    start_headless_display,
    stop_headless_display,
    validate_nocturne_env_names,
    write_episode_metrics_csv,
)
from evaluation.eval import Evaluator, load_actor_critic_checkpoint
from util import DotDict, ignore_warning, is_discrete_actions, make_agent, str2bool
from util.eval_helper import set_eval_worker_seeds

ignore_warning.configure_subprocess_env()

SCENARIO_INDEX_PATH = (
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/"
    "scenarios_index_filtered_valid.json"
)
SCENARIO_DATA_DIR = (
    "/media/chen/Dataset/ctrlsim_dataset/scenario_data/"
    "formatted_json_v2_no_tl_valid"
)
PREPROCESS_DIR = (
    "/media/chen/Dataset/ctrlsim_dataset/compressed_preprocessed_data/test"
)
VEHICLE_MAP_PATH = (
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/"
    "vehicle_map_filtered_valid.json"
)

def parse_args():
    """Parse CLI arguments for specialized ego policy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ego policy in Nocturne-CtrlSim with GT-replay opponents."
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
        "--model_tar",
        type=str,
        default="model",
        help="Name of .tar to evaluate.",
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
    parser.add_argument(
        "--record_video",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Record video of first environment evaluation process.",
    )
    return parser.parse_args()

def load_agent_from_checkpoint(
    *,
    base_path: str,
    model_tar: str,
    env_name: str,
    device: str,
    cli_args: DotDict,
) -> tuple[Any, DotDict, dict[str, Any], Any]:
    """Load the evaluation agent and runtime context for one checkpoint."""
    meta_json_path = os.path.join(base_path, "meta.json")
    checkpoint_path = os.path.join(base_path, f"{model_tar}.tar")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model path {checkpoint_path}")

    with open(meta_json_path) as meta_json_file:
        xpid_flags = DotDict(json.load(meta_json_file)["args"])
    xpid_flags_meta = DotDict(dict(xpid_flags))

    nocturne_required = collect_replay_nocturne_args(
        xpid_flags_meta,
        cli_args,
        {
            "scenario_index_path": SCENARIO_INDEX_PATH,
            "scenario_data_dir": SCENARIO_DATA_DIR,
            "preprocess_dir": PREPROCESS_DIR,
            "vehicle_map_path": VEHICLE_MAP_PATH,
        },
    )
    dummy_venv = ParallelAdversarialVecEnv(
        [
            lambda: build_replay_nocturne_env(
                NocturneCtrlSimAdversarial,
                **nocturne_required,
            )
        ],
        adversary=False,
        is_eval=True,
    )
    dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_name, device=device)

    agent = make_agent(name="agent", env=dummy_venv, args=xpid_flags, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_key = "agent"
    if "runner_state_dict" in checkpoint:
        load_actor_critic_checkpoint(
            agent.algo.actor_critic,
            checkpoint["runner_state_dict"]["agent_state_dict"][checkpoint_key],
        )
    else:
        load_actor_critic_checkpoint(agent.algo.actor_critic, checkpoint)
    return agent, xpid_flags, nocturne_required, dummy_venv


def run_single_checkpoint_evaluation(
    *,
    base_path: str,
    model_tar: str,
    env_names: list[str],
    device: str,
    cli_args: DotDict,
    video_dir: str,
    progress_position: int = 0,
) -> list[dict[str, float | str]]:
    """Run one checkpoint evaluation and return per-episode metrics."""
    agent, xpid_flags, nocturne_required, dummy_venv = load_agent_from_checkpoint(
        base_path=base_path,
        model_tar=model_tar,
        env_name=env_names[0],
        device=device,
        cli_args=cli_args,
    )
    xpid_flags.update(cli_args)
    xpid_flags.update({"use_skip": False})

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
            seed=cli_args.seed,
            **nocturne_required,
            record_video=cli_args.record_video,
        )
        return evaluator.evaluate(
            agent,
            deterministic=cli_args.deterministic,
            show_progress=cli_args.verbose,
            progress_position=progress_position,
            accumulator=None,
        )
    finally:
        if evaluator is not None:
            evaluator.close()
        dummy_venv.close()


class EgoReplayEvaluator(Evaluator):
    """Evaluator specialized for ego policy rollouts with GT-replay opponents."""

    def _init_parallel_envs(
        self,
        env_names,
        num_processes,
        device=None,
        record_video=False,
        **kwargs,
    ):
        """Initialize vectorized Nocturne envs fixed to replay mode."""
        validate_nocturne_env_names(env_names)
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}
        eval_seed = kwargs.get("seed")

        for env_name in env_names:
            env_kwargs = {k: v for k, v in kwargs.items() if k != "video_dir"}
            make_fn = [
                (
                    lambda idx: lambda: build_replay_nocturne_env(
                        NocturneCtrlSimAdversarial,
                        record_video=record_video,
                        process_idx=idx,
                        video_dir=kwargs.get("video_dir", "videos/"),
                        **env_kwargs,
                    )
                )(i)
                for i in range(self.num_processes)
            ]
            venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            venv = Evaluator.wrap_venv(venv, env_name, device=device)
            set_eval_worker_seeds(
                venv=venv,
                seed=eval_seed,
                num_processes=self.num_processes,
                singleton_env=False,
            )
            self.venv[env_name] = venv

        self.is_discrete_actions = is_discrete_actions(self.venv[env_names[0]])

    def evaluate(
        self,
        agent,
        deterministic=False,
        show_progress=False,
        progress_position=0,
        accumulator="mean",
        return_episode_returns=False,
        external_teacher=None,
    ):
        """Evaluate ego policy while all opponent vehicles follow GT replay."""
        del external_teacher
        episode_metrics = []

        for env_name, venv in self.venv.items():
            returns = []
            obs = venv.reset_random()

            recurrent_hidden_states = torch.zeros(
                self.num_processes,
                agent.algo.actor_critic.recurrent_hidden_state_size,
                device=self.device,
            )
            if (
                agent.algo.actor_critic.is_recurrent
                and agent.algo.actor_critic.rnn.arch == "lstm"
            ):
                recurrent_hidden_states = (
                    recurrent_hidden_states,
                    torch.zeros_like(recurrent_hidden_states),
                )
            masks = torch.ones(self.num_processes, 1, device=self.device)

            pbar = (
                tqdm(total=self.num_episodes, position=progress_position)
                if show_progress
                else None
            )

            while len(returns) < self.num_episodes:
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs,
                        recurrent_hidden_states,
                        masks,
                        deterministic=deterministic,
                    )

                action = action.cpu().numpy()
                action = agent.process_action(action)
                venv.step_prepare(action)
                model_outputs = [None] * self.num_processes
                obs, reward, done, infos = venv.step_complete(
                    model_outputs,
                    reset_random=True,
                )

                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device,
                )

                for i, info in enumerate(infos):
                    if "episode" not in info:
                        continue
                    returns.append(info["episode"]["r"])
                    episode_metrics.append(extract_episode_metrics(info))
                    if pbar:
                        pbar.update(1)
                    if agent.is_recurrent:
                        recurrent_hidden_states[0][i].zero_()
                        recurrent_hidden_states[1][i].zero_()
                    if len(returns) >= self.num_episodes:
                        break

            if pbar:
                pbar.close()

        if return_episode_returns:
            return episode_metrics, returns
        return episode_metrics


def main():
    """Run specialized ego policy evaluation with GT-replay opponents."""
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

        result_fname = f"ego-policy-eval-{args.model_tar}"
        env_names = args.env_names.split(",")
        validate_nocturne_env_names(env_names)

        if args.record_video:
            if len(env_names) != 1:
                raise ValueError("--record_video requires exactly one env_name")
            args.num_processes = 1

        episode_metrics = run_single_checkpoint_evaluation(
            base_path=base_path,
            model_tar=args.model_tar,
            env_names=env_names,
            device=device,
            cli_args=args,
            video_dir=video_dir,
        )
        result_fpath = resolve_csv_output_path(result_path, result_fname)
        write_episode_metrics_csv(result_fpath, episode_metrics)
    finally:
        stop_headless_display(display)


if __name__ == "__main__":
    main()
