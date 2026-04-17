"""Specialized Nocturne evaluation for ego policy with GT-replay opponents."""

import argparse
import csv
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.wrappers import ParallelAdversarialVecEnv
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

EPISODE_METRIC_FIELDS = (
    "number",
    "scenario_id",
    "collision",
    "offroad",
    "position_reached",
    "progress",
    "total_episode_reward",
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


def _validate_env_names(env_names):
    """Validate that all requested environments are Nocturne variants."""
    invalid = [name for name in env_names if not name.startswith("Nocturne")]
    if invalid:
        raise ValueError(
            "ego_policy_evaluation only supports Nocturne envs, got: "
            + ", ".join(invalid)
        )


def _collect_replay_nocturne_args(flags, cli_args):
    """Collect only Nocturne args needed by replay-only ego evaluation."""
    required = {
        "scenario_index_path": SCENARIO_INDEX_PATH,
        "scenario_data_dir": SCENARIO_DATA_DIR,
        "preprocess_dir": PREPROCESS_DIR,
        "vehicle_map_path": VEHICLE_MAP_PATH,
    }
    keys = [
        "opponent_checkpoint",
        "max_episode_steps",
        "student_accel_discretization",
        "student_steer_discretization",
        "done_on_position_reached_only",
        "goal_pos_tolerance",
        "use_speed_heading_target",
    ]
    for key in keys:
        if key in flags:
            required[key] = flags[key]
        elif key in cli_args:
            required[key] = cli_args[key]
    return required


def _build_nocturne_env(env_name, record_video=False, process_idx=None, **kwargs):
    """Build one Nocturne evaluation env fixed to opponent GT replay mode."""
    allowed_nocturne_keys = {
        "scenario_index_path",
        "opponent_checkpoint",
        "scenario_data_dir",
        "preprocess_dir",
        "vehicle_map_path",
        "max_episode_steps",
        "student_accel_discretization",
        "student_steer_discretization",
        "done_on_position_reached_only",
        "goal_pos_tolerance",
        "device",
        "tilting_mode",
        "use_speed_heading_target",
        "opponent_runtime_mode",
    }
    nocturne_kwargs = {k: v for k, v in kwargs.items() if k in allowed_nocturne_keys}
    if "tilting_mode" not in nocturne_kwargs:
        nocturne_kwargs["tilting_mode"] = "per_vehicle"
    nocturne_kwargs["opponent_runtime_mode"] = "replay"
    env = NocturneCtrlSimAdversarial(**nocturne_kwargs)

    if not record_video:
        return env

    video_dir = kwargs.get("video_dir", "videos/")
    original_env = env

    class VideoWrapper:
        """Wrap Nocturne env to manage video recording for evaluation."""

        def __init__(self):
            """Store wrapped env metadata for recording."""
            self.env = original_env
            self.video_dir = video_dir
            self.episode_count = 0
            self.recording_started = False
            self.process_idx = process_idx
            self.observation_space = original_env.observation_space
            self.action_space = original_env.action_space

        def _episode_name(self):
            """Return the recording file prefix for the current episode."""
            if self.process_idx is None:
                return f"episode_{self.episode_count:04d}"
            return f"process{self.process_idx}_episode_{self.episode_count:04d}"

        def _start_if_needed(self):
            """Start recording once per episode."""
            if self.recording_started:
                return
            self.env.start_recording(
                self.video_dir,
                self._episode_name(),
                fps=10,
                dpi=100,
            )
            self.recording_started = True

        def _stop_if_recording(self):
            """Stop recording and advance episode counter when needed."""
            if not self.recording_started:
                return
            if getattr(self.env, "recording_video", False):
                self.env.stop_recording(self._episode_name())
            self.episode_count += 1
            self.recording_started = False

        def reset(self, **kw):
            """Reset the wrapped env, preferring Nocturne random reset."""
            if hasattr(self.env, "reset_random") and not kw:
                return self.env.reset_random()
            return self.env.reset(**kw)

        def reset_random(self, **kw):
            """Reset the wrapped env to a random level."""
            return self.env.reset_random(**kw)

        def reset_agent(self, **kw):
            """Reset only the agent-facing state in the wrapped env."""
            if kw:
                return self.env.reset_agent(**kw)
            return self.env.reset_agent()

        def step_prepare(self, action):
            """Forward phase-1 step call and ensure recording has started."""
            self._start_if_needed()
            return self.env.step_prepare(action)

        def step_complete(self, model_output):
            """Forward phase-2 step call and stop recording on done."""
            self._start_if_needed()
            obs, reward, done, info = self.env.step_complete(model_output)
            if done:
                self._stop_if_recording()
            return obs, reward, done, info

        def close(self):
            """Close the wrapped env and flush any active recording."""
            self._stop_if_recording()
            self.env.close()

        def __getattr__(self, name):
            """Delegate unknown attributes to the wrapped env."""
            return getattr(self.env, name)

    return VideoWrapper()


def _extract_episode_metrics(info: dict[str, Any]) -> dict[str, float | str]:
    """Extract one completed episode's CSV metrics from Nocturne info."""
    episode_info = info.get("episode", {})
    total_episode_reward = episode_info.get("r", info.get("episode_reward", 0.0))
    return {
        "scenario_id": info.get("scenario_id", ""),
        "collision": float(info.get("collision_occurred", info.get("collision", 0.0))),
        "offroad": float(info.get("offroad_occurred", info.get("offroad", 0.0))),
        "position_reached": float(
            info.get("position_reached_occurred", info.get("position_reached", 0.0))
        ),
        "progress": float(info.get("max_progress", info.get("progress", 0.0))),
        "total_episode_reward": float(total_episode_reward),
    }


def _build_episode_metrics_mean_row(
    episode_metrics: list[dict[str, float | str]],
) -> dict[str, float | str]:
    """Build the final CSV mean row for episode metrics."""
    mean_row: dict[str, float | str] = {"number": "mean", "scenario_id": ""}
    for field in EPISODE_METRIC_FIELDS[2:]:
        values = [float(row[field]) for row in episode_metrics]
        mean_row[field] = float(np.mean(values)) if values else 0.0
    return mean_row


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

    nocturne_required = _collect_replay_nocturne_args(xpid_flags_meta, cli_args)
    dummy_venv = ParallelAdversarialVecEnv(
        [lambda: _build_nocturne_env(env_name, **nocturne_required)],
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
        _validate_env_names(env_names)
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}
        eval_seed = kwargs.get("seed")

        for env_name in env_names:
            make_fn = [
                (lambda idx: lambda: _build_nocturne_env(
                    env_name,
                    record_video,
                    process_idx=idx,
                    **kwargs,
                ))(i)
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
                    episode_metrics.append(_extract_episode_metrics(info))
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

    display = None
    if sys.platform.startswith("linux"):
        import pyvirtualdisplay

        display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
        display.start()
    device = "cuda"

    base_path = os.path.expandvars(os.path.expanduser(args.base_path))
    result_path = os.path.join(base_path, "evaluation")
    video_dir = result_path

    os.makedirs(result_path, exist_ok=True)
    if args.record_video:
        os.makedirs(video_dir, exist_ok=True)

    result_fname = f"ego-policy-eval-{args.model_tar}"

    env_names = args.env_names.split(",")
    _validate_env_names(env_names)

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

    result_fpath = os.path.join(result_path, result_fname)
    if os.path.exists(f"{result_fpath}.csv"):
        result_fpath = os.path.join(result_path, f"{result_fname}_redo")
    result_fpath = f"{result_fpath}.csv"

    with open(result_fpath, "w", newline="") as csvout:
        csvwriter = csv.writer(csvout)
        csvwriter.writerow(list(EPISODE_METRIC_FIELDS))
        for idx, row in enumerate(episode_metrics, start=1):
            csvwriter.writerow(
                [idx, *[row[field] for field in EPISODE_METRIC_FIELDS[1:]]]
            )
        mean_row = _build_episode_metrics_mean_row(episode_metrics)
        csvwriter.writerow([mean_row[field] for field in EPISODE_METRIC_FIELDS])
    if display:
        display.stop()


if __name__ == "__main__":
    main()
