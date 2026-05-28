"""Specialized Nocturne evaluation for ego policy with replay or policy opponents."""

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_inference import ExternalTeacher, build_external_teacher_kwargs
from ctrlsim_evaluation_metrics import (
    CTRLSIM_EGO_METRIC_FIELDS,
)
from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.nocturne_ctrlsim.core.episode_runtime import split_prepared_pack_batch
from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from envs.wrappers import ParallelAdversarialVecEnv
from evaluation.evaluation_common import (
    build_metrics_mean_row,
    build_replay_nocturne_env,
    collect_replay_nocturne_args,
    compute_solved_flag,
    extract_ctrlsim_episode_metrics,
    resolve_csv_output_path,
    start_headless_display,
    stop_headless_display,
    validate_nocturne_env_names,
    write_metrics_csv,
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
CHECKPOINT_TAR_PATTERN = re.compile(r"^(.+)\.tar$")
MULTI_CHECKPOINT_METRIC_FIELDS = (
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
SOLVED_PROGRESS_THRESHOLD = 0.85


@dataclass(frozen=True)
class EpisodeTemplate:
    """Fixed scenario and seed reused by policy-opponent evaluations."""

    scenario_id: str
    level_seed: int


@dataclass(frozen=True)
class CheckpointTarget:
    """One checkpoint name resolved within a concrete run directory."""

    checkpoint_dir: str
    model_tar: str


def parse_args():
    """Parse CLI arguments for specialized ego policy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ego policy in Nocturne-CtrlSim with replay or policy opponents."
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
        "--all_checkpoints",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Evaluate every .tar checkpoint under base_path.",
    )
    parser.add_argument(
        "--opponent_eval_mode",
        type=str,
        choices=["replay", "policy"],
        default="replay",
        help="Use GT replay or CtrlSim policy control for opponent vehicles.",
    )
    parser.add_argument(
        "--opponent_checkpoint",
        type=str,
        default=None,
        help="Optional CtrlSim checkpoint path for policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["none", "global", "per_vehicle"],
        default="none",
        help="Tilt mode for CtrlSim policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilt_range_min",
        type=float,
        default=None,
        help="Minimum sampled tilt value for policy-controlled opponents.",
    )
    parser.add_argument(
        "--tilt_range_max",
        type=float,
        default=None,
        help="Maximum sampled tilt value for policy-controlled opponents.",
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


def find_model_checkpoints(base_path: str) -> list[str]:
    """Return sorted stems for every .tar checkpoint under base_path."""
    matched = []
    for filename in os.listdir(base_path):
        match = CHECKPOINT_TAR_PATTERN.match(filename)
        if match is None:
            continue
        matched.append(match.group(1))
    matched.sort()
    return matched



def find_checkpoint_targets(base_path: str) -> list[CheckpointTarget]:
    """Return checkpoints under base_path and nested runs with local metadata."""
    root = Path(base_path)
    targets = [
        CheckpointTarget(checkpoint_dir=str(root), model_tar=model_tar)
        for model_tar in find_model_checkpoints(base_path)
    ]

    for meta_json_path in sorted(root.rglob("meta.json")):
        checkpoint_dir = meta_json_path.parent
        if checkpoint_dir == root:
            continue
        for model_tar in find_model_checkpoints(str(checkpoint_dir)):
            targets.append(
                CheckpointTarget(
                    checkpoint_dir=str(checkpoint_dir),
                    model_tar=model_tar,
                )
            )

    targets.sort(
        key=lambda target: (
            os.path.relpath(target.checkpoint_dir, base_path),
            target.model_tar,
        )
    )
    return targets


def read_scenario_ids(scenario_index_path: str) -> list[str]:
    """Read scenario IDs from a Nocturne scenario-index JSON file."""
    with open(scenario_index_path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        scenario_ids = payload.get("scenario_ids", [])
    else:
        scenario_ids = payload
    return [str(scenario_id) for scenario_id in scenario_ids]


def build_episode_templates(
    *,
    scenario_ids: Sequence[str],
    num_episodes: int,
    seed: int,
) -> list[EpisodeTemplate]:
    """Sample fixed episode templates from the evaluation seed."""
    if not scenario_ids:
        raise ValueError("scenario_ids must not be empty")
    rng = random.Random(seed)
    scenario_ids = list(scenario_ids)
    return [
        EpisodeTemplate(
            scenario_id=rng.choice(scenario_ids),
            level_seed=rng.randrange(0, 2**31),
        )
        for _ in range(num_episodes)
    ]


def build_scenario_level(template: EpisodeTemplate) -> ScenarioLevel:
    """Create a zero-tilt level for fixed policy-opponent evaluation."""
    return ScenarioLevel(
        scenario_id=template.scenario_id,
        seed=template.level_seed,
        goal_tilt=0,
        veh_veh_tilt=0,
        veh_edge_tilt=0,
    )


def _replace_obs_indices(obs: Any, replacement: Any, indices: Sequence[int]) -> Any:
    """Replace vectorized observations for selected worker indices."""
    if isinstance(obs, dict):
        for key, value in replacement.items():
            obs[key][list(indices)] = value
        return obs
    obs[list(indices)] = replacement
    return obs


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


def build_policy_opponent_teacher(
    *,
    checkpoint_path: str,
    device: str,
    cli_args: DotDict,
) -> ExternalTeacher:
    """Build the CtrlSim teacher used to control opponent vehicles."""
    teacher_kwargs = build_external_teacher_kwargs(
        checkpoint_path=checkpoint_path,
        device=device,
        inference_precision=str(cli_args.get("inference_precision", "fp32")),
        config_source=cli_args,
    )
    teacher_kwargs["base_seed"] = int(cli_args.get("seed", 1))
    return ExternalTeacher(**teacher_kwargs)


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
    opponent_eval_mode = str(cli_args.get("opponent_eval_mode", "replay"))
    external_teacher = None
    episode_templates = None
    if opponent_eval_mode == "policy":
        external_teacher = build_policy_opponent_teacher(
            checkpoint_path=str(nocturne_required["opponent_checkpoint"]),
            device=device,
            cli_args=xpid_flags,
        )
        episode_templates = build_episode_templates(
            scenario_ids=read_scenario_ids(str(nocturne_required["scenario_index_path"])),
            num_episodes=cli_args.num_episodes,
            seed=cli_args.seed,
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
            seed=cli_args.seed,
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
            episode_templates=episode_templates,
        )
    finally:
        if evaluator is not None:
            evaluator.close()
        dummy_venv.close()


class EgoReplayEvaluator(Evaluator):
    """Evaluator specialized for ego policy rollouts with replay or policy opponents."""

    def _init_parallel_envs(
        self,
        env_names,
        num_processes,
        device=None,
        record_video=False,
        **kwargs,
    ):
        """Initialize vectorized Nocturne envs for ego-policy evaluation."""
        validate_nocturne_env_names(env_names)
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}
        eval_seed = kwargs.get("seed")
        self.opponent_eval_mode = str(kwargs.pop("opponent_eval_mode", "replay"))
        opponent_runtime_mode = (
            "normal" if self.opponent_eval_mode == "policy" else "replay"
        )

        for env_name in env_names:
            env_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("video_dir", "opponent_runtime_mode")
            }
            make_fn = [
                (
                    lambda idx: lambda: build_replay_nocturne_env(
                        NocturneCtrlSimAdversarial,
                        record_video=record_video,
                        process_idx=idx,
                        opponent_runtime_mode=opponent_runtime_mode,
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
        episode_templates=None,
    ):
        """Evaluate ego policy while opponents use replay or CtrlSim policy."""
        episode_metrics = []
        use_policy_opponents = self.opponent_eval_mode == "policy"
        if use_policy_opponents and episode_templates is not None:
            return self._evaluate_fixed_episode_templates(
                agent,
                episode_templates=episode_templates,
                deterministic=deterministic,
                show_progress=show_progress,
                progress_position=progress_position,
                return_episode_returns=return_episode_returns,
                external_teacher=external_teacher,
            )

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
                prepared_batch = venv.step_prepare(action)
                if use_policy_opponents:
                    if external_teacher is None:
                        raise RuntimeError("Policy opponent evaluation requires ExternalTeacher.")
                    opponent_prepared, _ = split_prepared_pack_batch(prepared_batch)
                    model_outputs = external_teacher.run_batched_forward(opponent_prepared)
                else:
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
                    episode_metrics.append(
                        extract_ctrlsim_episode_metrics(
                            info,
                            offroad_progress_threshold=SOLVED_PROGRESS_THRESHOLD,
                        )
                    )
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

    def _evaluate_fixed_episode_templates(
        self,
        agent,
        *,
        episode_templates: Sequence[EpisodeTemplate],
        deterministic: bool,
        show_progress: bool,
        progress_position: int,
        return_episode_returns: bool,
        external_teacher,
    ):
        """Evaluate policy opponents on an explicit scenario/seed template list."""
        if external_teacher is None:
            raise RuntimeError("Policy opponent evaluation requires ExternalTeacher.")
        episode_metrics = []
        episode_returns = []
        fallback_level = build_scenario_level(episode_templates[0])

        for _env_name, venv in self.venv.items():
            completed = 0
            next_template_idx = 0
            active_templates: list[EpisodeTemplate | None] = []
            initial_levels = []
            for _ in range(self.num_processes):
                if next_template_idx < len(episode_templates):
                    template = episode_templates[next_template_idx]
                    active_templates.append(template)
                    initial_levels.append(build_scenario_level(template))
                    next_template_idx += 1
                else:
                    active_templates.append(None)
                    initial_levels.append(fallback_level)

            obs = venv.reset_to_level_batch(initial_levels)
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
                tqdm(total=len(episode_templates), position=progress_position)
                if show_progress
                else None
            )
            try:
                while completed < len(episode_templates):
                    with torch.no_grad():
                        _, action, _, recurrent_hidden_states = agent.act(
                            obs,
                            recurrent_hidden_states,
                            masks,
                            deterministic=deterministic,
                        )

                    action = agent.process_action(action.cpu().numpy())
                    prepared_batch = venv.step_prepare(action)
                    opponent_prepared, _ = split_prepared_pack_batch(prepared_batch)
                    model_outputs = external_teacher.run_batched_forward(opponent_prepared)
                    obs, _reward, done, infos = venv.step_complete(
                        model_outputs,
                        reset_random=False,
                        auto_reset_on_done=False,
                    )

                    reset_levels = []
                    reset_indices = []
                    for idx, info in enumerate(infos):
                        if not done[idx]:
                            continue
                        template = active_templates[idx]
                        if template is not None and "episode" in info:
                            episode_returns.append(info["episode"]["r"])
                            episode_metrics.append(
                                extract_ctrlsim_episode_metrics(
                                    info,
                                    offroad_progress_threshold=SOLVED_PROGRESS_THRESHOLD,
                                )
                            )
                            completed += 1
                            if pbar:
                                pbar.update(1)

                        if next_template_idx < len(episode_templates):
                            next_template = episode_templates[next_template_idx]
                            active_templates[idx] = next_template
                            reset_levels.append(build_scenario_level(next_template))
                            next_template_idx += 1
                        else:
                            active_templates[idx] = None
                            reset_levels.append(fallback_level)
                        reset_indices.append(idx)

                    if reset_indices:
                        reset_obs = venv.reset_to_level_indices(
                            reset_levels,
                            reset_indices,
                        )
                        obs = _replace_obs_indices(obs, reset_obs, reset_indices)

                    masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if agent.is_recurrent:
                        for idx, done_ in enumerate(done):
                            if done_:
                                recurrent_hidden_states[0][idx].zero_()
                                recurrent_hidden_states[1][idx].zero_()
            finally:
                if pbar:
                    pbar.close()

        if return_episode_returns:
            return episode_metrics, episode_returns
        return episode_metrics


def run_all_checkpoint_evaluations(
    *,
    base_path: str,
    result_path: str,
    env_names: list[str],
    device: str,
    cli_args: DotDict,
) -> None:
    """Evaluate every checkpoint under base_path and nested run directories."""
    checkpoint_targets = find_checkpoint_targets(base_path)
    if not checkpoint_targets:
        raise FileNotFoundError(
            f"No .tar checkpoints found under {base_path}"
        )

    pending_targets = []
    for target in checkpoint_targets:
        target_result_path = os.path.join(target.checkpoint_dir, "evaluation")
        result_file = os.path.join(target_result_path, f"eval-{target.model_tar}.csv")
        if os.path.exists(result_file):
            continue
        pending_targets.append((target, target_result_path, result_file))

    cli_args.record_video = False
    pbar = (
        tqdm(total=len(pending_targets), position=0)
        if cli_args.verbose
        else None
    )
    try:
        for target, target_result_path, result_file in pending_targets:
            os.makedirs(target_result_path, exist_ok=True)
            if pbar is not None:
                relative_dir = os.path.relpath(target.checkpoint_dir, base_path)
                label = target.model_tar if relative_dir == "." else f"{relative_dir}/{target.model_tar}"
                pbar.set_description_str(f"Evaluating {label}")
            episode_metrics = run_single_checkpoint_evaluation(
                base_path=target.checkpoint_dir,
                model_tar=target.model_tar,
                env_names=env_names,
                device=device,
                cli_args=cli_args,
                video_dir=target_result_path,
                progress_position=1,
            )
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


def main():
    """Run specialized ego policy evaluation with replay or policy opponents."""
    os.environ["OMP_NUM_THREADS"] = "1"
    args = DotDict(vars(parse_args()))
    args.setdefault("all_checkpoints", False)
    args.setdefault("record_video", False)
    args.setdefault("opponent_eval_mode", "replay")
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

        if getattr(args, "all_checkpoints", False):
            run_all_checkpoint_evaluations(
                base_path=base_path,
                result_path=result_path,
                env_names=env_names,
                device=device,
                cli_args=args,
            )
            return

        result_fname = f"ego-policy-eval-{args.model_tar}"
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
