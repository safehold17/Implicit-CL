"""Sweep fixed CtrlSim tilting parameters for ego-policy evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

from ctrlsim_evaluation_metrics import CTRLSIM_EGO_METRIC_FIELDS
from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.nocturne_ctrlsim.core.episode_runtime import split_prepared_pack_batch
from envs.nocturne_ctrlsim.core.level import ScenarioLevel
from envs.wrappers import ParallelAdversarialVecEnv
from evaluation.ego_policy_evaluation import (
    PREPROCESS_DIR,
    SCENARIO_DATA_DIR,
    SCENARIO_INDEX_PATH,
    SOLVED_PROGRESS_THRESHOLD,
    VEHICLE_MAP_PATH,
    build_policy_opponent_teacher,
    load_agent_from_checkpoint,
)
from evaluation.evaluation_common import (
    build_replay_nocturne_env,
    compute_solved_flag,
    extract_ctrlsim_episode_metrics,
    resolve_csv_output_path,
    start_headless_display,
    stop_headless_display,
    validate_nocturne_env_names,
)
from evaluation.eval import Evaluator
from util import DotDict, ignore_warning, is_discrete_actions, str2bool
from util.eval_helper import set_eval_worker_seeds

ignore_warning.configure_subprocess_env()

TILT_SUMMARY_FIELDS = (
    "tilt_value",
    "goal_tilt",
    "veh_veh_tilt",
    "veh_edge_tilt",
    "num_episodes",
    "collision",
    "offroad",
    "position_reached",
    "progress",
    "solved",
    "total_episode_reward",
    *CTRLSIM_EGO_METRIC_FIELDS,
)
METRIC_MEAN_FIELDS = TILT_SUMMARY_FIELDS[5:]
TILT_MIN_BOUND = -25
TILT_MAX_BOUND = 25


@dataclass(frozen=True)
class EpisodeTemplate:
    """Fixed scenario and seed reused across all tilt values."""

    scenario_id: str
    level_seed: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ego-policy tilting-parameter sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one ego-policy checkpoint while policy-controlled "
            "surrounding vehicles use fixed CtrlSim tilting parameters."
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
        help="Number of evaluation episodes per tilt value.",
    )
    parser.add_argument(
        "--model_tar",
        type=str,
        default="model",
        help="Checkpoint stem without the .tar suffix.",
    )
    parser.add_argument(
        "--opponent_checkpoint",
        type=str,
        default=None,
        help="CtrlSim teacher checkpoint for surrounding vehicles.",
    )
    parser.add_argument(
        "--enable_goal_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Apply the current sweep value to goal tilt.",
    )
    parser.add_argument(
        "--enable_veh_veh_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Apply the current sweep value to vehicle-vehicle tilt.",
    )
    parser.add_argument(
        "--enable_veh_edge_tilt",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Apply the current sweep value to vehicle-edge tilt.",
    )
    parser.add_argument(
        "--tilt_min",
        type=int,
        default=-25,
        help="Minimum fixed tilt value to evaluate.",
    )
    parser.add_argument(
        "--tilt_max",
        type=int,
        default=25,
        help="Maximum fixed tilt value to evaluate.",
    )
    parser.add_argument(
        "--tilt_interval",
        type=int,
        default=1,
        help="Step size between fixed tilt values.",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Evaluate the ego policy greedily.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Show progress bars.",
    )
    parser.add_argument(
        "--record_video",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Record video for the first evaluation process.",
    )
    return parser.parse_args()


def build_tilt_values(
    *,
    tilt_min: int,
    tilt_max: int,
    tilt_interval: int,
) -> list[int]:
    """Return descending integer tilt values including both endpoints."""
    if tilt_interval <= 0:
        raise ValueError("tilt_interval must be positive")
    if tilt_min > tilt_max:
        raise ValueError("tilt_min must be <= tilt_max")
    if tilt_min < TILT_MIN_BOUND or tilt_max > TILT_MAX_BOUND:
        raise ValueError("tilt values must stay within [-25, 25]")

    values = list(range(tilt_max, tilt_min - 1, -tilt_interval))
    if values[-1] != tilt_min:
        values.append(tilt_min)
    return values


def build_tilt_tuple(
    *,
    value: int,
    enable_goal_tilt: bool,
    enable_veh_veh_tilt: bool,
    enable_veh_edge_tilt: bool,
) -> tuple[int, int, int]:
    """Apply a sweep value to enabled tilt dimensions and zero the others."""
    return (
        value if enable_goal_tilt else 0,
        value if enable_veh_veh_tilt else 0,
        value if enable_veh_edge_tilt else 0,
    )


def build_tilt_sweep_progress_desc() -> str:
    """Return the outer progress-bar label for the tilt sweep."""
    return "tilt sweep"


def build_tilt_episode_progress_desc(value: int) -> str:
    """Return the inner progress-bar label for one tilt value."""
    return f"tilt={value} episodes"


def build_episode_templates(
    *,
    scenario_ids: Sequence[str],
    num_episodes: int,
    seed: int,
) -> list[EpisodeTemplate]:
    """Sample fixed episode templates reused by every tilt value."""
    if not scenario_ids:
        raise ValueError("scenario_ids must not be empty")
    rng = random.Random(seed)
    return [
        EpisodeTemplate(
            scenario_id=rng.choice(list(scenario_ids)),
            level_seed=rng.randrange(0, 2**31),
        )
        for _ in range(num_episodes)
    ]


def read_scenario_ids(scenario_index_path: str) -> list[str]:
    """Read scenario IDs from a Nocturne scenario-index JSON file."""
    with open(scenario_index_path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        scenario_ids = payload.get("scenario_ids", [])
    else:
        scenario_ids = payload
    return [str(scenario_id) for scenario_id in scenario_ids]


def build_scenario_level(
    template: EpisodeTemplate,
    tilt_tuple: tuple[int, int, int],
) -> ScenarioLevel:
    """Create the explicit ScenarioLevel used for one rollout."""
    return ScenarioLevel(
        scenario_id=template.scenario_id,
        seed=template.level_seed,
        goal_tilt=tilt_tuple[0],
        veh_veh_tilt=tilt_tuple[1],
        veh_edge_tilt=tilt_tuple[2],
    )


def build_tilt_mean_row(
    *,
    tilt_value: int,
    tilt_tuple: tuple[int, int, int],
    episode_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Average episode metrics for one fixed tilt value."""
    row: dict[str, float | int] = {
        "tilt_value": tilt_value,
        "goal_tilt": tilt_tuple[0],
        "veh_veh_tilt": tilt_tuple[1],
        "veh_edge_tilt": tilt_tuple[2],
        "num_episodes": len(episode_metrics),
    }
    for field in METRIC_MEAN_FIELDS:
        values = [float(metrics[field]) for metrics in episode_metrics]
        row[field] = sum(values) / len(values) if values else 0.0
    return row


def write_tilt_summary_csv(
    output_path: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write one summary row per evaluated tilt value."""
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TILT_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TILT_SUMMARY_FIELDS})


def _replace_obs_indices(obs: Any, replacement: Any, indices: Sequence[int]) -> Any:
    """Replace vectorized observations for reset worker indices."""
    if isinstance(obs, dict):
        for key, value in replacement.items():
            obs[key][list(indices)] = value
        return obs
    obs[list(indices)] = replacement
    return obs


class TiltingParameterEvaluator(Evaluator):
    """Evaluator for fixed-tilt sweeps with policy-controlled opponents."""

    def _init_parallel_envs(
        self,
        env_names,
        num_processes,
        device=None,
        record_video=False,
        **kwargs,
    ) -> None:
        """Initialize vectorized Nocturne envs in policy-opponent mode."""
        validate_nocturne_env_names(env_names)
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}
        eval_seed = kwargs.get("seed")

        for env_name in env_names:
            env_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("video_dir", "opponent_runtime_mode")
            }
            make_fn = [
                (
                    lambda idx: lambda: build_replay_nocturne_env(
                        NocturneCtrlSimAdversarial,
                        record_video=record_video,
                        process_idx=idx,
                        opponent_runtime_mode="normal",
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

    def evaluate_tilt(
        self,
        agent,
        *,
        templates: Sequence[EpisodeTemplate],
        tilt_tuple: tuple[int, int, int],
        external_teacher,
        deterministic: bool,
        show_progress: bool,
        progress_desc: str | None = None,
        progress_position: int = 0,
    ) -> list[dict[str, float | str]]:
        """Evaluate all templates once for a fixed tilt tuple."""
        if external_teacher is None:
            raise RuntimeError("Tilting parameter evaluation requires ExternalTeacher.")

        all_metrics = []
        fallback_level = build_scenario_level(templates[0], tilt_tuple)

        for _env_name, venv in self.venv.items():
            next_template_idx = 0
            active_templates: list[EpisodeTemplate | None] = []
            initial_levels = []
            for _ in range(self.num_processes):
                if next_template_idx < len(templates):
                    template = templates[next_template_idx]
                    active_templates.append(template)
                    initial_levels.append(build_scenario_level(template, tilt_tuple))
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
                tqdm(
                    total=len(templates),
                    position=progress_position,
                    desc=progress_desc,
                    leave=False,
                )
                if show_progress
                else None
            )

            try:
                while len(all_metrics) < len(templates):
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
                            metrics = extract_ctrlsim_episode_metrics(
                                info,
                                offroad_progress_threshold=SOLVED_PROGRESS_THRESHOLD,
                            )
                            metrics["solved"] = compute_solved_flag(
                                progress=float(metrics["progress"]),
                                collision=float(metrics["collision"]),
                                offroad=float(metrics["offroad"]),
                                progress_threshold=SOLVED_PROGRESS_THRESHOLD,
                            )
                            all_metrics.append(metrics)
                            if pbar is not None:
                                pbar.update(1)

                        if next_template_idx < len(templates):
                            next_template = templates[next_template_idx]
                            active_templates[idx] = next_template
                            reset_levels.append(
                                build_scenario_level(next_template, tilt_tuple)
                            )
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
                if pbar is not None:
                    pbar.close()

        return all_metrics[: len(templates)]


def run_tilting_parameter_evaluation(
    *,
    base_path: str,
    model_tar: str,
    env_names: list[str],
    device: str,
    cli_args: DotDict,
    result_path: str,
) -> str:
    """Run the fixed-tilt sweep and return the summary CSV path."""
    cli_args.opponent_eval_mode = "policy"
    cli_args.tilting_mode = "global"
    cli_args.tilt_range_min = cli_args.tilt_min
    cli_args.tilt_range_max = cli_args.tilt_max

    agent, xpid_flags, nocturne_required, dummy_venv = load_agent_from_checkpoint(
        base_path=base_path,
        model_tar=model_tar,
        env_name=env_names[0],
        device=device,
        cli_args=cli_args,
    )
    xpid_flags.update(cli_args)
    xpid_flags.update({"use_skip": False})
    scenario_ids = read_scenario_ids(str(nocturne_required["scenario_index_path"]))
    templates = build_episode_templates(
        scenario_ids=scenario_ids,
        num_episodes=cli_args.num_episodes,
        seed=cli_args.seed,
    )
    external_teacher = build_policy_opponent_teacher(
        checkpoint_path=str(nocturne_required["opponent_checkpoint"]),
        device=device,
        cli_args=xpid_flags,
    )
    tilt_values = build_tilt_values(
        tilt_min=cli_args.tilt_min,
        tilt_max=cli_args.tilt_max,
        tilt_interval=cli_args.tilt_interval,
    )

    evaluator = None
    try:
        evaluator = TiltingParameterEvaluator(
            env_names,
            num_processes=cli_args.num_processes,
            num_episodes=cli_args.num_episodes,
            frame_stack=xpid_flags.frame_stack,
            grayscale=xpid_flags.grayscale,
            use_global_critic=xpid_flags.use_global_critic,
            video_dir=result_path,
            seed=cli_args.seed,
            **nocturne_required,
            record_video=cli_args.record_video,
        )
        rows = []
        pbar = (
            tqdm(
                total=len(tilt_values),
                position=0,
                desc=build_tilt_sweep_progress_desc(),
            )
            if cli_args.verbose
            else None
        )
        try:
            for value in tilt_values:
                tilt_tuple = build_tilt_tuple(
                    value=value,
                    enable_goal_tilt=cli_args.enable_goal_tilt,
                    enable_veh_veh_tilt=cli_args.enable_veh_veh_tilt,
                    enable_veh_edge_tilt=cli_args.enable_veh_edge_tilt,
                )
                episode_metrics = evaluator.evaluate_tilt(
                    agent,
                    templates=templates,
                    tilt_tuple=tilt_tuple,
                    external_teacher=external_teacher,
                    deterministic=cli_args.deterministic,
                    show_progress=cli_args.verbose,
                    progress_desc=build_tilt_episode_progress_desc(value),
                    progress_position=1,
                )
                rows.append(
                    build_tilt_mean_row(
                        tilt_value=value,
                        tilt_tuple=tilt_tuple,
                        episode_metrics=episode_metrics,
                    )
                )
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        output_path = resolve_csv_output_path(
            result_path,
            f"ego-policy-tilting-parameter-{model_tar}",
        )
        write_tilt_summary_csv(output_path, rows)
        return output_path
    finally:
        if evaluator is not None:
            evaluator.close()
        dummy_venv.close()


def main() -> None:
    """Run ego-policy evaluation across fixed teacher tilting parameters."""
    os.environ["OMP_NUM_THREADS"] = "1"
    args = DotDict(vars(parse_args()))
    args.num_processes = min(args.num_processes, args.num_episodes)

    display = start_headless_display()
    try:
        device = "cuda"
        base_path = os.path.expandvars(os.path.expanduser(args.base_path))
        result_path = os.path.join(base_path, "evaluation")
        os.makedirs(result_path, exist_ok=True)

        env_names = args.env_names.split(",")
        validate_nocturne_env_names(env_names)
        if args.record_video and len(env_names) != 1:
            raise ValueError("--record_video requires exactly one env_name")
        if args.record_video:
            args.num_processes = 1

        run_tilting_parameter_evaluation(
            base_path=base_path,
            model_tar=args.model_tar,
            env_names=env_names,
            device=device,
            cli_args=args,
            result_path=result_path,
        )
    finally:
        stop_headless_display(display)


if __name__ == "__main__":
    main()
