#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ctrlsim_adapter.opponent_vehicle import CtrlSimOpponentAdapter
from ctrlsim_adapter.opponent_vehicle.inference_bridge import capture_sampling_rng_state
from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.wrappers import ParallelAdversarialVecEnv
from eval import Evaluator
from util import is_discrete_actions


class CtrlSimEgoWrapper:
    """
    VecEnv-friendly wrapper that drives ego with CtrlSimOpponentAdapter.

    The external action is ignored; ego actions are produced by CtrlSimOpponentAdapter.
    """

    def __init__(
        self,
        scenario_index_path: str,
        opponent_checkpoint: str,
        scenario_data_dir: str,
        preprocess_dir: str,
        vehicle_map_path: str,
        opponent_k: int,
        max_episode_steps: int,
        tilting_mode: str,
        tilt_range,
        show_level_log: bool,
        record_video: bool,
        show_vehicle_ids: bool,
        output_dir: str,
        xpid: str,
        device: str = "cuda",
        seed: int = 0,
        inference_precision: str = "fp32",
        action_repeat_interval: int = 2,
        sparse_inference_action_repeat: bool = False,
        **_kwargs,
    ):
        self.env = NocturneCtrlSimAdversarial(
            scenario_index_path=scenario_index_path,
            opponent_checkpoint=opponent_checkpoint,
            scenario_data_dir=scenario_data_dir,
            preprocess_dir=preprocess_dir,
            vehicle_map_path=vehicle_map_path,
            opponent_k=opponent_k,
            max_episode_steps=max_episode_steps,
            device=device,
            seed=seed,
            tilting_mode=tilting_mode,
            tilt_range=tilt_range,
            inference_precision=inference_precision,
            action_repeat_interval=action_repeat_interval,
            sparse_inference_action_repeat=sparse_inference_action_repeat,
        )
        self.tilting_mode = tilting_mode
        self.device = device
        self.checkpoint_path = opponent_checkpoint
        self.show_level_log = show_level_log
        self.record_video = record_video
        self.show_vehicle_ids = show_vehicle_ids
        self.output_dir = output_dir
        self.xpid = xpid
        self.inference_precision = inference_precision
        self.episode_idx = 0
        self.goal_pos_tolerance = 1.0
        self._opponent_reached_goal_ids = set()
        self._episode_position_reached = False
        self._single_env_teacher = None

        self.ego_adapter = CtrlSimOpponentAdapter(
            cfg=self.env.cfg,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.ego_adapter.set_tilting(0, 0, 0)
        self.ego_id = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = getattr(self.env, "spec", None)

    def _reset_ego_adapter(self):
        self.ego_id = self.env.ego_vehicle.getID() if self.env.ego_vehicle else None
        self.ego_adapter._veh_id_to_preproc_idx = dict(
            getattr(self.env, "_veh_id_to_preproc_idx", {})
        )
        self.ego_adapter.reset(
            self.env.scenario,
            self.env.vehicles,
            self.env._gt_data_dict,
            self.env._preproc_data,
            [self.ego_id] if self.ego_id is not None else [],
        )

    def _maybe_disable_opponent_tilting(self):
        # none: force opponent tilting to zero
        if self.tilting_mode == "none":
            self.env.opponent.set_tilting(0, 0, 0)

    def _log_level(self):
        if not self.show_level_log:
            return
        level = self.env.current_level
        if level is None:
            return
        print(
            "Level: scenario_id=%s seed=%s goal=%s veh_veh=%s veh_edge=%s"
            % (
                level.scenario_id,
                level.seed,
                level.goal_tilt,
                level.veh_veh_tilt,
                level.veh_edge_tilt,
            )
        )

    def _start_recording(self):
        if not self.record_video:
            return
        out_dir = os.path.join(self.output_dir, self.xpid)
        os.makedirs(out_dir, exist_ok=True)
        self.env.start_recording(
            out_dir,
            str(self.episode_idx),
            show_vehicle_ids=self.show_vehicle_ids,
        )

    def _stop_recording(self):
        if not self.record_video:
            return
        self.env.stop_recording(str(self.episode_idx))
        self.episode_idx += 1

    def _reset_episode_state(self):
        self._opponent_reached_goal_ids = set()
        self._episode_position_reached = False

    def _get_goal_pos(self, veh_id):
        goal_points = getattr(self.env, "_goal_points_by_id", None)
        if goal_points and veh_id in goal_points:
            return goal_points[veh_id]
        if veh_id == self.ego_id and getattr(self.env, "_ego_goal_dict", None) is not None:
            return self.env._ego_goal_dict.get("pos")
        return None

    def _is_within_goal(self, veh, goal_pos):
        if veh is None or goal_pos is None:
            return False
        pos = veh.getPosition()
        dist = np.linalg.norm(goal_pos - np.array([pos.x, pos.y]))
        return dist < self.goal_pos_tolerance

    def _stop_vehicle(self, veh):
        if veh is None:
            return "none"
        stopped = False
        stop_mode = "unknown"
        for method in ("set_speed", "setSpeed"):
            if hasattr(veh, method):
                try:
                    getattr(veh, method)(0.0)
                    stopped = True
                    stop_mode = method
                    break
                except Exception:
                    pass
        if not stopped:
            try:
                self.env.opponent.apply_action(veh, (-10.0, 0.0))
                stop_mode = "brake_action"
            except Exception:
                try:
                    veh.brake(10.0)
                    stop_mode = "brake_method"
                except Exception:
                    stop_mode = "unknown"
        try:
            veh.steering = 0.0
        except Exception:
            pass
        return stop_mode

    def _update_opponent_stop_states(self):
        for veh in self.env.opponent_vehicles:
            if veh is None:
                continue
            veh_id = veh.getID()
            goal_pos = self._get_goal_pos(veh_id)
            if goal_pos is None:
                continue
            if veh_id in self._opponent_reached_goal_ids:
                self._stop_vehicle(veh)
                continue
            if self._is_within_goal(veh, goal_pos):
                self._opponent_reached_goal_ids.add(veh_id)
                self._stop_vehicle(veh)

    def _ego_reached_goal(self):
        if self.ego_id is None or self.env.ego_vehicle is None:
            return False
        goal_pos = self._get_goal_pos(self.ego_id)
        return self._is_within_goal(self.env.ego_vehicle, goal_pos)

    def reset_random(self):
        obs = self.env.reset_random()
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        self._log_level()
        self._start_recording()
        return obs

    def reset(self):
        obs = self.env.reset()
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        self._log_level()
        self._start_recording()
        return obs

    def reset_agent(self):
        obs = self.env.reset_agent()
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        return obs

    def _postprocess_step(self, obs, reward, done, info):
        """Goal check, opponent stop, recording stop, info enrichment."""
        self._update_opponent_stop_states()
        position_reached = self._ego_reached_goal()
        if position_reached:
            self._episode_position_reached = True

        if position_reached:
            if not done:
                done = True
                if hasattr(self.env, "opponent"):
                    self.env.opponent.finalize(self.env.vehicles)
                if hasattr(self.env, "_get_info"):
                    info = self.env._get_info()
                else:
                    info = dict(info)
                    info.setdefault(
                        "episode",
                        {"r": reward, "l": self.env.current_step},
                    )
        info = dict(info)
        info["position_reached"] = float(position_reached)
        info["position_reached_occurred"] = float(self._episode_position_reached)
        if done:
            self._stop_recording()
        return obs, reward, done, info

    def _apply_ego_action(self, accel: float, steer: float) -> None:
        ego_veh = self.env.ego_vehicle
        if ego_veh is None:
            return
        if accel > 0:
            ego_veh.acceleration = accel
        else:
            ego_veh.brake(abs(accel))
        ego_veh.steering = steer

    def _step_with_ego_action(self, accel: float, steer: float, opponent_actions):
        self.env.current_step += 1
        self._apply_ego_action(accel, steer)
        obs, reward, done, info = self.env._step_post_actions(opponent_actions)
        self.ego_adapter.record_all_actions(
            self.env.current_step - 1,
            self.env.vehicles,
            {self.ego_id: (accel, steer)} if self.ego_id is not None else {},
        )
        return self._postprocess_step(obs, reward, done, info)

    def step(self, _action):
        prepared = self.step_prepare(_action)
        teacher = self._get_single_env_teacher()
        ego_outputs = teacher.batched_forward([prepared["ego"]])[0]
        opp_outputs = teacher.batched_forward([prepared["opponent"]])[0]
        return self.step_complete({"ego": ego_outputs, "opponent": opp_outputs})

    def _get_single_env_teacher(self):
        teacher = self._single_env_teacher
        if teacher is not None:
            return teacher

        from batch_inference import ExternalTeacher

        teacher = ExternalTeacher(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            inference_precision=self.inference_precision,
        )
        self._single_env_teacher = teacher
        return teacher

    # ========== Batch inference two-phase step ==========

    def step_prepare(self, _action):
        """Phase 1: Prepare inference data for both ego and opponent ctrlsim_adapter."""
        t = self.env.current_step
        worker_rng_state = capture_sampling_rng_state(self.device)
        ego_prepared = self.ego_adapter.prepare_step(
            t,
            self.env.vehicles,
            worker_rng_state=worker_rng_state,
        )
        runtime_mode = getattr(self.env, "opponent_runtime_mode", "normal")
        if runtime_mode == "normal" and len(self.env.opponent_vehicle_ids) > 0:
            opp_prepared = self.env.opponent.prepare_step(
                t,
                self.env.vehicles,
                worker_rng_state=worker_rng_state,
            )
        else:
            opp_prepared = None
        return {"ego": ego_prepared, "opponent": opp_prepared}

    def step_complete(self, model_outputs):
        """Phase 2: Apply predictions, step simulation, postprocess."""
        ego_outputs = model_outputs.get("ego")
        opp_outputs = model_outputs.get("opponent")

        # 1. Apply ego predictions → get physical ego action
        ego_actions = self.ego_adapter.apply_predictions(ego_outputs)
        if self.ego_id is not None and self.ego_id in ego_actions:
            accel, steer = ego_actions[self.ego_id]
        else:
            accel, steer = 0.0, 0.0

        # 2. Apply opponent predictions
        runtime_mode = getattr(self.env, "opponent_runtime_mode", "normal")
        if runtime_mode == "normal" and len(self.env.opponent_vehicle_ids) > 0:
            opponent_actions = self.env.opponent.apply_predictions(opp_outputs)
        else:
            opponent_actions = {}
        return self._step_with_ego_action(accel, steer, opponent_actions)

    def close(self):
        self.env.close()


class DummyEvalAgent:
    """No-op agent adapter for Evaluator.evaluate()."""

    def __init__(self, action_dim: int):
        self.algo = SimpleNamespace(
            actor_critic=SimpleNamespace(
                recurrent_hidden_state_size=1,
                is_recurrent=False,
            )
        )
        self.action_dim = action_dim

    def act(self, obs, recurrent_hidden_states, masks, deterministic=False):
        action = torch.zeros((obs.shape[0], self.action_dim), device=obs.device)
        return None, action, None, recurrent_hidden_states

    def process_action(self, action):
        return action

    @property
    def is_recurrent(self):
        return False


class CtrlSimEvaluator(Evaluator):
    @staticmethod
    def _make_env(env_name, record_video=False, **kwargs):
        _ = env_name
        kwargs["record_video"] = record_video
        return CtrlSimEgoWrapper(**kwargs)

    def _init_parallel_envs(self, env_names, num_processes, device=None, record_video=False, **kwargs):
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name: None for env_name in env_names}
        base_seed = kwargs.get("seed")

        for env_name in env_names:
            make_fn = []
            for process_id in range(self.num_processes):
                env_kwargs = dict(kwargs)
                if base_seed is not None:
                    env_kwargs["seed"] = int(base_seed) + process_id
                make_fn.append(
                    lambda env_name=env_name, env_kwargs=env_kwargs: self._make_env(
                        env_name,
                        record_video=record_video,
                        device=device,
                        **env_kwargs,
                    )
                )
            venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            venv = Evaluator.wrap_venv(venv, env_name, device=device)
            self.venv[env_name] = venv

        self.is_discrete_actions = is_discrete_actions(self.venv[env_names[0]])


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
        help="Progress threshold used in solved metric: solved if progress > threshold or position_reached.",
    )
    parser.add_argument(
        "--tilting_mode",
        type=str,
        choices=["global", "per_vehicle", "ego", "none"],
        default="per_vehicle",
    )
    parser.add_argument(
        "--tilt_range",
        type=float,
        nargs=2,
        default=[-25.0, 25.0],
        metavar=("MIN", "MAX"),
        help="Tilt sampling range for Nocturne, formatted as: MIN MAX (e.g., -25 -10).",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--show_vehicle_ids", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    parser.add_argument(
        "--action_repeat_interval",
        type=int,
        default=2,
        help="Action repeat cycle length N (the last step in each cycle repeats the previous action).",
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


def _build_tilting_columns(info, tilting_mode):
    opp_count = 7
    tilts = []
    ego_goal_tilt = 0.0
    ego_veh_veh_tilt = 0.0
    ego_veh_edge_tilt = 0.0

    if tilting_mode == "per_vehicle":
        for i in range(opp_count):
            g = info.get(f"per_vehicle_goal_tilt_{i}", 0.0)
            v = info.get(f"per_vehicle_veh_tilt_{i}", 0.0)
            e = info.get(f"per_vehicle_edge_tilt_{i}", 0.0)
            tilts.append((float(g), float(v), float(e)))
    elif tilting_mode == "global":
        g = float(info.get("goal_tilt", 0.0))
        v = float(info.get("veh_veh_tilt", 0.0))
        e = float(info.get("veh_edge_tilt", 0.0))
        for _ in range(opp_count):
            tilts.append((g, v, e))
    elif tilting_mode == "ego":
        ego_goal_tilt = float(info.get("goal_tilt", 0.0))
        ego_veh_veh_tilt = float(info.get("veh_veh_tilt", 0.0))
        ego_veh_edge_tilt = float(info.get("veh_edge_tilt", 0.0))
        for _ in range(opp_count):
            tilts.append((0.0, 0.0, 0.0))
    else:
        for _ in range(opp_count):
            tilts.append((0.0, 0.0, 0.0))

    goal_sum = 0.0
    goal_valid = 0
    veh_veh_sum = 0.0
    veh_veh_valid = 0
    veh_edge_sum = 0.0
    veh_edge_valid = 0
    for g, v, e in tilts:
        if g != 0.0:
            goal_valid += 1
            goal_sum += g
        if v != 0.0:
            veh_veh_valid += 1
            veh_veh_sum += v
        if e != 0.0:
            veh_edge_valid += 1
            veh_edge_sum += e

    veh_goal_avg = round(goal_sum / goal_valid, 2) if goal_valid > 0 else 0.0
    veh_veh_avg = round(veh_veh_sum / veh_veh_valid, 2) if veh_veh_valid > 0 else 0.0
    veh_edge_avg = round(veh_edge_sum / veh_edge_valid, 2) if veh_edge_valid > 0 else 0.0

    columns = {}
    for i, (g, v, e) in enumerate(tilts):
        columns[f"opp{i}_goal_tilt"] = g
        columns[f"opp{i}_veh_veh_tilt"] = v
        columns[f"opp{i}_veh_edge_tilt"] = e
    columns["veh_goal_avg"] = float(veh_goal_avg)
    columns["veh_veh_avg"] = float(veh_veh_avg)
    columns["veh_edge_avg"] = float(veh_edge_avg)
    columns["ego_goal_tilt"] = ego_goal_tilt
    columns["ego_veh_veh_tilt"] = ego_veh_veh_tilt
    columns["ego_veh_edge_tilt"] = ego_veh_edge_tilt
    return columns


def _extract_episode_metrics(
    info,
    episode_return,
    solved_threshold,
    progress_threshold,
    tilting_mode,
):
    if "episode" in info:
        episode_return = info["episode"].get("r", episode_return)
    _ = solved_threshold
    collision = info.get("collision_occurred", info.get("collision", 0.0))
    goal_reached = info.get("goal_reached_occurred", info.get("goal_reached", 0.0))
    position_reached = info.get(
        "position_reached_occurred", info.get("position_reached", 0.0)
    )
    offroad = info.get("offroad_occurred", info.get("offroad", 0.0))
    progress = info.get("avg_progress", info.get("progress", 0.0))
    solved = (
        1.0
        if (float(progress) > float(progress_threshold) or float(position_reached) > 0.0)
        else 0.0
    )

    metrics = {
        "scenario_id": info.get("scenario_id", ""),
        "seed": info.get("seed", ""),
        "test_returns": float(episode_return),
        "solved": float(solved),
        "collision": float(collision),
        "goal_reached": float(goal_reached),
        "position_reached": float(position_reached),
        "offroad": float(offroad),
        "progress": float(progress),
    }
    metrics.update(_build_tilting_columns(info, tilting_mode))
    return metrics


def evaluate_with_metrics(
    evaluator,
    agent,
    deterministic,
    show_progress,
    render,
    tilting_mode,
    progress_threshold,
    external_teacher=None,
):
    env_name = evaluator.env_names[0]
    venv = evaluator.venv[env_name]
    num_episodes = evaluator.num_episodes
    solved_threshold = evaluator.solved_threshold

    if env_name.startswith("Nocturne") and hasattr(venv, "reset_random"):
        obs = venv.reset_random()
    else:
        obs = venv.reset()

    recurrent_hidden_states = torch.zeros(
        evaluator.num_processes,
        agent.algo.actor_critic.recurrent_hidden_state_size,
        device=evaluator.device,
    )
    masks = torch.ones(evaluator.num_processes, 1, device=evaluator.device)

    episode_metrics = []
    pbar = tqdm(total=num_episodes) if show_progress else None

    while len(episode_metrics) < num_episodes:
        with torch.no_grad():
            # No-op action; ego is driven by CtrlSimEgoWrapper using ctrl-sim observations.
            action = torch.zeros(
                (obs.shape[0], agent.action_dim),
                device=obs.device,
            )

        action = action.cpu().numpy()
        if not evaluator.is_discrete_actions:
            action = agent.process_action(action)

        if external_teacher is not None:
            per_env_prepared = venv.step_prepare(action)
            ego_prepared = [p.get("ego") if p else None for p in per_env_prepared]
            opp_prepared = [p.get("opponent") if p else None for p in per_env_prepared]
            ego_results = external_teacher.batched_forward(ego_prepared)
            opp_results = external_teacher.batched_forward(opp_prepared)
            combined = [
                {"ego": e, "opponent": o}
                for e, o in zip(ego_results, opp_results)
            ]
            obs, reward, done, infos = venv.step_complete(combined, reset_random=True)
        else:
            obs, reward, done, infos = venv.step(action)

        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=evaluator.device,
        )

        for info in infos:
            if "episode" in info.keys():
                metrics = _extract_episode_metrics(
                    info,
                    info["episode"]["r"],
                    solved_threshold,
                    progress_threshold,
                    tilting_mode,
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


def _tilting_fields():
    fields = []
    for i in range(7):
        fields.append(f"opp{i}_goal_tilt")
        fields.append(f"opp{i}_veh_veh_tilt")
        fields.append(f"opp{i}_veh_edge_tilt")
    fields.append("veh_goal_avg")
    fields.append("veh_veh_avg")
    fields.append("veh_edge_avg")
    fields.append("ego_goal_tilt")
    fields.append("ego_veh_veh_tilt")
    fields.append("ego_veh_edge_tilt")
    return fields


def write_metrics_csv(output_dir, xpid, episode_metrics):
    out_dir = os.path.join(output_dir, xpid)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    fields = [
        "episode",
        "scenario_id",
        "seed",
        "test_returns",
        "solved",
        "collision",
        "goal_reached",
        "position_reached",
        "offroad",
        "progress",
    ] + _tilting_fields()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, metrics in enumerate(episode_metrics):
            row = {"episode": idx}
            row.update(metrics)
            writer.writerow(row)

        if episode_metrics:
            avg = {"episode": "avg"}
            non_avg_fields = {"scenario_id", "seed"}
            for field in fields:
                if field == "episode":
                    avg[field] = "avg"
                    continue
                if field in non_avg_fields:
                    avg[field] = ""
                    continue
                mean_value = float(np.mean([m[field] for m in episode_metrics]))
                avg[field] = f"{mean_value:.2f}"
            writer.writerow(avg)

    return csv_path


def main() -> None:
    args = parse_args()
    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), byteorder="little")
    tilt_range = tuple(sorted((float(args.tilt_range[0]), float(args.tilt_range[1]))))
    print(f"Tilting mode: {args.tilting_mode}")
    print(f"Tilt range: [{tilt_range[0]}, {tilt_range[1]}]")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Base seed: {base_seed}")

    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")

    video_dir = os.path.join(args.output_dir, args.xpid)
    if args.record_video:
        print(f"Video output dir: {video_dir}")

    env_names = ["Nocturne-CtrlSim-v0"]
    evaluator = CtrlSimEvaluator(
        env_names=env_names,
        num_processes=args.num_processes,
        num_episodes=args.num_episodes,
        device=args.device,
        seed=base_seed,
        scenario_index_path=args.scenario_index_path,
        opponent_checkpoint=args.checkpoint_path,
        scenario_data_dir=args.scenario_data_dir,
        preprocess_dir=args.preprocess_dir,
        vehicle_map_path=args.vehicle_map_path,
        max_episode_steps=args.num_steps,
        opponent_k=7,
        tilting_mode=args.tilting_mode,
        tilt_range=tilt_range,
        show_level_log=args.show_level_log,
        record_video=args.record_video,
        show_vehicle_ids=args.show_vehicle_ids,
        output_dir=args.output_dir,
        xpid=args.xpid,
        video_dir=video_dir,
        inference_precision=args.inference_precision,
        action_repeat_interval=args.action_repeat_interval,
        sparse_inference_action_repeat=args.sparse_inference_action_repeat,
    )

    from batch_inference import ExternalTeacher

    external_teacher = ExternalTeacher(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        base_seed=base_seed,
        inference_precision=args.inference_precision,
    )

    action_space = evaluator.venv[env_names[0]].action_space
    if evaluator.is_discrete_actions:
        action_dim = 1
    else:
        action_dim = action_space.shape[0]
    agent = DummyEvalAgent(action_dim=action_dim)

    episode_metrics = evaluate_with_metrics(
        evaluator,
        agent,
        deterministic=args.deterministic,
        show_progress=args.verbose,
        render=args.render,
        tilting_mode=args.tilting_mode,
        progress_threshold=args.progress_threshold,
        external_teacher=external_teacher,
    )
    csv_path = write_metrics_csv(args.output_dir, args.xpid, episode_metrics)
    print(f"Metrics saved to: {csv_path}")

    evaluator.close()


if __name__ == "__main__":
    main()
