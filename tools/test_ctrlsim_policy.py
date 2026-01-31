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

from adapters.ctrl_sim import CtrlSimOpponentAdapter
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
        opponent_tilting_mode: str,
        show_level_log: bool,
        record_video: bool,
        output_dir: str,
        xpid: str,
        device: str = "cuda",
        seed: int = 0,
        **_kwargs,
    ):
        if opponent_tilting_mode == "per_level":
            tilting_mode = "per_vehicle"
        elif opponent_tilting_mode == "global":
            tilting_mode = "global"
        else:
            # none: still use global mode, but tilting will be zeroed after reset
            tilting_mode = "global"

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
        )
        self.opponent_tilting_mode = opponent_tilting_mode
        self.device = device
        self.checkpoint_path = opponent_checkpoint
        self.show_level_log = show_level_log
        self.record_video = record_video
        self.output_dir = output_dir
        self.xpid = xpid
        self.episode_idx = 0
        self.goal_pos_tolerance = 1.0
        self._opponent_reached_goal_ids = set()
        self._opponent_stop_modes = {}
        self._logged_initial = False
        self._logged_ego_reached = False
        self.debug_csv_path = os.path.join(self.output_dir, self.xpid, "goal_stop_debug.csv")

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
        self.ego_adapter.reset(
            self.env.scenario,
            self.env.vehicles,
            self.env._gt_data_dict,
            self.env._preproc_data,
            [self.ego_id] if self.ego_id is not None else [],
        )

    def _maybe_disable_opponent_tilting(self):
        # none: force opponent tilting to zero
        if self.opponent_tilting_mode == "none":
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
        self.env.start_recording(out_dir, str(self.episode_idx))

    def _stop_recording(self):
        if not self.record_video:
            return
        self.env.stop_recording(str(self.episode_idx))
        self.episode_idx += 1

    def _reset_episode_state(self):
        self._opponent_reached_goal_ids = set()
        self._opponent_stop_modes = {}
        self._logged_initial = False
        self._logged_ego_reached = False

    def _get_goal_pos(self, veh_id):
        goal_points = getattr(self.env, "_goal_points_by_id", None)
        if goal_points and veh_id in goal_points:
            return goal_points[veh_id]
        if veh_id == self.ego_id and getattr(self.env, "_ego_goal_dict", None) is not None:
            return self.env._ego_goal_dict.get("pos")
        return None

    def _get_goal_pos_with_source(self, veh_id):
        goal_points = getattr(self.env, "_goal_points_by_id", None)
        if goal_points and veh_id in goal_points:
            return goal_points[veh_id], "goal_points_by_id"
        if veh_id == self.ego_id and getattr(self.env, "_ego_goal_dict", None) is not None:
            return self.env._ego_goal_dict.get("pos"), "ego_goal_dict"
        return None, "none"

    def _distance_to_goal(self, veh, goal_pos):
        if veh is None or goal_pos is None:
            return None
        pos = veh.getPosition()
        return float(np.linalg.norm(goal_pos - np.array([pos.x, pos.y])))

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

    def _log_debug_event(
        self,
        event,
        veh_id,
        step,
        has_goal_point,
        goal_source,
        dist_to_goal,
        stop_mode="",
    ):
        out_dir = os.path.join(self.output_dir, self.xpid)
        os.makedirs(out_dir, exist_ok=True)
        fields = [
            "episode",
            "step",
            "veh_id",
            "event",
            "has_goal_point",
            "goal_source",
            "dist_to_goal",
            "stop_mode",
        ]
        write_header = not os.path.exists(self.debug_csv_path)
        with open(self.debug_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "episode": self.episode_idx,
                    "step": int(step),
                    "veh_id": int(veh_id) if veh_id is not None else "",
                    "event": event,
                    "has_goal_point": bool(has_goal_point),
                    "goal_source": goal_source,
                    "dist_to_goal": "" if dist_to_goal is None else float(dist_to_goal),
                    "stop_mode": stop_mode,
                }
            )

    def _log_initial_states(self):
        if self._logged_initial:
            return
        self._logged_initial = True
        if self.env.ego_vehicle is not None and self.ego_id is not None:
            goal_pos, goal_source = self._get_goal_pos_with_source(self.ego_id)
            dist = self._distance_to_goal(self.env.ego_vehicle, goal_pos)
            self._log_debug_event(
                "ego_initial",
                self.ego_id,
                self.env.current_step,
                goal_pos is not None,
                goal_source,
                dist,
            )
        for veh in self.env.opponent_vehicles:
            if veh is None:
                continue
            veh_id = veh.getID()
            goal_pos, goal_source = self._get_goal_pos_with_source(veh_id)
            dist = self._distance_to_goal(veh, goal_pos)
            self._log_debug_event(
                "opponent_initial",
                veh_id,
                self.env.current_step,
                goal_pos is not None,
                goal_source,
                dist,
            )

    def _update_opponent_stop_states(self):
        for veh in self.env.opponent_vehicles:
            if veh is None:
                continue
            veh_id = veh.getID()
            goal_pos, goal_source = self._get_goal_pos_with_source(veh_id)
            if goal_pos is None:
                continue
            if veh_id in self._opponent_reached_goal_ids:
                self._stop_vehicle(veh)
                continue
            if self._is_within_goal(veh, goal_pos):
                self._opponent_reached_goal_ids.add(veh_id)
                stop_mode = self._stop_vehicle(veh)
                self._opponent_stop_modes[veh_id] = stop_mode
                dist = self._distance_to_goal(veh, goal_pos)
                self._log_debug_event(
                    "opponent_reached_goal",
                    veh_id,
                    self.env.current_step,
                    True,
                    goal_source,
                    dist,
                    stop_mode=stop_mode,
                )

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
        self._log_initial_states()
        return obs

    def reset(self):
        obs = self.env.reset()
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        self._log_level()
        self._start_recording()
        self._log_initial_states()
        return obs

    def reset_agent(self):
        obs = self.env.reset_agent()
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        self._log_initial_states()
        return obs

    def step(self, _action):
        t = self.env.current_step
        ego_actions = self.ego_adapter.step(t, self.env.vehicles)
        if self.ego_id is not None and self.ego_id in ego_actions:
            accel, steer = ego_actions[self.ego_id]
        else:
            accel, steer = 0.0, 0.0

        action = np.array(
            [
                np.clip(accel / 10.0, -1.0, 1.0),
                np.clip(steer / 0.7, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs, reward, done, info = self.env.step(action)
        self.ego_adapter.record_all_actions(
            self.env.current_step - 1,
            self.env.vehicles,
            {self.ego_id: (accel, steer)} if self.ego_id is not None else {},
        )
        self._update_opponent_stop_states()
        if self._ego_reached_goal():
            self.env._goal_reached = True
            if hasattr(self.env, "_episode_goal_reached"):
                self.env._episode_goal_reached = True
            if not self._logged_ego_reached:
                goal_pos, goal_source = self._get_goal_pos_with_source(self.ego_id)
                dist = self._distance_to_goal(self.env.ego_vehicle, goal_pos)
                self._log_debug_event(
                    "ego_reached_goal",
                    self.ego_id,
                    self.env.current_step,
                    goal_pos is not None,
                    goal_source,
                    dist,
                )
                self._logged_ego_reached = True
            if not done:
                done = True
                if hasattr(self.env, "opponent"):
                    self.env.opponent.finalize(self.env.vehicles)
                if hasattr(self.env, "_get_info"):
                    info = self.env._get_info()
                else:
                    info = dict(info)
                    info.setdefault("goal_reached", True)
                    info.setdefault(
                        "episode",
                        {"r": reward, "l": self.env.current_step},
                    )
        if done:
            self._stop_recording()
        return obs, reward, done, info

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

        for env_name in env_names:
            make_fn = [
                lambda: self._make_env(
                    env_name,
                    record_video,
                    device=device,
                    **kwargs,
                )
            ] * self.num_processes
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=90)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--opponent_tilting_mode",
        type=str,
        choices=["per_level", "global", "none"],
        default="per_level",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show_level_log", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    return parser.parse_args()


def _build_tilting_columns(info, opponent_tilting_mode):
    opp_count = 7
    tilts = []

    if opponent_tilting_mode == "per_level":
        for i in range(opp_count):
            g = info.get(f"per_vehicle_goal_tilt_{i}", 0.0)
            v = info.get(f"per_vehicle_veh_tilt_{i}", 0.0)
            e = info.get(f"per_vehicle_edge_tilt_{i}", 0.0)
            tilts.append((float(g), float(v), float(e)))
    elif opponent_tilting_mode == "global":
        g = float(info.get("goal_tilt", 0.0))
        v = float(info.get("veh_veh_tilt", 0.0))
        e = float(info.get("veh_edge_tilt", 0.0))
        for _ in range(opp_count):
            tilts.append((g, v, e))
    else:
        for _ in range(opp_count):
            tilts.append((0.0, 0.0, 0.0))

    difficulty = 0.0
    valid = 0
    for g, v, e in tilts:
        if g == 0 and v == 0 and e == 0:
            continue
        valid += 1
        difficulty += (g + v + e) / 3.0
    if valid > 0:
        difficulty /= valid

    columns = {}
    for i, (g, v, e) in enumerate(tilts):
        columns[f"opp{i}_goal_tilt"] = g
        columns[f"opp{i}_veh_veh_tilt"] = v
        columns[f"opp{i}_veh_edge_tilt"] = e
    columns["difficulty"] = float(difficulty)
    return columns


def _extract_episode_metrics(info, episode_return, solved_threshold, opponent_tilting_mode):
    if "episode" in info:
        episode_return = info["episode"].get("r", episode_return)
    solved = 1.0 if episode_return > solved_threshold else 0.0
    collision = info.get("collision_occurred", info.get("collision", 0.0))
    goal_reached = info.get("goal_reached_occurred", info.get("goal_reached", 0.0))
    offroad = info.get("offroad_occurred", info.get("offroad", 0.0))
    progress = info.get("avg_progress", info.get("progress", 0.0))

    metrics = {
        "test_returns": float(episode_return),
        "solved_rate": float(solved),
        "collision": float(collision),
        "goal_reached": float(goal_reached),
        "offroad": float(offroad),
        "progress": float(progress),
    }
    metrics.update(_build_tilting_columns(info, opponent_tilting_mode))
    return metrics


def evaluate_with_metrics(
    evaluator, agent, deterministic, show_progress, render, opponent_tilting_mode
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

        if env_name.startswith("Nocturne"):
            obs, reward, done, infos = venv.step_env(action, reset_random=True)
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
                    info, info["episode"]["r"], solved_threshold, opponent_tilting_mode
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
    fields.append("difficulty")
    return fields


def write_metrics_csv(output_dir, xpid, episode_metrics):
    out_dir = os.path.join(output_dir, xpid)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    fields = [
        "episode",
        "test_returns",
        "solved_rate",
        "collision",
        "goal_reached",
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
            for field in fields:
                if field == "episode":
                    continue
                avg[field] = float(np.mean([m[field] for m in episode_metrics]))
            writer.writerow(avg)

    return csv_path


def main() -> None:
    args = parse_args()
    print(f"Opponent tilting mode: {args.opponent_tilting_mode}")
    print(f"Checkpoint: {args.checkpoint_path}")

    if args.record_video and args.num_processes != 1:
        raise ValueError("--record_video requires --num_processes=1")

    video_dir = os.path.join(args.output_dir, args.xpid)
    if args.record_video:
        print(f"Video output dir: {video_dir}")

    env_names = ["Nocturne-CtrlSim-Adversarial-v0"]
    evaluator = CtrlSimEvaluator(
        env_names=env_names,
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
        opponent_tilting_mode=args.opponent_tilting_mode,
        show_level_log=args.show_level_log,
        record_video=args.record_video,
        output_dir=args.output_dir,
        xpid=args.xpid,
        video_dir=video_dir,
    )

    action_dim = evaluator.venv[env_names[0]].action_space.shape[0]
    agent = DummyEvalAgent(action_dim=action_dim)

    episode_metrics = evaluate_with_metrics(
        evaluator,
        agent,
        deterministic=args.deterministic,
        show_progress=args.verbose,
        render=args.render,
        opponent_tilting_mode=args.opponent_tilting_mode,
    )
    csv_path = write_metrics_csv(args.output_dir, args.xpid, episode_metrics)
    print(f"Metrics saved to: {csv_path}")

    evaluator.close()

    print("\nExample:")
    print(
        "python tools/test_ctrlsim_policy.py \\\n  --scenario_index_path <path> \\\n  --scenario_data_dir <path> \\\n  --preprocess_dir <path> \\\n  --checkpoint_path <path> \\\n  --opponent_tilting_mode per_level \\\n  --output_dir <dir> --xpid <xpid> \\\n  --record_video"
    )


if __name__ == "__main__":
    main()
