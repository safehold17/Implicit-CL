"""Runtime helpers shared by CtrlSim evaluation scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from batch_inference import build_external_teacher_kwargs
from ctrlsim_adapter.opponent_vehicle import CtrlSimOpponentAdapter
from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
from envs.wrappers import ParallelAdversarialVecEnv
from evaluation.eval import Evaluator
from util import is_discrete_actions


class CtrlSimEgoWrapper:
    """VecEnv-friendly wrapper that drives ego with CtrlSimOpponentAdapter."""

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
        action_repeat_frequency: int = 2,
        kl_loss_computation_frequency: int = 2,
        sparse_inference_action_repeat: bool = False,
        student_accel_discretization: int = 20,
        student_steer_discretization: int = 50,
        collect_ego_ctrlsim_rtg: bool = False,
        opponent_runtime_mode: str = "normal",
        teacher_control_mode: str = "split",
        ego_tilt_override: tuple[int, int, int] | None = None,
        **_kwargs,
    ):
        """Initialize one evaluation wrapper for split or joint teacher control."""
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
            action_repeat_frequency=action_repeat_frequency,
            kl_loss_computation_frequency=kl_loss_computation_frequency,
            sparse_inference_action_repeat=sparse_inference_action_repeat,
            student_accel_discretization=student_accel_discretization,
            student_steer_discretization=student_steer_discretization,
            opponent_runtime_mode=opponent_runtime_mode,
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
        self.collect_ego_ctrlsim_rtg = bool(collect_ego_ctrlsim_rtg)
        self.teacher_control_mode = str(teacher_control_mode)
        if self.teacher_control_mode not in {"split", "joint"}:
            raise ValueError(
                "teacher_control_mode must be one of ['joint', 'split'], "
                f"got {teacher_control_mode!r}."
            )
        self.ego_tilt_override = (
            None
            if ego_tilt_override is None
            else tuple(int(v) for v in ego_tilt_override)
        )

        self.ego_adapter = CtrlSimOpponentAdapter(
            cfg=self.env.cfg,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            action_repeat_frequency=action_repeat_frequency,
            kl_loss_computation_frequency=kl_loss_computation_frequency,
            sparse_inference_action_repeat=sparse_inference_action_repeat,
            use_enhanced_regret=self.collect_ego_ctrlsim_rtg,
        )
        self.ego_adapter.set_tilting(0, 0, 0)
        self.ego_id = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = getattr(self.env, "spec", None)

    def _build_joint_teacher_tilt_mapping(self) -> dict[int, tuple[int, int, int]]:
        """Build one per-vehicle tilt mapping for joint-teacher control."""
        mapping = {}
        opponent_tilt_by_id = dict(getattr(self.env.opponent, "per_vehicle_tilting", {}) or {})
        current_tilt = getattr(self.env.opponent, "current_tilt", None)
        default_opponent_tilt = (
            0,
            0,
            0,
        ) if current_tilt is None else (
            int(current_tilt.goal_tilt),
            int(current_tilt.veh_veh_tilt),
            int(current_tilt.veh_edge_tilt),
        )

        for veh_id in getattr(self.env, "opponent_vehicle_ids", []):
            mapping[int(veh_id)] = tuple(
                int(v) for v in opponent_tilt_by_id.get(int(veh_id), default_opponent_tilt)
            )

        if self.ego_id is not None:
            ego_tilt = getattr(self, "ego_tilt_override", None)
            if ego_tilt is None:
                ego_tilt = (0, 0, 0)
            mapping[int(self.ego_id)] = tuple(int(v) for v in ego_tilt)
        return mapping

    def _reset_ego_adapter(self) -> None:
        """Reset the active teacher adapter state after environment reset."""
        self.ego_id = self.env.ego_vehicle.getID() if self.env.ego_vehicle else None
        teacher_control_mode = getattr(self, "teacher_control_mode", "split")
        if teacher_control_mode == "joint":
            joint_controlled_ids = []
            if self.ego_id is not None:
                joint_controlled_ids.append(self.ego_id)
            joint_controlled_ids.extend(
                veh_id
                for veh_id in self.env.opponent_vehicle_ids
                if veh_id != self.ego_id
            )
            self.env.opponent.set_per_vehicle_tilting(
                self._build_joint_teacher_tilt_mapping()
            )
            self.env.opponent._veh_id_to_preproc_idx = dict(
                getattr(self.env, "_veh_id_to_preproc_idx", {})
            )
            self.env.opponent.reset(
                self.env.scenario,
                self.env.vehicles,
                self.env._gt_data_dict,
                self.env._preproc_data,
                joint_controlled_ids,
                ego_id=self.ego_id,
                require_policy=True,
            )
            return
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

    def _maybe_disable_opponent_tilting(self) -> None:
        """Force opponent tilting to zero in none mode."""
        if self.tilting_mode == "none":
            self.env.opponent.set_tilting(0, 0, 0)

    def _log_level(self) -> None:
        """Print current level metadata when level logging is enabled."""
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

    def _start_recording(self) -> None:
        """Start episode video recording when requested."""
        if not self.record_video:
            return
        out_dir = os.path.join(self.output_dir, self.xpid)
        os.makedirs(out_dir, exist_ok=True)
        self.env.start_recording(
            out_dir,
            str(self.episode_idx),
            show_vehicle_ids=self.show_vehicle_ids,
        )

    def _stop_recording(self) -> None:
        """Stop episode video recording when requested."""
        if not self.record_video:
            return
        self.env.stop_recording(str(self.episode_idx))
        self.episode_idx += 1

    def _reset_episode_state(self) -> None:
        """Reset per-episode goal reach bookkeeping."""
        self._opponent_reached_goal_ids = set()
        self._episode_position_reached = False

    def _finish_reset(self, obs, *, log_level: bool, start_recording: bool):
        """Apply the shared post-reset bookkeeping for all reset paths."""
        self._reset_episode_state()
        self._maybe_disable_opponent_tilting()
        self._reset_ego_adapter()
        if log_level:
            self._log_level()
        if start_recording:
            self._start_recording()
        return obs

    def _get_goal_pos(self, veh_id):
        """Return the goal position for one vehicle when available."""
        goal_points = getattr(self.env, "_goal_points_by_id", None)
        if goal_points and veh_id in goal_points:
            return goal_points[veh_id]
        if veh_id == self.ego_id and getattr(self.env, "_ego_goal_dict", None) is not None:
            return self.env._ego_goal_dict.get("pos")
        return None

    def _is_within_goal(self, veh, goal_pos, *, tolerance: float | None = None) -> bool:
        """Check whether one vehicle is within the goal tolerance."""
        if veh is None or goal_pos is None:
            return False
        pos = veh.getPosition()
        dist = np.linalg.norm(goal_pos - np.array([pos.x, pos.y]))
        goal_tolerance = (
            self.goal_pos_tolerance if tolerance is None else float(tolerance)
        )
        return bool(dist < goal_tolerance)

    def _stop_vehicle(self, veh) -> str:
        """Stop one vehicle using the first supported API."""
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

    def _update_opponent_stop_states(self) -> None:
        """Freeze opponents once they reach their goals."""
        goal_tolerance = float(getattr(self.env, "goal_pos_tolerance", 2.0))
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
            if self._is_within_goal(veh, goal_pos, tolerance=goal_tolerance):
                self._opponent_reached_goal_ids.add(veh_id)
                self._stop_vehicle(veh)

    def _ego_reached_goal(self) -> bool:
        """Return whether ego is within the goal tolerance."""
        if bool(getattr(self.env, "_position_reached", False)):
            return True
        if self.ego_id is None or self.env.ego_vehicle is None:
            return False
        goal_pos = self._get_goal_pos(self.ego_id)
        if goal_pos is None:
            return False
        return self._is_within_goal(
            self.env.ego_vehicle,
            goal_pos,
            tolerance=float(getattr(self.env, "goal_pos_tolerance", 2.0)),
        )

    def reset_random(self):
        """Reset the wrapped env by sampling a random level."""
        return self._finish_reset(
            self.env.reset_random(),
            log_level=True,
            start_recording=True,
        )

    def reset(self):
        """Reset the wrapped env with its default reset path."""
        return self._finish_reset(
            self.env.reset(),
            log_level=True,
            start_recording=True,
        )

    def reset_to_level(self, level):
        """Reset the wrapped env to an explicit scenario level."""
        return self._finish_reset(
            self.env.reset_to_level(level),
            log_level=True,
            start_recording=True,
        )

    def reset_agent(self):
        """Reset agent state while keeping the current level."""
        return self._finish_reset(
            self.env.reset_agent(),
            log_level=False,
            start_recording=False,
        )

    def _postprocess_step(self, obs, reward, done, info):
        """Update post-step episode state and enrich step info."""
        self._update_opponent_stop_states()
        info = dict(info)
        position_reached = bool(info.get("position_reached", 0.0)) or self._ego_reached_goal()
        position_reached_occurred = (
            bool(info.get("position_reached_occurred", 0.0))
            or self._episode_position_reached
            or position_reached
        )
        if position_reached_occurred:
            self._episode_position_reached = True

        if position_reached and not done:
            done = True
            if hasattr(self.env, "opponent"):
                self.env.opponent.finalize(self.env.vehicles)
            if hasattr(self.env, "_get_info"):
                info = self.env._get_info()
            else:
                info.setdefault(
                    "episode",
                    {"r": reward, "l": self.env.current_step},
                )
            info = dict(info)
            position_reached = bool(info.get("position_reached", 0.0)) or position_reached
            position_reached_occurred = (
                bool(info.get("position_reached_occurred", 0.0))
                or self._episode_position_reached
                or position_reached
            )
        if position_reached:
            info["progress"] = 1.0
            info["max_progress"] = 1.0
        info["position_reached"] = float(position_reached)
        info["position_reached_occurred"] = float(position_reached_occurred)
        if done:
            self._stop_recording()
        return obs, reward, done, info

    def _apply_ego_action(self, accel: float, steer: float) -> None:
        """Apply one ego action to the simulator vehicle."""
        ego_veh = self.env.ego_vehicle
        if ego_veh is None:
            return
        if accel > 0:
            ego_veh.acceleration = accel
        else:
            ego_veh.brake(abs(accel))
        ego_veh.steering = steer

    def _step_with_ego_action(self, accel: float, steer: float, opponent_actions):
        """Advance the simulator using one ego action and opponent action map."""
        self.env.current_step += 1
        self.env._last_ego_student_action = (float(accel), float(steer))
        self._apply_ego_action(accel, steer)
        obs, reward, done, info = self.env.runtime.step_post_actions(opponent_actions)
        teacher_control_mode = getattr(self, "teacher_control_mode", "split")
        if teacher_control_mode == "joint":
            applied_actions = dict(opponent_actions)
            if self.ego_id is not None:
                applied_actions[self.ego_id] = (accel, steer)
            self.env.opponent.record_all_actions(
                self.env.current_step - 1,
                self.env.vehicles,
                applied_actions,
            )
        else:
            self.ego_adapter.record_all_actions(
                self.env.current_step - 1,
                self.env.vehicles,
                {self.ego_id: (accel, steer)} if self.ego_id is not None else {},
            )
        return self._postprocess_step(obs, reward, done, info)

    def step(self, _action):
        """Run one single-env inference step through the local teacher."""
        prepared = self.step_prepare(_action)
        teacher = self._get_single_env_teacher()
        ego_outputs = None
        if prepared["ego"] is not None:
            ego_outputs = teacher.run_batched_forward([prepared["ego"]])[0]
        opp_outputs = teacher.run_batched_forward([prepared["opponent"]])[0]
        return self.step_complete({"ego": ego_outputs, "opponent": opp_outputs})

    def _get_single_env_teacher(self):
        """Lazily build the single-env teacher used by wrapper.step()."""
        teacher = self._single_env_teacher
        if teacher is not None:
            return teacher

        from batch_inference import ExternalTeacher

        teacher = ExternalTeacher(
            **build_external_teacher_kwargs(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                inference_precision=self.inference_precision,
                config_source=self,
            )
        )
        self._single_env_teacher = teacher
        return teacher

    def step_prepare(self, _action):
        """Prepare ego/opponent inference payloads for one environment step."""
        t = self.env.current_step
        runtime_mode = getattr(self.env, "opponent_runtime_mode", "normal")
        teacher_control_mode = getattr(self, "teacher_control_mode", "split")
        if teacher_control_mode == "joint":
            joint_prepared = None
            if runtime_mode == "normal":
                joint_prepared = self.env.opponent.prepare_step(
                    t,
                    self.env.vehicles,
                )
            return {
                "ego": None,
                "ego_ctrlsim": None,
                "opponent": joint_prepared,
            }
        ego_pack = self.ego_adapter.prepare_step_pack(
            t,
            self.env.vehicles,
            ego_id=self.ego_id,
            include_ego_ctrlsim_prepared=self.collect_ego_ctrlsim_rtg,
        )
        opponent_prepared = None
        if runtime_mode == "normal" and len(self.env.opponent_vehicle_ids) > 0:
            opponent_prepared = self.env.opponent.prepare_step(
                t,
                self.env.vehicles,
            )
        return {
            "ego": ego_pack.get("opponent_prepared"),
            "ego_ctrlsim": ego_pack.get("ego_ctrlsim_prepared"),
            "opponent": opponent_prepared,
        }

    def step_complete(self, model_outputs):
        """Apply teacher outputs and advance the wrapped environment."""
        teacher_control_mode = getattr(self, "teacher_control_mode", "split")
        if teacher_control_mode == "joint":
            joint_actions = self.env.opponent.apply_predictions(
                model_outputs.get("opponent")
            )
            if self.ego_id in joint_actions:
                accel, steer = joint_actions[self.ego_id]
            else:
                history_steps = int(getattr(self.env.opponent, "history_steps", 0))
                current_step = int(getattr(self.env, "current_step", 0))
                if self.ego_id is not None and current_step >= history_steps - 1:
                    raise ValueError(
                        f"Missing ego CtrlSim action for ego_id={self.ego_id} "
                        f"at step={current_step}."
                    )
                accel, steer = 0.0, 0.0
            opponent_actions = {
                veh_id: action
                for veh_id, action in joint_actions.items()
                if veh_id != self.ego_id
            }
            return self._step_with_ego_action(accel, steer, opponent_actions)

        ego_actions = self.ego_adapter.apply_predictions(model_outputs.get("ego"))
        if self.ego_id in ego_actions:
            accel, steer = ego_actions[self.ego_id]
        else:
            history_steps = int(getattr(self.ego_adapter, "history_steps", 0))
            current_step = int(getattr(self.env, "current_step", 0))
            if self.ego_id is not None and current_step >= history_steps - 1:
                raise ValueError(
                    f"Missing ego CtrlSim action for ego_id={self.ego_id} "
                    f"at step={current_step}."
                )
            accel, steer = 0.0, 0.0

        runtime_mode = getattr(self.env, "opponent_runtime_mode", "normal")
        opponent_actions = {}
        if runtime_mode == "normal" and len(self.env.opponent_vehicle_ids) > 0:
            opponent_actions = self.env.opponent.apply_predictions(
                model_outputs.get("opponent")
            )
        return self._step_with_ego_action(accel, steer, opponent_actions)

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()


class CtrlSimEvaluator(Evaluator):
    """Evaluator specialized for CtrlSim ego / Nocturne opponent rollout."""

    @staticmethod
    def _make_env(env_name, record_video: bool = False, **kwargs):
        """Build one wrapped evaluation env."""
        _ = env_name
        kwargs["record_video"] = record_video
        return CtrlSimEgoWrapper(**kwargs)

    def _init_parallel_envs(
        self,
        env_names,
        num_processes,
        device=None,
        record_video: bool = False,
        **kwargs,
    ):
        """Initialize parallel evaluation envs for all requested names."""
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


def _get_action_dim_from_venv(venv) -> int:
    """Return the external action width expected by one wrapped venv."""
    action_space = venv.action_space
    if action_space.__class__.__name__ == "Box":
        return int(action_space.shape[0])
    return 1


def build_zero_action_batch(venv, num_envs: int) -> np.ndarray:
    """Build a no-op action batch for wrappers that ignore external actions."""
    return np.zeros((num_envs, _get_action_dim_from_venv(venv)), dtype=np.float32)


def _attach_ego_ctrlsim_rtg_to_infos(
    infos,
    ego_ctrlsim_rtgs,
    ego_ctrlsim_rtg_metadata,
):
    """Attach optional ego RTG side-channel outputs to step infos."""
    enriched_infos = []
    for info, rtg, metadata in zip(infos, ego_ctrlsim_rtgs, ego_ctrlsim_rtg_metadata):
        enriched_info = dict(info)
        enriched_info["ego_ctrlsim_pred_rtg"] = rtg
        enriched_info["ego_ctrlsim_pred_rtg_metadata"] = metadata
        enriched_info["ego_ctrlsim_pred_rtg_step"] = (
            None if metadata is None else int(metadata["step_t"])
        )
        enriched_infos.append(enriched_info)
    return enriched_infos


def _split_ctrlsim_eval_prepared_batch(per_env_prepared):
    """Split named CtrlSim evaluator payload streams."""
    ego_prepared = [
        item.get("ego") if item else None
        for item in per_env_prepared
    ]
    ego_ctrlsim_prepared = [
        item.get("ego_ctrlsim") if item else None
        for item in per_env_prepared
    ]
    opponent_prepared = [
        item.get("opponent") if item else None
        for item in per_env_prepared
    ]
    return ego_prepared, ego_ctrlsim_prepared, opponent_prepared


def run_batched_ctrlsim_step(
    *,
    venv,
    action,
    external_teacher,
    reset_random: bool = False,
    auto_reset_on_done: bool = True,
    collect_ego_ctrlsim_rtg: bool = False,
):
    """Run one batched CtrlSim step with optional ego RTG side-channel."""
    per_env_prepared = venv.step_prepare(action)
    (
        ego_prepared,
        ego_ctrlsim_prepared,
        opponent_prepared,
    ) = _split_ctrlsim_eval_prepared_batch(per_env_prepared)

    ego_results = external_teacher.run_batched_forward(ego_prepared)
    ego_ctrlsim_rtgs = [None] * len(per_env_prepared)
    ego_ctrlsim_rtg_metadata = [None] * len(per_env_prepared)
    if collect_ego_ctrlsim_rtg:
        (
            opponent_results,
            _,
            ego_ctrlsim_rtgs,
            ego_ctrlsim_rtg_metadata,
        ) = external_teacher.run_batched_forward_with_ego_logits(
            opponent_prepared,
            ego_ctrlsim_prepared,
        )
    else:
        opponent_results = external_teacher.run_batched_forward(opponent_prepared)

    combined_outputs = [
        {"ego": ego_result, "opponent": opponent_result}
        for ego_result, opponent_result in zip(ego_results, opponent_results)
    ]
    obs, reward, done, infos = venv.step_complete(
        combined_outputs,
        reset_random=reset_random,
        auto_reset_on_done=auto_reset_on_done,
    )
    if collect_ego_ctrlsim_rtg:
        infos = _attach_ego_ctrlsim_rtg_to_infos(
            infos,
            ego_ctrlsim_rtgs,
            ego_ctrlsim_rtg_metadata,
        )
    return obs, reward, done, infos


def build_ctrlsim_evaluator(
    args: Any,
    *,
    base_seed: int,
    num_processes: int,
    tilt_range: tuple[float, float],
    collect_ego_ctrlsim_rtg: bool,
) -> CtrlSimEvaluator:
    """Construct a CtrlSim evaluator from one CLI-style args object."""
    return CtrlSimEvaluator(
        env_names=["Nocturne-CtrlSim-v0"],
        num_processes=num_processes,
        num_episodes=getattr(args, "num_episodes", num_processes),
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
        inference_precision=args.inference_precision,
        action_repeat_frequency=args.action_repeat_frequency,
        kl_loss_computation_frequency=args.kl_loss_computation_frequency,
        sparse_inference_action_repeat=args.sparse_inference_action_repeat,
        student_accel_discretization=args.student_accel_discretization,
        student_steer_discretization=args.student_steer_discretization,
        collect_ego_ctrlsim_rtg=collect_ego_ctrlsim_rtg,
    )


def build_ctrlsim_external_teacher(args: Any, *, base_seed: int):
    """Construct the batched ExternalTeacher shared by CtrlSim evaluation scripts."""
    from batch_inference import ExternalTeacher

    teacher_kwargs = build_external_teacher_kwargs(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        inference_precision=args.inference_precision,
        config_source=args,
    )
    teacher_kwargs["base_seed"] = base_seed
    return ExternalTeacher(**teacher_kwargs)
