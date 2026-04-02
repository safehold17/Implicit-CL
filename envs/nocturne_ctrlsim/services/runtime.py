"""Runtime controller for Nocturne-CtrlSim episodes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..level import normalize_per_vehicle_tilting
from .gt_helpers import (
    build_episode_gt_action_cache,
    get_gt_action,
    is_ego_position_reached,
)
from .scenario_helpers import remove_background_moving_vehicles
from .simulation_info import (
    check_done,
    compute_current_progress,
    get_info,
    update_episode_progress,
)
from ..student_env_policy import apply_student_action
from ..student_reward import compute_student_reward


def split_prepared_pack_batch(
    prepared_batch: list[Optional[Dict[str, Any]]],
) -> tuple[list[Optional[Dict[str, Any]]], list[Optional[Dict[str, Any]]]]:
    """拆分 batch prepared pack 为 opponent/ego_ctrlsim 两路。 / Split a batch prepared pack into opponent and ego_ctrlsim streams."""
    opponent_prepared: list[Optional[Dict[str, Any]]] = []
    ego_ctrlsim_prepared: list[Optional[Dict[str, Any]]] = []
    for item in prepared_batch:
        if item is None:
            opponent_prepared.append(None)
            ego_ctrlsim_prepared.append(None)
            continue
        if "opponent_prepared" in item or "ego_ctrlsim_prepared" in item:
            opponent_prepared.append(item.get("opponent_prepared"))
            ego_ctrlsim_prepared.append(item.get("ego_ctrlsim_prepared"))
            continue
        opponent_prepared.append(item)
        ego_ctrlsim_prepared.append(None)
    return opponent_prepared, ego_ctrlsim_prepared


class NocturneCtrlSimRuntime:
    """Owns episode runtime setup and per-step execution."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def initialize_simulation(self) -> None:
        env = self.env
        if env.current_level is None:
            return

        level = env.current_level
        env.current_step = 0
        env.reset_metrics()

        env._collision_occurred = False
        env._goal_reached = False
        env._offroad_occurred = False
        env._position_reached = False

        env._episode_collision_occurred = False
        env._episode_goal_reached = False
        env._episode_offroad_occurred = False
        env._episode_position_reached = False
        env._episode_steps = 0
        env._episode_progress = 0.0

        np.random.seed(level.seed)

        env._gt_data_dict = env.data_bridge.get_ground_truth(
            env.scenario_data_dir,
            f"{level.scenario_id}.json",
        )
        env._gt_traj_cache = {
            veh_id: np.asarray(data["traj"])
            for veh_id, data in env._gt_data_dict.items()
            if isinstance(data, dict) and "traj" in data
        }
        build_episode_gt_action_cache(env)

        env._load_scenario_impl(env, level.scenario_id)

        env._veh_id_to_preproc_idx = {
            veh.getID(): idx for idx, veh in enumerate(env.vehicles)
        }

        (
            ego_id,
            opponent_ids,
            ego_selection_mode,
            opponent_vehicle_num,
        ) = env._load_vehicle_ids_for_scenario_impl(env, level.scenario_id)
        env.ego_vehicle = env._get_vehicle_by_id_impl(env, ego_id)
        if env.ego_vehicle is None:
            raise ValueError(
                f"ego_vehicle_id {ego_id} from vehicle map does not exist in scenario "
                f"'{level.scenario_id}'."
            )
        env.ego_selection_mode = ego_selection_mode
        env.current_opponent_vehicle_num = int(opponent_vehicle_num)

        env._preproc_data, file_exists = env.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        if not file_exists:
            raise FileNotFoundError(
                f"Preprocessed data not found for scenario '{level.scenario_id}'. "
                f"Check preprocess_dir: {env.data_bridge.preprocess_dir}"
            )

        runtime_mode = getattr(env, "opponent_runtime_mode", "normal")
        if runtime_mode == "disable":
            env.opponent_vehicle_ids = []
            env.opponent_vehicles = []
        else:
            env.opponent_vehicle_ids = opponent_ids
            env.opponent_vehicles = []
            missing_opponent_ids = []
            for veh_id in opponent_ids:
                veh = env._get_vehicle_by_id_impl(env, veh_id)
                if veh is None:
                    missing_opponent_ids.append(veh_id)
                    continue
                env.opponent_vehicles.append(veh)
            if missing_opponent_ids:
                raise ValueError(
                    f"opponent_vehicle_ids {missing_opponent_ids} from vehicle map do not exist in scenario "
                    f"'{level.scenario_id}'."
                )

        env._initialize_ego_goal_state_impl(env)
        env._goal_points_by_id = {}
        if env.ego_vehicle is not None and env._ego_goal_dict is not None:
            env._goal_points_by_id[env.ego_vehicle.getID()] = env._ego_goal_dict["pos"]
        for veh_id in env.opponent_vehicle_ids:
            goal_pos = env._get_goal_point_for_vehicle_impl(env, veh_id)
            if goal_pos is not None:
                env._goal_points_by_id[veh_id] = goal_pos

        if env.tilting_mode == "per_vehicle" and env.current_level is not None:
            actual_n = len(env.opponent_vehicle_ids)
            per = list(
                normalize_per_vehicle_tilting(
                    env.current_level.per_vehicle_tilting,
                    env.per_vehicle_tilting_length,
                )
            )
            cutoff = actual_n * 3
            if cutoff < len(per):
                for idx in range(cutoff, len(per)):
                    per[idx] = 0
                env.current_level.per_vehicle_tilting = tuple(per)
                if len(env.level_params_vec) >= 4 + env.per_vehicle_tilting_length:
                    for idx in range(env.per_vehicle_tilting_length):
                        env.level_params_vec[4 + idx] = per[idx]

        if runtime_mode == "normal":
            if env.tilting_mode == "global":
                env.opponent.set_tilting(
                    level.goal_tilt,
                    level.veh_veh_tilt,
                    level.veh_edge_tilt,
                )
            elif env.tilting_mode == "per_vehicle":
                sorted_opponent_ids = sorted(env.opponent_vehicle_ids)
                per_vehicle_mapping = {}
                per = level.per_vehicle_tilting
                for idx, veh_id in enumerate(sorted_opponent_ids):
                    base = 3 * idx
                    if base + 2 < len(per):
                        per_vehicle_mapping[veh_id] = (
                            per[base],
                            per[base + 1],
                            per[base + 2],
                        )
                    else:
                        per_vehicle_mapping[veh_id] = (0, 0, 0)
                env.opponent.set_per_vehicle_tilting(per_vehicle_mapping)
            else:
                env.opponent.set_tilting(0, 0, 0)
        else:
            env.opponent.set_tilting(0, 0, 0)
            env.opponent.per_vehicle_tilting = None

        if env.remove_background_vehicles:
            remove_background_moving_vehicles(env)

        vehicles_to_control = (
            env.opponent_vehicle_ids if runtime_mode == "normal" else []
        )
        env.opponent._veh_id_to_preproc_idx = dict(env._veh_id_to_preproc_idx)
        env.opponent.reset(
            env.scenario,
            env.vehicles,
            env._gt_data_dict,
            env._preproc_data,
            vehicles_to_control,
            ego_id=env.ego_vehicle.getID() if env.ego_vehicle else None,
        )
        env._road_graph_cache = env.data_bridge.get_road_data(env.scenario)

    def step_prepare(self, action: np.ndarray) -> Dict[str, Optional[Dict]]:
        env = self.env
        env.current_step += 1
        env._last_ego_student_action = apply_student_action(env, action)

        if getattr(env, "opponent", None) is None:
            return {
                "opponent_prepared": None,
                "ego_ctrlsim_prepared": None,
            }

        ego_id = env.ego_vehicle.getID() if getattr(env, "ego_vehicle", None) else None
        return env.opponent.prepare_step_pack(
            env.current_step - 1,
            env.vehicles,
            ego_id=ego_id,
            include_ego_ctrlsim_prepared=bool(
                getattr(env, "use_ego_ctrlsim_kl_loss", False)
                or getattr(env, "opponent_policy_reweighting_enabled", False)
            ),
        )

    def step_complete(
        self,
        model_outputs: Optional[Dict],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        env = self.env
        runtime_mode = getattr(env, "opponent_runtime_mode", "normal")
        if runtime_mode == "normal" and len(env.opponent_vehicle_ids) > 0:
            opponent_actions = env.opponent.apply_predictions(model_outputs)
        else:
            opponent_actions = {}
        return self.step_post_actions(opponent_actions)

    def get_single_env_teacher(self):
        env = self.env
        teacher = env._single_env_teacher
        if teacher is not None:
            return teacher

        from batch_inference import ExternalTeacher, build_external_teacher_kwargs

        teacher = ExternalTeacher(
            **build_external_teacher_kwargs(
                checkpoint_path=env.opponent_checkpoint,
                device=env.device,
                inference_precision=env.inference_precision,
                config_source=env,
            )
        )
        teacher.validate_student_action_space(
            student_accel_discretization=env.student_accel_discretization,
            student_steer_discretization=env.student_steer_discretization,
        )
        env._single_env_teacher = teacher
        return teacher

    def step_post_actions(
        self,
        opponent_actions: Dict[int, Tuple[float, float]],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        env = self.env
        for veh_id, action in opponent_actions.items():
            veh = env._get_vehicle_by_id_impl(env, veh_id)
            if veh is not None:
                env.opponent.apply_action(veh, action)

        ego_id = env.ego_vehicle.getID() if env.ego_vehicle else None
        controlled_ids = set(opponent_actions.keys())
        applied_actions_for_history = dict(opponent_actions)
        if ego_id is not None:
            controlled_ids.add(ego_id)
            ego_action = getattr(env, "_last_ego_student_action", None)
            if ego_action is not None:
                applied_actions_for_history[ego_id] = ego_action

        for veh in env.vehicles:
            veh_id = veh.getID()
            if veh_id in controlled_ids:
                continue
            gt_action = get_gt_action(env, veh_id, env.current_step - 1, veh)
            if gt_action is not None:
                env.opponent.apply_action(veh, gt_action)
                applied_actions_for_history[veh_id] = gt_action

        env.opponent.record_all_actions(
            env.current_step - 1,
            env.vehicles,
            applied_actions_for_history,
        )

        if hasattr(env.opponent, "cache_last_valid_positions"):
            env.opponent.cache_last_valid_positions(env.vehicles)
        env.sim.step(env.dt)
        if hasattr(env.opponent, "post_step_fix_opponent_positions"):
            env.opponent.post_step_fix_opponent_positions(
                env.vehicles,
                env._goal_points_by_id,
                env.current_step,
            )

        if env.recording_video and env.video_recorder is not None:
            env.video_recorder.capture_frame(
                env.scenario,
                env.vehicles,
                roads_data=env._road_graph_cache,
                highlight_vehicle_ids=[env.ego_vehicle.getID()]
                if env.ego_vehicle
                else None,
                opponent_vehicle_ids=env.opponent_vehicle_ids,
                goal_points_by_id=env._goal_points_by_id,
                scenario_id=(
                    getattr(env.current_level, "scenario_id", None)
                    if env.current_level
                    else None
                ),
                show_vehicle_ids=getattr(env, "recording_show_vehicle_ids", False),
            )

        obs = env._get_student_observation_impl(env)
        reward = compute_student_reward(env)

        env._episode_steps += 1
        if env._collision_occurred:
            env._episode_collision_occurred = True
        if env._goal_reached:
            env._episode_goal_reached = True
        if env._offroad_occurred:
            env._episode_offroad_occurred = True
        env._position_reached = is_ego_position_reached(env)
        if env._position_reached:
            env._episode_position_reached = True

        current_progress = compute_current_progress(env)
        env._episode_progress = update_episode_progress(
            previous_progress=env._episode_progress,
            current_progress=current_progress,
            position_reached=env._position_reached,
        )

        done = check_done(env)
        if done:
            env.opponent.finalize(env.vehicles)
        info = get_info(env)
        return obs, reward, done, info
