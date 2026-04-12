"""Runtime controller for Nocturne-CtrlSim episodes.

This module owns the episode lifecycle after the environment has already been
bootstrapped. It is responsible for:

1. Constructing episode-scoped runtime state for the selected scenario.
2. Splitting an env step into prepare/complete/post phases for batched teacher
   inference.
3. Applying opponent, ego, and GT fallback actions in the correct order.
4. Refreshing observation/reward-facing caches after simulator state changes.
5. Folding simulator state into the `(obs, reward, done, info)` API expected
   by the training loop.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .level import normalize_per_vehicle_tilting
from ..utils.tilt_helpers import sample_policy_reweighting_ego_tilt
from ..services.ground_truth import (
    build_episode_gt_action_cache,
    get_gt_action,
    is_ego_position_reached,
)
from ..services.scenario_runtime import remove_background_moving_vehicles
from ..services.simulation_info import (
    check_done,
    compute_current_progress,
    get_info,
    update_episode_progress,
)
from ..student.observation_action import (
    apply_student_action,
    refresh_student_vehicle_cache,
)
from ..student.student_reward import (
    get_student_component_applied_return,
    compute_student_reward,
    reset_student_component_applied_return,
)
from ctrlsim_adapter._data_bridge.scenario import road_data_to_numpy


def split_prepared_pack_batch(
    prepared_batch: list[Optional[Dict[str, Any]]],
) -> tuple[list[Optional[Dict[str, Any]]], list[Optional[Dict[str, Any]]]]:
    """Split packed prepared payloads into opponent and ego-ctrlsim streams.

    The batched inference path supports both legacy payloads and the newer
    packed payload format:

    - Legacy payload: a single prepared item only for opponent inference.
    - Packed payload: a dict carrying `opponent_prepared` and optionally
      `ego_ctrlsim_prepared`.

    This helper normalizes both layouts into two aligned lists so downstream
    batching code can process the two streams independently without losing
    episode ordering.
    """
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
    """Own episode setup and per-step execution for one env instance."""

    def __init__(self, env: Any) -> None:
        """Bind the runtime controller to its parent environment."""
        self.env = env

    def initialize_simulation(self) -> None:
        """Initialize all runtime state for a newly selected scenario.

        This runs once per reset after the level/scenario has been chosen. It
        rebuilds simulator-facing state, episode bookkeeping, opponent control
        metadata, GT caches, and the student-side caches reused across the
        episode.
        """
        env = self.env
        if env.current_level is None:
            return

        level = env.current_level
        env.current_step = 0
        env.reset_metrics()

        # Per-step outcome flags are updated after every simulator transition.
        env._collision_occurred = False
        env._goal_reached = False
        env._offroad_occurred = False
        env._position_reached = False

        # Episode-aggregated flags and counters accumulate over the whole
        # rollout and are later reported through `info`.
        env._episode_collision_occurred = False
        env._episode_goal_reached = False
        env._episode_offroad_occurred = False
        env._episode_position_reached = False
        env._episode_steps = 0
        env._episode_progress = 0.0
        reset_student_component_applied_return(env)
        env._student_component_applied_return_before_inference_step = (
            get_student_component_applied_return(env)
        )

        # Keep the episode boundary aligned with ``level.seed`` for downstream
        # logic that still implicitly depends on NumPy's global RNG state.
        np.random.seed(level.seed)

        # Load GT and materialize any episode-scoped GT-derived caches before
        # the simulator starts stepping.
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

        # Materialize the actual Nocturne scenario and populate `env.vehicles`.
        env._load_scenario_impl(env, level.scenario_id)

        # Build fast lookup tables immediately after vehicles are available.
        # These are reused throughout the runtime hot path.
        env._veh_id_to_vehicle = {veh.getID(): veh for veh in env.vehicles}
        env._veh_id_to_preproc_idx = {
            veh.getID(): idx for idx, veh in enumerate(env.vehicles)
        }

        # Resolve which agents are controlled by ego/opponent for this
        # scenario.
        (
            ego_id,
            opponent_ids,
            ego_selection_mode,
            opponent_vehicle_num,
        ) = env._load_vehicle_ids_for_scenario_impl(env, level.scenario_id)
        env.ego_vehicle = env._veh_id_to_vehicle.get(ego_id)
        if env.ego_vehicle is None:
            raise ValueError(
                f"ego_vehicle_id {ego_id} from vehicle map does not exist in scenario "
                f"'{level.scenario_id}'."
            )
        env.ego_selection_mode = ego_selection_mode
        env.current_opponent_vehicle_num = int(opponent_vehicle_num)

        # Load CtrlSim preprocessing artifacts that must stay aligned with the
        # selected scenario.
        env._preproc_data, file_exists = env.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        if not file_exists:
            raise FileNotFoundError(
                f"Preprocessed data not found for scenario '{level.scenario_id}'. "
                f"Check preprocess_dir: {env.data_bridge.preprocess_dir}"
            )

        runtime_mode = getattr(env, "opponent_runtime_mode", "normal")
        env.current_ego_reweight_tilt = sample_policy_reweighting_ego_tilt(
            use_policy_reweighting=env.use_policy_reweighting,
            opponent_runtime_mode=runtime_mode,
            tilt_range=env.tilt_range,
            rng=env.np_random,
        )
        if runtime_mode == "disable":
            # Disable mode keeps the simulator alive but hands no non-ego agent
            # to the opponent policy.
            env.opponent_vehicle_ids = []
            env.opponent_vehicles = []
        else:
            env.opponent_vehicle_ids = opponent_ids
            env.opponent_vehicles = []
            missing_opponent_ids = []
            # Resolve direct vehicle references once so later code does not pay
            # repeated ID lookup cost in hot paths.
            for veh_id in opponent_ids:
                veh = env._veh_id_to_vehicle.get(veh_id)
                if veh is None:
                    missing_opponent_ids.append(veh_id)
                    continue
                env.opponent_vehicles.append(veh)
            if missing_opponent_ids:
                raise ValueError(
                    f"opponent_vehicle_ids {missing_opponent_ids} from vehicle map do not exist in scenario "
                    f"'{level.scenario_id}'."
                )

        # Initialize ego goal first, then collect all available goal points into
        # a single mapping shared by fixups and visualization.
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
                # Zero any unused tail slots so downstream readers do not see
                # stale per-vehicle tilt values from a previous scenario with
                # more controlled opponents.
                for idx in range(cutoff, len(per)):
                    per[idx] = 0
                env.current_level.per_vehicle_tilting = tuple(per)
                if len(env.level_params_vec) >= 4 + env.per_vehicle_tilting_length:
                    for idx in range(env.per_vehicle_tilting_length):
                        env.level_params_vec[4 + idx] = per[idx]

        if runtime_mode == "normal":
            # Forward the selected tilt configuration to the opponent adapter.
            # The exact shape depends on whether the env uses a single global
            # tilt or an independent triple per opponent vehicle.
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
            # Non-normal runtime modes never let the opponent adapter inject
            # reward tilt into simulator behavior.
            env.opponent.set_tilting(0, 0, 0)
            env.opponent.per_vehicle_tilting = None

        if env.remove_background_vehicles:
            remove_background_moving_vehicles(env)

        # Reset the opponent adapter against the fresh episode state so its
        # internal history buffers stay aligned with the current simulator
        # objects and preprocessing artifacts.
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
        env.opponent._ego_reweight_tilt = tuple(env.current_ego_reweight_tilt)
        # Build student-side caches used in the hot path:
        # - compact road graph arrays are valid for the whole episode
        # - road-edge polylines are reused by shaped reward computation
        # - the vehicle snapshot is refreshed per step but seeded here so the
        #   initial observation after reset can read it immediately
        env._road_graph_cache = env.data_bridge.get_road_data(env.scenario)
        # Pre-flatten road geometry once per episode so observation code can
        # consume a compact numpy representation without a second bridge call.
        # ``point_indices`` inside this cache preserve deterministic ordering
        # when multiple road points are equally close to the ego vehicle.
        env._road_graph_np = road_data_to_numpy(env._road_graph_cache)
        env._student_road_edge_polylines = tuple(
            env.data_bridge.extract_road_edge_polylines(env._road_graph_cache)
        )
        refresh_student_vehicle_cache(env)

    def step_prepare(self, action: np.ndarray) -> Dict[str, Optional[Dict]]:
        """Apply ego action locally and prepare model inputs for this step.

        This is the first half of the split-step API used by batched teacher
        inference. It advances the logical step counter, applies the student's
        discrete ego action to the simulator vehicle object, and asks the
        opponent adapter to package the inference payload for the current state.
        """
        env = self.env
        if bool(getattr(env, "use_enhanced_regret", False)):
            env._student_component_applied_return_before_inference_step = (
                get_student_component_applied_return(env)
            )
        env.current_step += 1
        env._last_ego_student_action = apply_student_action(env, action)

        if getattr(env, "opponent", None) is None:
            return {
                "opponent_prepared": None,
                "ego_ctrlsim_prepared": None,
            }

        ego_id = env.ego_vehicle.getID() if getattr(env, "ego_vehicle", None) else None
        # ``current_step - 1`` is the simulator time index of the state that the
        # model is about to act on. The prepare path packages that state before
        # any dynamics update happens in ``step_post_actions``.
        return env.opponent.prepare_step_pack(
            env.current_step - 1,
            env.vehicles,
            ego_id=ego_id,
            include_ego_ctrlsim_prepared=bool(
                getattr(env, "use_ego_ctrlsim_kl_loss", False)
                or getattr(env, "use_policy_reweighting", False)
                or getattr(env, "use_enhanced_regret", False)
            ),
        )

    def step_complete(
        self,
        model_outputs: Optional[Dict],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Turn model outputs into simulator actions and finish the env step."""
        env = self.env
        runtime_mode = getattr(env, "opponent_runtime_mode", "normal")
        if runtime_mode == "normal" and len(env.opponent_vehicle_ids) > 0:
            opponent_actions = env.opponent.apply_predictions(model_outputs)
        else:
            if getattr(env, "opponent", None) is not None:
                env.opponent.reset_ego_action_scale()
            opponent_actions = {}
        return self.step_post_actions(opponent_actions)

    def get_single_env_teacher(self):
        """Lazily create and cache the single-env teacher inference wrapper."""
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
        """Apply actions, step the simulator, and derive env outputs.

        The action application order is:

        1. Opponent model actions for controlled opponent agents.
        2. Student action for ego, which was already written to the vehicle in
           ``step_prepare`` and is only recorded here for history alignment.
        3. GT fallback actions for every remaining uncontrolled vehicle.

        After all actions are accounted for, the simulator advances one step and
        the method refreshes every env-facing signal derived from the new state.
        """
        env = self.env
        veh_id_to_vehicle = env._veh_id_to_vehicle
        # Apply model-produced actions to opponent-controlled agents using the
        # episode-scoped vehicle lookup table built during initialization.
        for veh_id, action in opponent_actions.items():
            veh = veh_id_to_vehicle.get(veh_id)
            if veh is not None:
                env.opponent.apply_action(veh, action)

        ego_id = env.ego_vehicle.getID() if env.ego_vehicle else None
        controlled_ids = set(opponent_actions.keys())
        applied_actions_for_history = dict(opponent_actions)
        if ego_id is not None:
            # The ego action has already been written onto the vehicle object in
            # ``step_prepare``. Here we only inject it into the action history
            # so downstream bookkeeping stays aligned with the simulator step.
            controlled_ids.add(ego_id)
            ego_action = getattr(env, "_last_ego_student_action", None)
            if ego_action is not None:
                applied_actions_for_history[ego_id] = ego_action

        # Any vehicle not explicitly controlled this step falls back to GT so
        # the rest of the scene continues evolving consistently.
        for veh in env.vehicles:
            veh_id = veh.getID()
            if veh_id in controlled_ids:
                continue
            gt_action = get_gt_action(env, veh_id, env.current_step - 1, veh)
            if gt_action is not None:
                env.opponent.apply_action(veh, gt_action)
                applied_actions_for_history[veh_id] = gt_action

        # Record the complete action set that led into this simulator advance.
        env.opponent.record_all_actions(
            env.current_step - 1,
            env.vehicles,
            applied_actions_for_history,
        )

        # Give the adapter a chance to save pre-step validity state, then
        # advance simulator dynamics exactly once for this env step.
        if hasattr(env.opponent, "cache_last_valid_positions"):
            env.opponent.cache_last_valid_positions(env.vehicles)
        env.sim.step(env.dt)
        # Some adapters perform post-step clamps or goal-based position fixes
        # after dynamics integration.
        if hasattr(env.opponent, "post_step_fix_opponent_positions"):
            env.opponent.post_step_fix_opponent_positions(
                env.vehicles,
                env._goal_points_by_id,
                env.current_step,
            )
        # Observation and reward code consume a per-step vehicle snapshot rather
        # than rescanning simulator objects independently.
        refresh_student_vehicle_cache(env)

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

        # Build the next observation and scalar reward from the post-step state.
        obs = env._get_student_observation_impl(env)
        reward = compute_student_reward(env)

        # Update episode-level summary flags after reward logic has refreshed
        # the current-step outcome fields.
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

        # Progress is tracked as a monotonic episode aggregate rather than a raw
        # per-step measurement.
        current_progress = compute_current_progress(env)
        env._episode_progress = update_episode_progress(
            previous_progress=env._episode_progress,
            current_progress=current_progress,
            position_reached=env._position_reached,
        )

        # Finalize the adapter once the env declares termination so any buffered
        # histories are flushed before the next reset.
        done = check_done(env)
        if done:
            env.opponent.finalize(env.vehicles)
        info = get_info(env)
        return obs, reward, done, info
