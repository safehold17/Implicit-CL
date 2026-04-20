"""Bootstrap helpers for constructing the Nocturne-CtrlSim env."""

from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from typing import Any

import gym
import numpy as np

from .config import NocturneCtrlSimEnvConfig
from ..student.observation_action import (
    StudentObservationConfig,
    get_student_obs_dim,
)
from ..student.student_reward import reset_student_component_applied_return
from ..utils.tilt_helpers import init_level_params_vec


@dataclass(frozen=True)
class NocturneCtrlSimBootstrapDependencies:
    """Runtime dependencies needed to bootstrap the environment."""

    scenario_index_cls: Any
    create_minimal_config: Any
    data_bridge_cls: Any
    opponent_cls: Any
    runtime_cls: Any
    load_scenario: Any
    get_vehicle_by_id: Any
    load_vehicle_ids_for_scenario: Any
    initialize_ego_goal_state: Any
    get_goal_point_for_vehicle: Any
    get_student_observation: Any


def _init_config_state(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
) -> None:
    """Initialize direct config-backed env attributes."""
    env.use_ego_ctrlsim_kl_loss = config.use_ego_ctrlsim_kl_loss
    env.use_enhanced_regret = config.use_enhanced_regret
    env.use_policy_reweighting = config.use_policy_reweighting
    env.policy_reweighting_target = config.policy_reweighting_target
    env.policy_reweighting_config = config.policy_reweighting_config
    env.action_repeat_frequency = config.action_repeat_frequency
    env.kl_loss_computation_frequency = config.kl_loss_computation_frequency
    env.fixed_environment = config.fixed_environment
    env.solvable_progress_threshold = config.solvable_progress_threshold
    env.current_ego_reweight_tilt = (0, 0, 0)
    env._set_process_seed(config.seed)


def _init_scenario_pool(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
    deps: NocturneCtrlSimBootstrapDependencies,
) -> None:
    """Initialize scenario index state and dynamic pool bookkeeping."""
    env.scenario_index_path = config.scenario_index_path
    env.scenario_index = deps.scenario_index_cls(config.scenario_index_path)
    env.scenario_ids = list(env.scenario_index.scenario_ids)
    env.scenario_id_to_index = dict(env.scenario_index.scenario_id_to_index)
    env.index_to_scenario_id = dict(env.scenario_index.index_to_scenario_id)
    env.dynamic_scenario_pool = config.dynamic_scenario_pool
    env.max_scenario_pool_size = config.max_scenario_pool_size
    env._scenario_pool_dirty = False


def _resolve_vehicle_map_path(config: NocturneCtrlSimEnvConfig) -> str | None:
    """Resolve vehicle-map path relative to the project root when needed."""
    if not config.vehicle_map_path:
        return None
    if os.path.isabs(config.vehicle_map_path):
        return config.vehicle_map_path
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return os.path.join(project_root, config.vehicle_map_path)


def _init_data_bridge_and_opponent(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
    deps: NocturneCtrlSimBootstrapDependencies,
) -> None:
    """Initialize model/config loading, data bridge, and opponent adapter."""
    cfg = config.cfg
    if cfg is None:
        cfg = deps.create_minimal_config(
            checkpoint_path=config.opponent_checkpoint,
            scenario_dir=config.scenario_data_dir,
            preprocess_dir=config.preprocess_dir,
        )

    env.cfg = cfg
    env.opponent_checkpoint = config.opponent_checkpoint
    env.scenario_data_dir = config.scenario_data_dir
    env.preprocess_dir = config.preprocess_dir
    env.preproc_cache_size = config.preproc_cache_size
    env.inference_precision = config.inference_precision
    env.opponent_runtime_mode = config.opponent_runtime_mode
    env.vehicle_map_path = _resolve_vehicle_map_path(config)
    env._vehicle_map_cache = None

    env.data_bridge = deps.data_bridge_cls(
        cfg,
        config.preprocess_dir,
        preproc_cache_size=config.preproc_cache_size,
    )
    env.opponent = deps.opponent_cls(
        cfg=cfg,
        checkpoint_path=config.opponent_checkpoint,
        device=config.device,
        action_repeat_frequency=config.action_repeat_frequency,
        kl_loss_computation_frequency=config.kl_loss_computation_frequency,
        sparse_inference_action_repeat=config.sparse_inference_action_repeat,
        use_ego_ctrlsim_kl_loss=config.use_ego_ctrlsim_kl_loss,
        use_enhanced_regret=config.use_enhanced_regret,
        use_policy_reweighting=config.use_policy_reweighting,
        policy_reweighting_reward_scale=env.policy_reweighting_config.reward_scale,
        policy_reweighting_epsilon=env.policy_reweighting_config.epsilon,
        policy_reweighting_target=config.policy_reweighting_target,
        load_on_init=(config.opponent_runtime_mode == "normal"),
    )
    env.opponent._ego_action_scale = 1.0
    env.opponent._ego_reweight_tilt = tuple(env.current_ego_reweight_tilt)


def _init_environment_attributes(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
) -> None:
    """Initialize environment config fields and mutable state."""
    max_episode_steps = config.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = env.cfg.nocturne.steps
    if max_episode_steps != env.cfg.nocturne.steps:
        warnings.warn(
            f"max_episode_steps ({max_episode_steps}) != cfg.nocturne.steps "
            f"({env.cfg.nocturne.steps}); using the passed max_episode_steps for termination."
        )

    env.max_episode_steps = max_episode_steps
    env.done_on_position_reached_only = config.done_on_position_reached_only
    env.early_termination = config.early_termination
    env.goal_pos_tolerance = config.goal_pos_tolerance
    env.device = config.device
    env.opponent_k = config.opponent_k
    env.per_vehicle_tilting_length = 3 * env.opponent_k
    env.dt = env.cfg.nocturne.dt

    env.tilting_mode = config.tilting_mode
    env.mutation_mode = config.mutation_mode
    env.mutation_range = config.mutation_range
    env.tilt_range = config.tilt_range
    env.show_tilting_params = config.show_tilting_params
    env.show_vehicle_ids = config.show_vehicle_ids
    env.show_ego_vehicle_selection = config.show_ego_vehicle_selection
    env.remove_background_vehicles = config.remove_background_vehicles
    env.removed_vehicle_ids = []

    env.current_level = None
    env.current_step = 0
    env.adversary_step_count = 0
    env._set_level_seed(config.seed)

    env.sim = None
    env.scenario = None
    env.vehicles = []
    env._vehicle_by_id_cache = {}
    env._single_env_teacher = None
    env.ego_vehicle = None
    env.opponent_vehicles = []
    env.opponent_vehicle_ids = []
    env.current_opponent_vehicle_num = 0
    env.ego_selection_mode = "unknown"

    env._gt_data_dict = {}
    env._gt_traj_cache = {}
    env._gt_action_target_cache = {}
    env._gt_action_runtime_cache = {}
    env._preproc_data = None
    env._veh_id_to_preproc_idx = {}

    env._ego_goal_dict = None
    env._ego_goal_dist_normalizer = 1.0
    env._ego_vehicle_data_dict = {}
    env._goal_points_by_id = {}

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
    reset_student_component_applied_return(env)
    env._student_component_applied_return_before_inference_step = np.zeros(
        3,
        dtype=np.float32,
    )
    env._last_completed_complexity_info = None

    env.level_params_vec = init_level_params_vec(
        tilting_mode=config.tilting_mode,
        per_vehicle_tilting_length=env.per_vehicle_tilting_length,
    )

    env._max_observable_agents = config.student_num_neighbors
    env._top_k_road_points = config.student_top_k_road
    env.veh_veh_collision_rew_multiplier = config.veh_veh_collision_rew_multiplier
    env.veh_edge_collision_rew_multiplier = config.veh_edge_collision_rew_multiplier
    env.pos_target_achieved_rew_multiplier = (
        config.pos_target_achieved_rew_multiplier
    )
    env.use_persistent_position_reward = config.use_persistent_position_reward
    env.use_pos_shaped = config.use_pos_shaped
    env.use_approaching_goal = config.use_approaching_goal
    env.use_speed_shaped = config.use_speed_shaped
    env.use_heading_shaped = config.use_heading_shaped
    env.use_speed_heading_target = config.use_speed_heading_target
    env.shaped_goal_reward = config.shaped_goal_reward
    env.shaped_goal_distance_scaling = config.shaped_goal_distance_scaling
    env.approaching_goal_scaling = config.approaching_goal_scaling
    env.use_veh_veh_shaped = config.use_veh_veh_shaped
    env.use_veh_edge_shaped = config.use_veh_edge_shaped
    env.max_veh_veh_distance = config.max_veh_veh_distance
    env.veh_edge_reward_distance_clip = config.veh_edge_reward_distance_clip
    env.reset_random_max_retries = config.reset_random_max_retries


def _init_observation_and_action_spaces(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
) -> None:
    """Initialize student/adversary spaces and encoding metadata."""
    env._road_graph_cache = None
    env._road_graph_np = None
    env._student_road_edge_polylines = ()
    env._student_vehicle_cache = None
    env.student_observation_config = StudentObservationConfig(
        max_neighbors=int(env._max_observable_agents),
        top_k_road_points=int(env._top_k_road_points),
    )
    env._obs_dim = get_student_obs_dim(env.student_observation_config)
    if config.requested_obs_dim is not None and int(config.requested_obs_dim) != env._obs_dim:
        warnings.warn(
            f"obs_dim ({config.requested_obs_dim}) is ignored for Nocturne student observations; "
            f"using centralized observation dim {env._obs_dim}."
        )

    env.student_accel_discretization = config.student_accel_discretization
    env.student_steer_discretization = config.student_steer_discretization
    env.student_num_actions = (
        env.student_accel_discretization * env.student_steer_discretization
    )
    env.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env._obs_dim,),
        dtype=np.float32,
    )
    env.action_space = gym.spaces.Discrete(env.student_num_actions)

    if env.tilting_mode == "none":
        env.adversary_action_dim = 1
    elif env.tilting_mode == "global":
        env.adversary_action_dim = 4
    elif env.tilting_mode == "per_vehicle":
        env.adversary_action_dim = 1 + env.per_vehicle_tilting_length
    else:
        raise ValueError(f"Unsupported tilting_mode: {env.tilting_mode}")

    env.adversary_max_steps = 1
    env.random_z_dim = config.random_z_dim
    env.passable = True
    env.adversary_action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(env.adversary_action_dim,),
        dtype=np.float32,
    )
    env.adversary_ts_obs_space = gym.spaces.Box(
        low=0,
        high=env.adversary_max_steps,
        shape=(1,),
        dtype=np.int32,
    )
    env.adversary_randomz_obs_space = gym.spaces.Box(
        low=0,
        high=1.0,
        shape=(config.random_z_dim,),
        dtype=np.float32,
    )
    env.adversary_image_obs_space = gym.spaces.Box(
        low=-25.0,
        high=max(len(env.scenario_ids), 25),
        shape=(len(env.level_params_vec),),
        dtype=np.float32,
    )
    env.adversary_observation_space = gym.spaces.Dict(
        {
            "image": env.adversary_image_obs_space,
            "time_step": env.adversary_ts_obs_space,
            "random_z": env.adversary_randomz_obs_space,
        }
    )

    n_u_chars = max(12, len(str(np.iinfo(np.int32).max)))
    env.encoding_u_chars = np.dtype(("U", n_u_chars))


def _init_support_services(
    env: Any,
    deps: NocturneCtrlSimBootstrapDependencies,
) -> None:
    """Initialize late-bound helpers, metrics, recorder, and runtime."""
    env._load_scenario_impl = deps.load_scenario
    env._get_vehicle_by_id_impl = deps.get_vehicle_by_id
    env._load_vehicle_ids_for_scenario_impl = deps.load_vehicle_ids_for_scenario
    env._initialize_ego_goal_state_impl = deps.initialize_ego_goal_state
    env._get_goal_point_for_vehicle_impl = deps.get_goal_point_for_vehicle
    env._get_student_observation_impl = deps.get_student_observation

    env.reset_metrics()
    env.video_recorder = None
    env.recording_video = False
    env.runtime = deps.runtime_cls(env)


def bootstrap_nocturne_ctrlsim_env(
    env: Any,
    config: NocturneCtrlSimEnvConfig,
    deps: NocturneCtrlSimBootstrapDependencies,
) -> None:
    """Populate a fresh env instance from validated config and dependencies."""
    _init_config_state(env, config)
    _init_scenario_pool(env, config, deps)
    _init_data_bridge_and_opponent(env, config, deps)
    _init_environment_attributes(env, config)
    _init_observation_and_action_spaces(env, config)
    _init_support_services(env, deps)
