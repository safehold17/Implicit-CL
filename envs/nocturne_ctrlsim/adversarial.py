"""
Nocturne + CtRL-Sim environment

Support DCD framework requirements:
- PLR (Prioritized Level Replay) mechanism
- PAIRED / ACCEL etc. UED algorithms (through Adversary interface)
- dynamic scenario pool size
- Level mutation and editing
"""
import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from .level import ScenarioLevel

from .scenario_helpers import (
    get_vehicle_by_id,
    load_scenario,
    remove_background_moving_vehicles,
)

from .gt_helpers import (
    get_goal_point_for_vehicle,
    initialize_ego_goal_state,
)
from .student_env_policy import (
    build_student_observation_config,
    get_student_obs_dim,
    get_student_observation,
)
from .runtime import NocturneCtrlSimRuntime

from .simulation_info import (
    get_complexity_info,
    reset_metrics as sim_reset_metrics,
)

from .utils.vehicle_map_helpers import load_vehicle_ids_for_scenario

from .utils.video_recorder import NocturneVideoRecorder

from .utils import visualization as viz

from tools.build_scenario_index import ScenarioIndex

from ctrlsim_adapter.config_loader import create_minimal_config
from ctrlsim_adapter.data_bridge import DataBridge
from ctrlsim_adapter.opponent_vehicle import CtrlSimOpponentAdapter
class NocturneCtrlSimAdversarial(gym.Env):
    """
    DCD adversarial environment: Nocturne scenario + CtRL-Sim opponent
    
    Supports two usage modes:
    
    1. **PAIRED/ACCEL mode** (environment adversary building):
       - call reset() to initialize adversary building process
       - call step_adversary() to build level step by step
       - call reset_agent() to let student start training after building
    
    2. **DR/PLR mode** (direct sampling):
       - call reset_random() to randomly generate level
       - or call reset_to_level() to load specified level
    
    Adversary action space (single-step joint action):
    - none: [scenario_idx]
    - global/ego: [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt]
    - per_vehicle: [scenario_idx, per_vehicle_tilt_0, ..., per_vehicle_tilt_N-1]
    """
    
    def __init__(
        self,
        scenario_index_path: str,
        opponent_checkpoint: str,
        scenario_data_dir: str,
        preprocess_dir: str,
        **kwargs
    ):
        """
        Args:
            scenario_index_path: Scenario index JSON file path
            opponent_checkpoint
            scenario_data_dir: Nocturne scenario data directory
            preprocess_dir: ctrl-sim preprocessed data directory
            kwargs: optional runtime/environment settings.
        """
        super().__init__()

        vehicle_map_path = kwargs.get('vehicle_map_path', "data/vehicle_map_valid.json")
        preproc_cache_size = int(kwargs.get('preproc_cache_size', 64))
        opponent_k = kwargs.get('opponent_k', 7)
        max_episode_steps = kwargs.get('max_episode_steps', 90)
        device = kwargs.get('device', 'cuda')
        cfg = kwargs.get('cfg', None)
        seed = kwargs.get('seed', 0)
        fixed_environment = kwargs.get('fixed_environment', False)
        random_z_dim = kwargs.get('random_z_dim', 50)
        dynamic_scenario_pool = kwargs.get('dynamic_scenario_pool', False)
        max_scenario_pool_size = kwargs.get('max_scenario_pool_size', 10000)
        tilting_mode = kwargs.get('tilting_mode', 'per_vehicle')
        mutation_mode = kwargs.get('mutation_mode', 'all')
        mutation_range = kwargs.get('mutation_range', 5.0)
        show_tilting_params = kwargs.get('show_tilting_params', True)
        show_vehicle_ids = kwargs.get('show_vehicle_ids', True)
        show_ego_vehicle_selection = kwargs.get('show_ego_vehicle_selection', True)
        remove_background_vehicles = kwargs.get('remove_background_vehicles', True)
        requested_obs_dim = kwargs.get('obs_dim')
        student_accel_discretization = int(kwargs['student_accel_discretization'])
        student_steer_discretization = int(kwargs['student_steer_discretization'])
        if student_accel_discretization < 2:
            raise ValueError(
                "student_accel_discretization must be >= 2, "
                f"got {student_accel_discretization}"
            )
        if student_steer_discretization < 2:
            raise ValueError(
                "student_steer_discretization must be >= 2, "
                f"got {student_steer_discretization}"
            )
        tilt_range = kwargs.get("tilt_range")
        if tilt_range is None:
            tilt_range = (-25.0, 25.0)
        try:
            tilt_low, tilt_high = tilt_range
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tilt_range must be a length-2 iterable of (low, high)"
            ) from exc
        tilt_range = (float(tilt_low), float(tilt_high))
        if tilt_range[0] > tilt_range[1]:
            raise ValueError(
                "tilt_range must satisfy low <= high, "
                f"got {tilt_range}"
            )
        action_repeat_interval = int(
            kwargs.get("action_repeat_interval", 2)
        )
        sparse_inference_action_repeat = bool(
            kwargs.get("sparse_inference_action_repeat", False)
        )
        self.use_ego_ctrlsim_kl_loss = bool(
            kwargs.get("use_ego_ctrlsim_kl_loss", False)
        )

        self.fixed_environment = fixed_environment
        self._set_process_seed(seed)
        
        # ========== Scenario index (support dynamic extension) ==========
        self.scenario_index_path = scenario_index_path
        self.scenario_index = ScenarioIndex(scenario_index_path)
        self.scenario_ids = list(self.scenario_index.scenario_ids)
        self.scenario_id_to_index = dict(self.scenario_index.scenario_id_to_index)
        self.index_to_scenario_id = dict(self.scenario_index.index_to_scenario_id)
        
        # Dynamic scenario pool config
        self.dynamic_scenario_pool = dynamic_scenario_pool
        self.max_scenario_pool_size = max_scenario_pool_size
        self._scenario_pool_dirty = False
        
        # ==========Opponent vehicle (Ctrlsim) config loading ==========
        if cfg is None:
            cfg = create_minimal_config(
                checkpoint_path=opponent_checkpoint,
                scenario_dir=scenario_data_dir,
                preprocess_dir=preprocess_dir,
            )
        self.cfg = cfg
        self.opponent_checkpoint = opponent_checkpoint
        self.scenario_data_dir = scenario_data_dir
        self.preprocess_dir = preprocess_dir
        self.preproc_cache_size = preproc_cache_size
        self.inference_precision = kwargs.get('inference_precision', 'fp32')
        opponent_runtime_mode = kwargs.get('opponent_runtime_mode', 'normal')
        
        # Vehicle map path (for loading pre-computed ego/opponent IDs)
        if vehicle_map_path:
            if os.path.isabs(vehicle_map_path):
                self.vehicle_map_path = vehicle_map_path
            else:
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                self.vehicle_map_path = os.path.join(project_root, vehicle_map_path)
        else:
            self.vehicle_map_path = None
        self._vehicle_map_cache = None
        
        # ========== Data bridge ==========
        self.data_bridge = DataBridge(
            cfg,
            preprocess_dir,
            preproc_cache_size=preproc_cache_size,
        )
        
        # ========== Opponent policy adapter ==========
        self.opponent = CtrlSimOpponentAdapter(
            cfg=cfg,
            checkpoint_path=opponent_checkpoint,
            device=device,
            action_repeat_interval=action_repeat_interval,
            sparse_inference_action_repeat=sparse_inference_action_repeat,
            load_on_init=(opponent_runtime_mode == 'normal'),
        )
        
        # ========== Environment config ==========
        if max_episode_steps is None:
            max_episode_steps = cfg.nocturne.steps
        if max_episode_steps != cfg.nocturne.steps:
            warnings.warn(
                f"max_episode_steps ({max_episode_steps}) != cfg.nocturne.steps "
                f"({cfg.nocturne.steps}); using the passed max_episode_steps for termination."
            )
        self.max_episode_steps = max_episode_steps
        self.done_on_position_reached_only = bool(
            kwargs.get('done_on_position_reached_only', True)
        )
        self.device = device
        self.opponent_k = int(opponent_k)
        if self.opponent_k < 0:
            raise ValueError(f"opponent_k must be non-negative, got {opponent_k}")
        self.per_vehicle_tilting_length = 3 * self.opponent_k
        self.dt = cfg.nocturne.dt
        
        # ========== Tilting config ==========
        if tilting_mode not in ['global', 'per_vehicle', 'ego', 'none']:
            raise ValueError(
                "tilting_mode must be 'global', 'per_vehicle', 'ego', or 'none', "
                f"got {tilting_mode}"
            )
        if mutation_mode not in ['one', 'all']:
            raise ValueError(
                "mutation_mode must be 'one' or 'all', "
                f"got {mutation_mode}"
            )
        self.tilting_mode = tilting_mode
        self.mutation_mode = mutation_mode
        self.mutation_range = mutation_range
        self.tilt_range = tilt_range
        self.show_tilting_params = show_tilting_params
        self.show_vehicle_ids = show_vehicle_ids
        self.show_ego_vehicle_selection = show_ego_vehicle_selection
        self.remove_background_vehicles = remove_background_vehicles
        self.opponent_runtime_mode = opponent_runtime_mode
        self.removed_vehicle_ids: List[int] = []
        
        # ========== State variables ==========
        self.current_level: Optional[ScenarioLevel] = None
        self.current_step = 0
        self.adversary_step_count = 0  # Adversary building steps
        self._set_level_seed(seed)
        
        # Nocturne simulation object
        self.sim = None
        self.scenario = None
        self.vehicles: List = []
        self._vehicle_by_id_cache: Dict = {}
        self._single_env_teacher = None
        self.ego_vehicle = None
        self.opponent_vehicles: List = []
        self.opponent_vehicle_ids: List[int] = []
        self.current_opponent_vehicle_num: int = 0
        self.ego_selection_mode: str = "unknown"
        
        # Ground truth and preprocessed data
        self._gt_data_dict: Dict = {}
        self._gt_traj_cache: Dict = {}
        self._gt_action_target_cache: Dict = {}
        self._gt_action_runtime_cache: Dict = {}
        self._preproc_data: Optional[Dict] = None
        self._veh_id_to_preproc_idx: Dict[int, int] = {}
        
        # Ego vehicle's goal and reward related state (for _compute_reward)
        self._ego_goal_dict: Optional[Dict] = None
        self._ego_goal_dist_normalizer: float = 1.0
        self._ego_vehicle_data_dict: Dict = {}  # Track ego's historical data
        self._goal_points_by_id: Dict[int, np.ndarray] = {}
        
        # Termination condition
        self._collision_occurred: bool = False
        self._goal_reached: bool = False
        self._offroad_occurred: bool = False
        self._position_reached: bool = False
        
        # Episode statistics (for training monitoring)
        self._episode_collision_occurred: bool = False
        self._episode_goal_reached: bool = False
        self._episode_offroad_occurred: bool = False
        self._episode_position_reached: bool = False
        self._episode_steps: int = 0
        self._episode_progress: float = 0.0  # Target progress [0, 1]
        
        # Cache for last completed episode (for get_complexity_info)
        self._last_completed_complexity_info: Optional[Dict] = None
        
        # Level parameters vector (for adversary building)
        # [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilts...]
        self.level_params_vec = self._init_level_params_vec()
        
        # ========== Student observation config (from args) ==========
        # These parameters are used in make_agent, set default values here
        self._max_observable_agents = kwargs.get('student_num_neighbors', 16)
        self._top_k_road_points = kwargs.get('student_top_k_road', 64)
        self.veh_veh_collision_rew_multiplier = kwargs.get('veh_veh_collision_rew_multiplier', 10.0)
        self.veh_edge_collision_rew_multiplier = kwargs.get('veh_edge_collision_rew_multiplier', 10.0)
        self.pos_target_achieved_rew_multiplier = kwargs.get(
            'pos_target_achieved_rew_multiplier', 10.0
        )
        self.use_persistent_position_reward = kwargs.get(
            'use_persistent_position_reward', False
        )
        self.use_pos_shaped = kwargs.get('use_pos_shaped', True)
        self.use_approaching_goal = kwargs.get('use_approaching_goal', True)
        self.use_speed_shaped = kwargs.get('use_speed_shaped', True)
        self.use_heading_shaped = kwargs.get('use_heading_shaped', True)
        self.use_speed_heading_target = kwargs.get(
            'use_speed_heading_target', True
        )
        self.shaped_goal_reward = kwargs.get('shaped_goal_reward', True)
        self.shaped_goal_distance_scaling = kwargs.get(
            'shaped_goal_distance_scaling', 0.2
        )
        self.approaching_goal_scaling = kwargs.get('approaching_goal_scaling', 1.0)
        self.use_veh_veh_shaped = kwargs.get('use_veh_veh_shaped', True)
        self.use_veh_edge_shaped = kwargs.get('use_veh_edge_shaped', True)
        self.max_veh_veh_distance = kwargs.get('max_veh_veh_distance', 15.0)
        self.veh_edge_reward_distance_clip = kwargs.get(
            'veh_edge_reward_distance_clip', 5.0
        )
        self.reset_random_max_retries = int(kwargs.get('reset_random_max_retries', 10))
        
        # Cache road data (filled after _initialize_simulation)
        self._road_graph_cache: Optional[List[Dict]] = None
        self._load_scenario_impl = load_scenario
        self._get_vehicle_by_id_impl = get_vehicle_by_id
        self._load_vehicle_ids_for_scenario_impl = load_vehicle_ids_for_scenario
        self._initialize_ego_goal_state_impl = initialize_ego_goal_state
        self._get_goal_point_for_vehicle_impl = get_goal_point_for_vehicle
        self._get_student_observation_impl = get_student_observation
        
        # ========== Observation and action space (Student) ==========
        self.student_observation_config = build_student_observation_config(
            max_neighbors=self._max_observable_agents,
            top_k_road_points=self._top_k_road_points,
        )
        self._obs_dim = get_student_obs_dim(self.student_observation_config)
        if requested_obs_dim is not None and int(requested_obs_dim) != self._obs_dim:
            warnings.warn(
                f"obs_dim ({requested_obs_dim}) is ignored for Nocturne student observations; "
                f"using centralized observation dim {self._obs_dim}."
            )
        self.student_accel_discretization = student_accel_discretization
        self.student_steer_discretization = student_steer_discretization
        self.student_num_actions = (
            self.student_accel_discretization * self.student_steer_discretization
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._obs_dim,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.student_num_actions)
        
        # ========== Adversary space definition ==========
        # Adversary building steps: scenario_id + tilt parameters (or scenario only in 'none' mode)
        # set action dimension based on the tilting mode

        if self.tilting_mode == 'none':
            self.adversary_action_dim = 1
        elif self.tilting_mode in ('ego', 'global'):
            self.adversary_action_dim = 4
        elif self.tilting_mode == 'per_vehicle':
            self.adversary_action_dim = 1 + self.per_vehicle_tilting_length
        else:
            raise ValueError(f"Unsupported tilting_mode: {self.tilting_mode}")

        self.adversary_max_steps = 1

        self.random_z_dim = random_z_dim
        self.passable = True  # Driving scenario default passable


        # Adversary action space: continuous action [-1, 1]
        self.adversary_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.adversary_action_dim,),
            dtype=np.float32,
        )

        # Adversary observation space
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, 
            high=self.adversary_max_steps, 
            shape=(1,), 
            dtype=np.int32
        )
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, 
            high=1.0, 
            shape=(random_z_dim,), 
            dtype=np.float32
        )
        # image: current level parameters
        self.adversary_image_obs_space = gym.spaces.Box(
            low=-25.0, 
            high=max(len(self.scenario_ids), 25),
            shape=(len(self.level_params_vec),), 
            dtype=np.float32
        )
        self.adversary_observation_space = gym.spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space
        })
        
        # ========== Encoding format ==========
        # Use string array, compatible with BipedalWalker
        n_u_chars = max(12, len(str(np.iinfo(np.int32).max)))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))
        
        # ========== Metrics tracking ==========
        self.reset_metrics()
        
        # ========== Video recording ==========
        self.video_recorder: Optional[NocturneVideoRecorder] = None
        self.recording_video = False
        self.runtime = NocturneCtrlSimRuntime(self)

    def _set_process_seed(self, seed: int, reseed_numpy: bool = True) -> None:
        """Set process-level RNG streams used by this environment instance."""
        seed = int(seed)
        self.seed_value = seed
        random.seed(seed)
        if reseed_numpy:
            np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # Keep mutation randomness process-specific and reproducible.
        self._mutation_random_state = np.random.RandomState(seed)
        # Keep level-seed generation reproducible and process-local.
        self._level_seed_random_state = np.random.RandomState(seed)

    def _sample_level_seed(self) -> int:
        """Sample a level seed from process-local RNG in int32 range."""
        int32_max = np.iinfo(np.int32).max
        return int(self._level_seed_random_state.randint(1, int32_max))

    def _set_level_seed(self, seed: int) -> int:
        """Set the current level seed tracked by the environment."""
        self.level_seed = int(seed)
        return self.level_seed

    def _set_current_level(self, level: ScenarioLevel) -> ScenarioLevel:
        """Set current level and synchronize its seed state."""
        self.current_level = level
        self._set_level_seed(level.seed)
        return self.current_level

    def _coerce_level(self, level: Union[ScenarioLevel, str, np.ndarray]) -> ScenarioLevel:
        """Convert supported level input formats into a ScenarioLevel."""
        if isinstance(level, ScenarioLevel):
            return level
        if isinstance(level, str):
            return ScenarioLevel.from_level_string(level)
        if isinstance(level, np.ndarray):
            if level.dtype.kind == 'U':  # string array unicode
                return self._decode_string_encoding(level)
            return ScenarioLevel.from_encoding(level, self.index_to_scenario_id)
        raise TypeError(f"Unsupported level type for reset/mutation: {type(level)}")

    def _init_level_params_vec(self) -> List[int]:
        """Build default level parameter vector based on current tilting mode."""
        if self.tilting_mode == 'per_vehicle':
            # Vector layout: [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt]
            return [0, 0, 0, 0] + [0] * self.per_vehicle_tilting_length
        if self.tilting_mode == 'none':
            return [0]
        return [0, 0, 0, 0]

    def _normalize_per_vehicle_tilting(self, per_vehicle_tilting: Tuple[int, ...]) -> Tuple[int, ...]:
        """Normalize per-vehicle tilting vector length to match current opponent_k."""
        normalized = [int(round(float(v))) for v in per_vehicle_tilting]
        if len(normalized) < self.per_vehicle_tilting_length:
            normalized.extend([0] * (self.per_vehicle_tilting_length - len(normalized)))
        elif len(normalized) > self.per_vehicle_tilting_length:
            normalized = normalized[:self.per_vehicle_tilting_length]
        return tuple(normalized)

    # ========== Visualization helpers (bound from visualization module) ==========
    render = viz.render
    start_recording = viz.start_recording
    stop_recording = viz.stop_recording
    
    # ========== Basic environment interface ==========
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed of the environment"""
        if seed is not None:
            # Keep mutation RNG stream aligned with per-env seed, while
            # avoiding global numpy reseeding side effects.
            seed = int(seed)
            self._set_process_seed(seed, reseed_numpy=True)
            self._set_level_seed(seed)
        return [self.level_seed]
    
    # ========== Adversary interface (PAIRED/ACCEL) ==========
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment, prepare adversary building process
        
        Returns:
            adversary observation dictionary: {'image', 'time_step', 'random_z'}
        """
        self.adversary_step_count = 0
        
        # Reset level parameters to default values
        self.level_params_vec = self._init_level_params_vec()
        
        # Generate new level seed
        self._set_level_seed(self._sample_level_seed())
        
        return self._build_adversary_obs()

    def _normalize_adversary_action(self, action: Any) -> np.ndarray:
        """Normalize adversary action into a clipped 1D float vector."""
        import torch

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        raw_action = np.asarray(action, dtype=np.float32)
        normalized_action = raw_action.reshape(-1)

        if normalized_action.size != self.adversary_action_dim:
            raise ValueError(
                f"Adversary action size mismatch: expected {self.adversary_action_dim}, "
                f"got {normalized_action.size}, raw_shape={raw_action.shape}"
            )

        return np.clip(normalized_action, -1.0, 1.0)

    def _map_action_to_scenario_idx(self, scenario_action: float) -> int:
        """Map scenario action from [-1, 1] to scenario index range."""
        num_scenarios = len(self.scenario_ids)
        scenario_idx = int((float(scenario_action) + 1.0) / 2.0 * num_scenarios)
        return int(np.clip(scenario_idx, 0, num_scenarios - 1))

    def _map_action_to_tilt(self, tilt_action: float) -> int:
        """Map tilt action from [-1, 1] to configured tilt range."""
        tilt_scale = (self.tilt_range[1] - self.tilt_range[0]) / 2.0
        tilt_value = float(tilt_action) * tilt_scale
        tilt_value = np.clip(tilt_value, self.tilt_range[0], self.tilt_range[1])
        return int(round(float(tilt_value)))
    
    def step_adversary(
        self,
        action: Any,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:

        # step adversary is only necessary in PAIRED algo
        # PLR could use reset_random()

        """
        Action mapping (single-step joint action):
        - none: [scenario_idx]
        - global/ego: [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt]
        - per_vehicle: [scenario_idx, per_vehicle_tilt_0, ..., per_vehicle_tilt_N-1]
        
        Args:
            action: continuous action vector in [-1, 1]
        
        Returns:
            (obs, reward, done, info)
            - obs: adversary observation
            - reward: always 0 (adversary reward calculated after rollout)
            - done: whether the building is completed
            - info: additional information
        """
        action_vec = self._normalize_adversary_action(action)
        self.level_params_vec[0] = self._map_action_to_scenario_idx(action_vec[0])
        runtime_mode = getattr(self, 'opponent_runtime_mode', 'normal')

        if runtime_mode != 'normal':
            if self.tilting_mode in ('global', 'ego'):
                self.level_params_vec[1] = 0
                self.level_params_vec[2] = 0
                self.level_params_vec[3] = 0
            elif self.tilting_mode == 'per_vehicle':
                self.level_params_vec[1] = 0
                self.level_params_vec[2] = 0
                self.level_params_vec[3] = 0
                self.level_params_vec[4:4 + self.per_vehicle_tilting_length] = (
                    [0] * self.per_vehicle_tilting_length
                )
            elif self.tilting_mode == 'none':
                pass
            else:
                raise ValueError(f"Unsupported tilting_mode: {self.tilting_mode}")
        else:
            if self.tilting_mode in ('global', 'ego'):
                self.level_params_vec[1] = self._map_action_to_tilt(action_vec[1])
                self.level_params_vec[2] = self._map_action_to_tilt(action_vec[2])
                self.level_params_vec[3] = self._map_action_to_tilt(action_vec[3])
            elif self.tilting_mode == 'per_vehicle':
                per_vehicle_values = []
                for v in action_vec[1:1 + self.per_vehicle_tilting_length]:
                    per_vehicle_values.append(self._map_action_to_tilt(v))

                self.level_params_vec[4:4 + self.per_vehicle_tilting_length] = per_vehicle_values
            elif self.tilting_mode == 'none':
                pass
            else:
                raise ValueError(f"Unsupported tilting_mode: {self.tilting_mode}")
        
        self.adversary_step_count += 1
        
        # Check if the building is completed
        done = self.adversary_step_count >= self.adversary_max_steps
        
        if done:
            # Building completed, create ScenarioLevel and initialize environment
            self._build_level_from_params()
        
        return self._build_adversary_obs(), 0, done, {}

    def _build_adversary_obs(self) -> Dict[str, np.ndarray]:
        """Build adversary observation dictionary from current environment state."""
        return {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z(),
        }

    def generate_random_z(self) -> np.ndarray:
        """Generate random condition vector (for adversary observation)."""
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def set_opponent_runtime_mode(self, mode: str) -> str:
        valid_modes = {'disable', 'replay', 'normal'}
        if mode not in valid_modes:
            raise ValueError(
                f"opponent_runtime_mode must be one of {sorted(valid_modes)}, got {mode}"
            )
        self.opponent_runtime_mode = mode
        return self.opponent_runtime_mode

    @property
    def processed_action_dim(self) -> int:
        """Processed action dimension (compatible with AdversarialRunner)."""
        return self.adversary_action_dim
    
    def _build_level_from_params(self) -> None:
        """Build ScenarioLevel from level_params_vec and initialize environment"""

        # Check if the scenario pool mapping needs to be rebuilt
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        
        scenario_idx = int(self.level_params_vec[0])
        scenario_id = self._resolve_scenario_id(scenario_idx)
        
        if self.tilting_mode == 'per_vehicle':
            per_vehicle_tilting = tuple(
                int(round(float(v)))
                for v in self.level_params_vec[4:4 + self.per_vehicle_tilting_length]
            )
            per_vehicle_tilting = self._normalize_per_vehicle_tilting(per_vehicle_tilting)
            self.current_level = ScenarioLevel(
                scenario_id=scenario_id,
                seed=self.level_seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=per_vehicle_tilting,
            )
        elif self.tilting_mode == 'none':
            self.current_level = ScenarioLevel(
                scenario_id=scenario_id,
                seed=self.level_seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=(),
            )
        else:
            self.current_level = ScenarioLevel(
                scenario_id=scenario_id,
                seed=self.level_seed,
                goal_tilt=self.level_params_vec[1],
                veh_veh_tilt=self.level_params_vec[2],
                veh_edge_tilt=self.level_params_vec[3],
            )
        
        # Initialize simulation environment (but not return observation, wait for reset_agent to call) 
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Initialize Nocturne simulation (delegated to runtime)."""
        self.runtime.initialize_simulation()
    
    # ========== PLR/DR interface ==========

    @staticmethod
    def _is_retryable_vehicle_id_error(error: Exception) -> bool:
        """Whether reset_random should retry for this strict vehicle-map error."""
        if not isinstance(error, ValueError):
            return False
        msg = str(error)
        return (
            "ego_vehicle_id is missing" in msg
            or (
                "ego_vehicle_id" in msg
                and "does not exist in scenario" in msg
            )
            or (
                "opponent_vehicle_ids" in msg
                and "do not exist in scenario" in msg
            )
        )
    
    def reset_random(self) -> np.ndarray:
        """
        Randomly generate new level and reset
        
        Entry point for DCD Domain Randomization.
        Randomly sample all parameters.
        
        Returns:
            student initial observation
        """
        max_retries = self.reset_random_max_retries
        last_error: Optional[Exception] = None
        last_scenario_id: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            level = self._sample_random_level()
            last_scenario_id = level.scenario_id
            try:
                return self.reset_to_level(level)
            except Exception as e:
                if not self._is_retryable_vehicle_id_error(e):
                    raise
                last_error = e
                if attempt < max_retries:
                    print(
                        f"Warning: reset_random failed for scenario '{last_scenario_id}' "
                        f"({attempt}/{max_retries}) due to strict vehicle-map ID mismatch: {e}. "
                        "Retrying with another random scenario."
                    )

        raise RuntimeError(
            f"reset_random failed after {max_retries} retries due to strict vehicle-map ID errors. "
            f"Last scenario: '{last_scenario_id}'. Last error: {last_error}"
        ) from last_error

    def _sample_random_level(self) -> ScenarioLevel:
        """Randomly generate level"""
        scenario_id = np.random.choice(self.scenario_ids)
        seed = self._sample_level_seed()
        runtime_mode = getattr(self, 'opponent_runtime_mode', 'normal')

        # In disable/replay runtime modes, tilting is not used by policy.
        # Keep sampled levels scenario-only to avoid unnecessary tilt sampling.
        if runtime_mode != 'normal':
            per_vehicle_tilting = (
                (0,) * self.per_vehicle_tilting_length
                if self.tilting_mode == 'per_vehicle'
                else ()
            )
            return ScenarioLevel(
                scenario_id=scenario_id,
                seed=seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=per_vehicle_tilting,
            )

        def _sample_tilt() -> int:
            return round(float(np.random.uniform(*self.tilt_range)))

        if self.tilting_mode in ('global', 'ego'):
            # Global mode: sample 3 global tilts, per-vehicle tilting to 0
            goal_tilt = _sample_tilt()
            veh_veh_tilt = _sample_tilt()
            veh_edge_tilt = _sample_tilt()
            per_vehicle_tilting = (0,) * self.per_vehicle_tilting_length
        elif self.tilting_mode == 'none':
            goal_tilt = 0
            veh_veh_tilt = 0
            veh_edge_tilt = 0
            per_vehicle_tilting = (0,) * self.per_vehicle_tilting_length
        else:  # per_vehicle mode
            # Per-vehicle mode: global tilts to 0, sample per-vehicle tilts
            goal_tilt = 0
            veh_veh_tilt = 0
            veh_edge_tilt = 0
            per_vehicle_tilting = tuple(
                _sample_tilt() for _ in range(self.per_vehicle_tilting_length)
            )

        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=seed,
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
            per_vehicle_tilting=per_vehicle_tilting,
        )
    
    def reset_to_level(self, level: Union[ScenarioLevel, str, np.ndarray]) -> np.ndarray:
        """
        Load specified level
        
        Supports three input formats (compatible with DCD LevelStore):
        1. ScenarioLevel object
        2. String (from to_level_string())
        3. numpy array (from encoding)
        
        Args:
            level: Level object, string or encoding array
        
        Returns:
            student initial observation
        """
        level = self._coerce_level(level)
        runtime_mode = getattr(self, 'opponent_runtime_mode', 'normal')

        # Keep replay/disable behavior robust even when replayed levels
        # already carry non-zero tilting fields.
        if runtime_mode != 'normal':
            per_vehicle_tilting = (
                (0,) * self.per_vehicle_tilting_length
                if self.tilting_mode == 'per_vehicle'
                else ()
            )
            level = ScenarioLevel(
                scenario_id=level.scenario_id,
                seed=level.seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=per_vehicle_tilting,
            )
        
        # Update level_params_vec to keep consistent
        scenario_idx = self.scenario_id_to_index.get(level.scenario_id, 0)
        if self.tilting_mode == 'per_vehicle':
            normalized_per_vehicle_tilting = self._normalize_per_vehicle_tilting(
                level.per_vehicle_tilting
            )
            level = ScenarioLevel(
                scenario_id=level.scenario_id,
                seed=level.seed,
                goal_tilt=level.goal_tilt,
                veh_veh_tilt=level.veh_veh_tilt,
                veh_edge_tilt=level.veh_edge_tilt,
                per_vehicle_tilting=normalized_per_vehicle_tilting,
            )
            self.level_params_vec = [
                scenario_idx,
                0,
                0,
                0,
                *normalized_per_vehicle_tilting,
            ]
        elif self.tilting_mode == 'none':
            level = ScenarioLevel(
                scenario_id=level.scenario_id,
                seed=level.seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=(),
            )
            self.level_params_vec = [
                scenario_idx,
            ]
        else:
            self.level_params_vec = [
                scenario_idx,
                level.goal_tilt,
                level.veh_veh_tilt,
                level.veh_edge_tilt,
            ]
        self._set_current_level(level)
        
        # Initialize simulation
        self._initialize_simulation()
        
        # Return student observation
        return get_student_observation(self)
    
    def reset_agent(self) -> np.ndarray:
        """
        Reset in current level (without changing level configuration)
        
        Used for:
        1. Starting student after adversary building in PAIRED
        2. Multiple evaluations of the same level in PLR
        
        Returns:
            student initial observation
        """
        if self.current_level is None:
            raise ValueError("Must call reset_to_level or complete step_adversary first")
        
        # Reinitialize simulation
        self._initialize_simulation()
        
        return get_student_observation(self)
    
    def mutate_level(
        self,
        level: Optional[Union[ScenarioLevel, str, np.ndarray]] = None,
        num_edits: Optional[int] = None,
    ) -> np.ndarray:
        """
        Mutate level and reset.

        - If ``level`` is None: mutate current loaded level.
        - If ``level`` is provided: mutate the provided base level in one pass.
        
        Level editing.
        
        Mutation strategy:
        - Determine mutation_mode (one/all)
        - Determine tilting_mode and apply corresponding deltas

        Returns:
            student initial observation after mutation
        """
        # Keep this arg for cross-env API compatibility.
        del num_edits

        if level is None:
            if self.current_level is None:
                raise ValueError("Must call reset_to_level first")
            base_level = self.current_level
        else:
            base_level = self._coerce_level(level)

        if self.tilting_mode == 'none':
            return self.reset_to_level(base_level)

        from dataclasses import replace

        mutated = self._mutate_level_internal(base_level)
        # Ensure the mutated level carries a fresh seed for subsequent resets.
        mutated = replace(mutated, seed=self._set_level_seed(self._sample_level_seed()))
        return self.reset_to_level(mutated)

    def _mutate_level_internal(
        self,
        level: ScenarioLevel
    ) -> ScenarioLevel:
        """
        Execute level mutation
        
        Mutation strategy:
        1. Determine mutation_mode (one/all)
        2. If one, randomly choose a tilting dimension
        3. Sample delta(s) from [-mutation_range, mutation_range]
        4. Apply deltas based on tilting_mode

        Note: only mutate tilt parameters, do not change scenario_id
        """
        from dataclasses import replace

        if self.tilting_mode == 'none':
            return level

        rng = self._mutation_random_state
        dims = [rng.randint(0, 3)] if self.mutation_mode == 'one' else [0, 1, 2]
        params = ['goal_tilt', 'veh_veh_tilt', 'veh_edge_tilt']

        def _clip_and_round(value: float) -> int:
            return round(float(np.clip(value, *self.tilt_range)))

        if self.tilting_mode in ('global', 'ego'):
            mutations = {}
            if self.mutation_mode == 'one':
                dim = dims[0]
                param = params[dim]
                delta = rng.uniform(-self.mutation_range, self.mutation_range)
                mutations[param] = _clip_and_round(getattr(level, param) + delta)
            else:
                deltas = rng.uniform(-self.mutation_range, self.mutation_range, size=3)
                for param, delta in zip(params, deltas):
                    mutations[param] = _clip_and_round(getattr(level, param) + delta)
            return replace(level, **mutations)

        # per_vehicle mode: update per_vehicle_tilting only
        per = list(self._normalize_per_vehicle_tilting(level.per_vehicle_tilting))
        num_vehicles = self.opponent_k
        if self.mutation_mode == 'one':
            dim = dims[0]
            deltas = rng.uniform(-self.mutation_range, self.mutation_range, size=num_vehicles)
            for i, delta in enumerate(deltas):
                idx = i * 3 + dim
                per[idx] = _clip_and_round(per[idx] + delta)
        else:
            deltas = rng.uniform(
                -self.mutation_range,
                self.mutation_range,
                size=self.per_vehicle_tilting_length,
            )
            for idx, delta in enumerate(deltas):
                per[idx] = _clip_and_round(per[idx] + delta)
        return replace(level, per_vehicle_tilting=tuple(per))
    
    # ========== Batch inference two-phase step ==========

    @staticmethod
    def _extract_opponent_prepared(prepared: Optional[Dict]) -> Optional[Dict]:
        """从 prepared pack 中取 opponent prepared。 / Extract the opponent prepared payload from a prepared pack."""
        if prepared is None:
            return None
        if "opponent_prepared" in prepared or "ego_ctrlsim_prepared" in prepared:
            return prepared.get("opponent_prepared")
        return prepared

    def step_prepare(self, action: np.ndarray) -> Optional[Dict]:
        """Phase 1 delegated to runtime."""
        return self.runtime.step_prepare(action)

    def step_complete(
        self, model_outputs: Optional[Dict]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Phase 2 delegated to runtime."""
        return self.runtime.step_complete(model_outputs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        prepared = self.step_prepare(action)
        opponent_prepared = self._extract_opponent_prepared(prepared)
        if opponent_prepared is None:
            model_output = None
        else:
            teacher = self.runtime.get_single_env_teacher()
            model_output = teacher.run_batched_forward([opponent_prepared])[0]
        return self.step_complete(model_output)

    def _get_single_env_teacher(self):
        return self.runtime.get_single_env_teacher()

    def _step_post_actions(
        self, opponent_actions: Dict[int, Tuple[float, float]]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Shared tail delegated to runtime."""
        return self.runtime.step_post_actions(opponent_actions)
    
    # ========== Level properties and encoding ==========
    
    @property
    def level(self) -> str:
        """
        Return current level string representation
        
        Used for LevelStore storage (string mode)
        """
        if self.current_level is None:
            return ""
        return self.current_level.to_level_string()
    
    def get_level(self) -> str:
        return self.level
    
    @property
    def encoding(self) -> np.ndarray:
        """
        Return current level encoding
        
        Compatible with BipedalWalker: string array
        [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilts, seed]
        
        Used for PLR buffer storage (byte mode)
        """
        if self.current_level is None:
            enc = [0, 0, 0, 0] + [0] * self.per_vehicle_tilting_length + [self.level_seed]
        else:
            scenario_idx = self.scenario_id_to_index.get(
                self.current_level.scenario_id, 0
            )
            normalized_per_vehicle_tilting = self._normalize_per_vehicle_tilting(
                self.current_level.per_vehicle_tilting
            )
            enc = [
                scenario_idx,
                self.current_level.goal_tilt,
                self.current_level.veh_veh_tilt,
                self.current_level.veh_edge_tilt,
                *normalized_per_vehicle_tilting,
                self.current_level.seed,
            ]
        
        # Convert to string array (compatible with BipedalWalker)
        enc_str = [str(x) for x in enc]
        return np.array(enc_str, dtype=self.encoding_u_chars)
    
    def get_encodings(self) -> List[np.ndarray]:
        """Return encoding list (compatible with vectorized env interface)"""
        return [self.encoding]

    def _decode_string_encoding(self, encoding: np.ndarray) -> ScenarioLevel:
        """Decode string array encoding to ScenarioLevel."""
        # Check if the scenario pool mapping needs to be rebuilt
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        (
            scenario_idx,
            goal_tilt,
            veh_veh_tilt,
            veh_edge_tilt,
            per_vehicle_tilting,
            seed,
        ) = ScenarioLevel.decode_encoding_fields(encoding)
        scenario_id = self._resolve_scenario_id(scenario_idx)
        per_vehicle_tilting = self._normalize_per_vehicle_tilting(per_vehicle_tilting)

        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=seed,
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
            per_vehicle_tilting=per_vehicle_tilting,
        )
    
    # ========== Dynamic scenario pool support ==========
    
    def add_scenario(self, scenario_id: str) -> bool:
        """Add new scenario to scenario pool"""
        if not self.dynamic_scenario_pool:
            return False
        
        if scenario_id in self.scenario_id_to_index:
            return False
        
        if len(self.scenario_ids) >= self.max_scenario_pool_size:
            old_id = self.scenario_ids.pop(0)
            old_idx = self.scenario_id_to_index.pop(old_id)
            del self.index_to_scenario_id[old_idx]
        
        new_idx = len(self.scenario_ids)
        self.scenario_ids.append(scenario_id)
        self.scenario_id_to_index[scenario_id] = new_idx
        self.index_to_scenario_id[new_idx] = scenario_id
        
        self._scenario_pool_dirty = True
        return True
    
    def get_scenario_pool_size(self) -> int:
        """Return current scenario pool size"""
        return len(self.scenario_ids)
    
    def rebuild_index_mappings(self):
        """Rebuild index mappings"""
        self.scenario_id_to_index = {
            sid: i for i, sid in enumerate(self.scenario_ids)
        }
        self.index_to_scenario_id = {
            i: sid for i, sid in enumerate(self.scenario_ids)
        }
        self._scenario_pool_dirty = False

    def _resolve_scenario_id(self, scenario_idx: int) -> str:
        """Resolve scenario index to scenario ID with warning and fallback."""
        if scenario_idx not in self.index_to_scenario_id:
            warnings.warn(
                f"Scenario index {scenario_idx} not found in mapping. "
                f"Falling back to first scenario: {self.scenario_ids[0]}"
            )
        return self.index_to_scenario_id.get(scenario_idx, self.scenario_ids[0])
    
    # ========== Metrics and information ==========
    def get_complexity_info(self) -> Dict[str, Any]:
        return get_complexity_info(self)

    def reset_metrics(self):
        sim_reset_metrics(self)
    
    def close(self):
        """Close environment"""
        # Clear cached episode snapshot
        self._last_completed_complexity_info = None
        
        # If recording, stop first
        if self.recording_video:
            self.stop_recording()
        
        # Clean up recorder
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        
        # Clean up Nocturne resources
        if self.sim is not None:
            # TODO: Clean up Nocturne resources
            pass
