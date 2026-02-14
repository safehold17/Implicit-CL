"""
Nocturne + CtRL-Sim environment

Support DCD framework requirements:
- PLR (Prioritized Level Replay) mechanism
- PAIRED / ACCEL etc. UED algorithms (through Adversary interface)
- dynamic scenario pool size
- Level mutation and editing
"""
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from .level import ScenarioLevel

from .scenario_helpers import (
    get_vehicle_by_id,
    load_scenario,
    remove_background_moving_vehicles,
)

from .opponent_policy import (
    get_goal_point_for_vehicle,
    get_gt_action,
    initialize_ego_goal_state,
    is_ego_position_reached,
)
from .student_reward import compute_student_reward
from .student_env_policy import (
    apply_student_action,
    get_student_observation,
)

from .simulation_info import (
    check_done,
    get_complexity_info,
    get_info,
    reset_metrics as sim_reset_metrics,
)

from .vehicle_map_helpers import load_vehicle_ids_for_scenario

from .video_recorder import NocturneVideoRecorder

from . import visualization as viz

from tools.build_scenario_index import ScenarioIndex

from adapters.ctrl_sim import (
    CtrlSimOpponentAdapter,
    DataBridge,
    create_minimal_config,
)


def rand_int_seed() -> int:
    # generate 4 bytes (32 bits) random number
    return int.from_bytes(os.urandom(4), byteorder="little")


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
        obs_dim = kwargs.get('obs_dim', 128)
        action_dim = kwargs.get('action_dim', 2)
        tilt_range = kwargs.get('tilt_range', None)

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
        self.scenario_data_dir = scenario_data_dir
        self.preprocess_dir = preprocess_dir
        
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
        self.data_bridge = DataBridge(cfg, preprocess_dir)
        
        # ========== Opponent policy adapter ==========
        self.opponent = CtrlSimOpponentAdapter(
            cfg=cfg,
            checkpoint_path=opponent_checkpoint,
            device=device,
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
        self.opponent_runtime_mode = kwargs.get('opponent_runtime_mode', 'normal')
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
        self.ego_vehicle = None
        self.opponent_vehicles: List = []
        self.opponent_vehicle_ids: List[int] = []
        self.ego_selection_mode: str = "unknown"
        
        # Ground truth and preprocessed data
        self._gt_data_dict: Dict = {}
        self._gt_traj_cache: Dict = {}
        self._preproc_data: Optional[Dict] = None
        
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
        self._last_completed_episode_info: Optional[Dict] = None
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
        self.use_pos_shaped = kwargs.get('use_pos_shaped', False)
        self.reset_random_max_retries = int(kwargs.get('reset_random_max_retries', 10))
        
        # Cache road data (filled after _initialize_simulation)
        self._road_graph_cache: Optional[List[Dict]] = None
        
        # ========== Observation and action space (Student) ==========
        # Calculate Late Fusion observation dimension: ego(6) + partners(K×6) + road_graph(R×13)
        late_fusion_obs_dim = 6 + self._max_observable_agents * 6 + self._top_k_road_points * 13
        
        # Use config obs_dim or calculated dimension (take larger one for compatibility)
        self._obs_dim = max(obs_dim, late_fusion_obs_dim)
        self._action_dim = action_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._obs_dim,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self._action_dim,), 
            dtype=np.float32
        )
        
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
        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))
        
        # ========== Metrics tracking ==========
        self.reset_metrics()
        
        # ========== Video recording ==========
        self.video_recorder: Optional[NocturneVideoRecorder] = None
        self.recording_video = False

    def _set_process_seed(self, seed: int, reseed_numpy: bool = True) -> None:
        """Set process-level RNG streams used by this environment instance."""
        seed = int(seed)
        self.seed_value = seed
        if reseed_numpy:
            np.random.seed(seed)
        # Keep mutation randomness process-specific and reproducible.
        self._mutation_random_state = np.random.RandomState(seed)

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
            self._set_process_seed(seed, reseed_numpy=False)
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
        self._set_level_seed(rand_int_seed())
        
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
        """Initialize Nocturne simulation (internal method)"""
        if self.current_level is None:
            return
            
        level = self.current_level
        self.current_step = 0
        self.reset_metrics()
        
        # Reset termination condition states
        self._collision_occurred = False
        self._goal_reached = False
        self._offroad_occurred = False
        self._position_reached = False
        
        # Reset episode statistics
        self._episode_collision_occurred = False
        self._episode_goal_reached = False
        self._episode_offroad_occurred = False
        self._episode_position_reached = False
        self._episode_steps = 0
        self._episode_progress = 0.0
        
        # Set random seed
        np.random.seed(level.seed)
        
        # Important: must get GT data first, then load main scenario
        # Reason: get_ground_truth() internally creates a temporary Simulation and steps,
        # this will destroy the global state of Nocturne, causing the vehicle objects in the subsequent Simulation
        # to become invalid (segmentation fault when setting attributes).
        # Solution: get GT data first (let the temporary Simulation complete and destroy),
        # then load main scenario.
        
        # Get ground truth data (need to add .json suffix)
        self._gt_data_dict = self.data_bridge.get_ground_truth(
            self.scenario_data_dir, 
            f"{level.scenario_id}.json"
        )
        self._gt_traj_cache = {
            veh_id: np.asarray(data["traj"])
            for veh_id, data in self._gt_data_dict.items()
            if isinstance(data, dict) and "traj" in data
        }
        
        # Load Nocturne scenario (must after getting GT data)
        load_scenario(self, level.scenario_id)
        
        # Load vehicle IDs from map (strict mode: no dynamic fallback)
        ego_id, opponent_ids, ego_selection_mode = load_vehicle_ids_for_scenario(self, level.scenario_id)
        self.ego_vehicle = get_vehicle_by_id(self, ego_id)
        if self.ego_vehicle is None:
            raise ValueError(
                f"ego_vehicle_id {ego_id} from vehicle map does not exist in scenario '{level.scenario_id}'."
            )
        self.ego_selection_mode = ego_selection_mode
        
        # Load preprocessed data (with check)
        self._preproc_data, file_exists = self.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        if not file_exists:
            raise FileNotFoundError(
                f"Preprocessed data not found for scenario '{level.scenario_id}'. "
                f"Check preprocess_dir: {self.data_bridge.preprocess_dir}"
            )
        
        runtime_mode = getattr(self, 'opponent_runtime_mode', 'normal')

        # Select opponent vehicles from map only (strict mode), unless runtime mode disables opponents.
        if runtime_mode == 'disable':
            self.opponent_vehicle_ids = []
            self.opponent_vehicles = []
        else:
            self.opponent_vehicle_ids = opponent_ids
            self.opponent_vehicles = []
            missing_opponent_ids = []
            for vid in opponent_ids:
                veh = get_vehicle_by_id(self, vid)
                if veh is None:
                    missing_opponent_ids.append(vid)
                else:
                    self.opponent_vehicles.append(veh)
            if missing_opponent_ids:
                raise ValueError(
                    f"opponent_vehicle_ids {missing_opponent_ids} from vehicle map do not exist in scenario "
                    f"'{level.scenario_id}'."
                )
        
        # Initialize ego vehicle's goal and reward related states
        initialize_ego_goal_state(self)
        self._goal_points_by_id = {}
        if self.ego_vehicle is not None and self._ego_goal_dict is not None:
            self._goal_points_by_id[self.ego_vehicle.getID()] = self._ego_goal_dict['pos']
        for veh_id in self.opponent_vehicle_ids:
            goal_pos = get_goal_point_for_vehicle(self, veh_id)
            if goal_pos is not None:
                self._goal_points_by_id[veh_id] = goal_pos

        # If in per-vehicle tilting mode, zero out tilts for non-existent opponents
        if self.tilting_mode == 'per_vehicle' and self.current_level is not None:
            actual_n = len(self.opponent_vehicle_ids)
            per = list(self._normalize_per_vehicle_tilting(self.current_level.per_vehicle_tilting))
            cutoff = actual_n * 3
            if cutoff < len(per):
                for i in range(cutoff, len(per)):
                    per[i] = 0
                self.current_level.per_vehicle_tilting = tuple(per)
                # Keep level_params_vec in sync if present
                if len(self.level_params_vec) >= 4 + self.per_vehicle_tilting_length:
                    for i in range(self.per_vehicle_tilting_length):
                        self.level_params_vec[4 + i] = per[i]
        
        # Set opponent behavior for current runtime mode.
        if runtime_mode == 'normal':
            if self.tilting_mode == 'global':
                # Global mode: all opponents share the same tilts
                self.opponent.set_tilting(
                    level.goal_tilt,
                    level.veh_veh_tilt,
                    level.veh_edge_tilt
                )
            elif self.tilting_mode == 'per_vehicle':
                # Per-vehicle mode: each opponent has independent tilts
                sorted_opponent_ids = sorted(self.opponent_vehicle_ids)
                per_vehicle_mapping = {}
                per = level.per_vehicle_tilting
                for i, veh_id in enumerate(sorted_opponent_ids):
                    if i * 3 + 2 < len(per):
                        base = 3 * i
                        per_vehicle_mapping[veh_id] = (per[base], per[base+1], per[base+2])
                    else:
                        per_vehicle_mapping[veh_id] = (0, 0, 0)
                self.opponent.set_per_vehicle_tilting(per_vehicle_mapping)
            else:
                # Ego/none mode: opponents always use zero tilts
                self.opponent.set_tilting(0, 0, 0)
        else:
            # disable/replay mode: no policy tilting behavior.
            self.opponent.set_tilting(0, 0, 0)
            self.opponent.per_vehicle_tilting = None

        if self.remove_background_vehicles:
            remove_background_moving_vehicles(self)
        vehicles_to_control = self.opponent_vehicle_ids if runtime_mode == 'normal' else []

        self.opponent.reset(
            self.scenario,
            self.vehicles,
            self._gt_data_dict,
            self._preproc_data,
            vehicles_to_control,
            ego_id=self.ego_vehicle.getID() if self.ego_vehicle else None,
        )
        
        # Cache road data (for Student observation)
        self._road_graph_cache = self.data_bridge.get_road_data(self.scenario)
    
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
        seed = rand_int_seed()

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
        mutated = replace(mutated, seed=self._set_level_seed(rand_int_seed()))
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step simulation (Student action)
        
        Data flow:
        1. Opponent policy inference (using CtrlSimOpponentAdapter)
        2. Apply all actions to Nocturne simulation
        3. Step simulation
        4. Calculate reward and termination conditions
        5. If recording is enabled, capture current frame
        """
        self.current_step += 1
        
        # 1. Opponent policy inference
        opponent_actions = self.opponent.step(self.current_step - 1, self.vehicles)
        
        # 2. Apply student action to ego vehicle
        apply_student_action(self, action)
        
        # 3. Apply opponent action
        for veh_id, (accel, steer) in opponent_actions.items():
            veh = get_vehicle_by_id(self, veh_id)
            if veh is not None:
                self.opponent.apply_action(veh, (accel, steer))
        
        # 4. Apply GT action to uncontrolled vehicles (see policy_evaluator.py line 536-538)
        ego_id = self.ego_vehicle.getID() if self.ego_vehicle else None
        controlled_ids = set(opponent_actions.keys())
        if ego_id is not None:
            controlled_ids.add(ego_id)
        
        for veh in self.vehicles:
            veh_id = veh.getID()
            if veh_id not in controlled_ids:
                gt_action = get_gt_action(self, veh_id, self.current_step - 1, veh)
                if gt_action is not None:
                    self.opponent.apply_action(veh, gt_action)
        
        # 5. Record all vehicle actions (for next update_state)
        self.opponent.record_all_actions(
            self.current_step - 1, 
            self.vehicles, 
            opponent_actions
        )
        
        # 6. Step simulation
        if hasattr(self.opponent, "cache_last_valid_positions"):
            self.opponent.cache_last_valid_positions(self.vehicles)
        self.sim.step(self.dt)
        if hasattr(self.opponent, "post_step_fix_opponent_positions"):
            self.opponent.post_step_fix_opponent_positions(
                self.vehicles,
                self._goal_points_by_id,
                self.current_step,
            )
        
        # 7. If recording is enabled, capture current frame
        if self.recording_video and self.video_recorder is not None:
            self.video_recorder.capture_frame(
                self.scenario,
                self.vehicles,
                roads_data=self._road_graph_cache,
                highlight_vehicle_ids=[self.ego_vehicle.getID()] if self.ego_vehicle else None,
                opponent_vehicle_ids=self.opponent_vehicle_ids,
                goal_points_by_id=self._goal_points_by_id,
                scenario_id=getattr(self.current_level, "scenario_id", None) if self.current_level else None,
                show_vehicle_ids=getattr(self, "recording_show_vehicle_ids", False),
            )
        
        # 8. Calculate reward and termination conditions
        obs = get_student_observation(self)
        reward = compute_student_reward(self)
        
        # Update episode statistics
        self._episode_steps += 1
        if self._collision_occurred:
            self._episode_collision_occurred = True
        if self._goal_reached:
            self._episode_goal_reached = True
        if self._offroad_occurred:
            self._episode_offroad_occurred = True
        self._position_reached = is_ego_position_reached(self)
        if self._position_reached:
            self._episode_position_reached = True
        
        # Calculate target progress (current distance vs initial distance)
        if self.ego_vehicle and self._ego_goal_dict and self._ego_goal_dist_normalizer > 0:
            ego_pos = self.ego_vehicle.getPosition()
            goal_pos = self._ego_goal_dict['pos']
            current_dist = np.linalg.norm(goal_pos - np.array([ego_pos.x, ego_pos.y]))
            self._episode_progress = max(0.0, 1.0 - current_dist / self._ego_goal_dist_normalizer)
        
        done = check_done(self)
        if done:
            self.opponent.finalize(self.vehicles)
        info = get_info(self)
        
        return obs, reward, done, info
    
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
        # Clear cached episode statistics
        self._last_completed_episode_info = None
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
