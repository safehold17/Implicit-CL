"""
Nocturne + CtRL-Sim environment

Support DCD framework requirements:
- PLR (Prioritized Level Replay) mechanism
- PAIRED / ACCEL etc. UED algorithms (through Adversary interface)
- dynamic scenario pool size
- Level mutation and editing
"""
import gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

from .level import ScenarioLevel, PER_VEHICLE_TILTING_LENGTH
from .video_recorder import NocturneVideoRecorder
from .visualization import VisualizationMixin
from util.build_scenario_index import ScenarioIndex
from adapters.ctrl_sim import (
    CtrlSimOpponentAdapter,
    DataBridge,
    load_ctrl_sim_config,
    create_minimal_config,
)


# ========== Level parameter ranges ==========
DEFAULT_TILT_RANGE = [-25, 25]  # tilting parameter range
DEFAULT_TILT_MUTATION_STD = 1.0  # perturbation amplitude when mutating (same as config.yaml)
DEFAULT_OBS_DIM = 128  # observation dimension
DEFAULT_ACTION_DIM = 2  # action dimension (accel, steer)

# Default level parameter vector: [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt]
DEFAULT_LEVEL_PARAMS = [0, 0, 0, 0]


def rand_int_seed():
    import os
    # generate 4 bytes (32 bits) random number
    return int.from_bytes(os.urandom(4), byteorder="little")


class NocturneCtrlSimAdversarial(VisualizationMixin, gym.Env):
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
    
    Adversary action space (4 steps building):
    - Step 0: select scenario_id (discrete: map to scenario pool index)
    - Step 1: set goal_tilt (continuous: [-1, 1] -> [-25, 25])
    - Step 2: set veh_veh_tilt (continuous: [-1, 1] -> [-25, 25])
    - Step 3: set veh_edge_tilt (continuous: [-1, 1] -> [-25, 25])
    """
    
    def __init__(
        self,
        scenario_index_path: str,
        opponent_checkpoint: str,
        scenario_data_dir: str,
        preprocess_dir: str,
        opponent_k: int = 7,
        max_episode_steps: int = 90,
        device: str = 'cuda',
        cfg: Any = None,
        seed: int = 0,
        fixed_environment: bool = False,
        # Adversary config
        random_z_dim: int = 50,
        # Dynamic scenario pool config
        dynamic_scenario_pool: bool = False,
        max_scenario_pool_size: int = 10000,
        # Tilting mode
        tilting_mode: str = 'global',

        obs_dim: int = DEFAULT_OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        tilt_range: List[int] = None,
        tilt_mutation_std: float = DEFAULT_TILT_MUTATION_STD,
        **kwargs
    ):
        """
        Args:
            scenario_index_path: Scenario index JSON file path
            opponent_checkpoint
            scenario_data_dir: Nocturne scenario data directory
            preprocess_dir: ctrl-sim preprocessed data directory
            opponent_k: number of opponent vehicles (select the nearest K to ego)
            max_episode_steps: maximum number of steps (default 90, same as ctrl-sim)
            device
            cfg: Hydra config object (optional, if None, create automatically)
            seed
            fixed_environment: whether to fix the environment (for evaluation)
            random_z_dim: random vector dimension (for conditional generation)
            dynamic_scenario_pool: whether to enable dynamic scenario pool
            max_scenario_pool_size: dynamic scenario pool maximum size
        """
        super().__init__()
        
        self.seed_value = seed
        self.fixed_environment = fixed_environment
        np.random.seed(seed)
        
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
        
        # ========== Config loading ==========
        if cfg is None:
            cfg = create_minimal_config(
                checkpoint_path=opponent_checkpoint,
                scenario_dir=scenario_data_dir,
                preprocess_dir=preprocess_dir,
            )
        self.cfg = cfg
        self.scenario_data_dir = scenario_data_dir
        self.preprocess_dir = preprocess_dir
        
        # ========== Data bridge ==========
        self.data_bridge = DataBridge(cfg, preprocess_dir)
        
        # ========== Opponent policy adapter ==========
        self.opponent = CtrlSimOpponentAdapter(
            cfg=cfg,
            checkpoint_path=opponent_checkpoint,
            device=device,
        )
        
        # ========== Environment config ==========
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.opponent_k = opponent_k
        self.dt = cfg.nocturne.dt
        
        # ========== Tilting config ==========
        if tilting_mode not in ['global', 'per_vehicle']:
            raise ValueError(f"tilting_mode must be 'global' or 'per_vehicle', got {tilting_mode}")
        self.tilting_mode = tilting_mode
        
        # ========== State variables ==========
        self.current_level: Optional[ScenarioLevel] = None
        self.current_step = 0
        self.adversary_step_count = 0  # Adversary building steps
        self.level_seed = seed
        
        # Nocturne simulation object
        self.sim = None
        self.scenario = None
        self.vehicles: List = []
        self.ego_vehicle = None
        self.opponent_vehicles: List = []
        self.opponent_vehicle_ids: List[int] = []
        
        # Ground truth and preprocessed data
        self._gt_data_dict: Dict = {}
        self._preproc_data: Optional[Dict] = None
        
        # Ego vehicle's goal and reward related state (for _compute_reward)
        self._ego_goal_dict: Optional[Dict] = None
        self._ego_goal_dist_normalizer: float = 1.0
        self._ego_vehicle_data_dict: Dict = {}  # Track ego's historical data
        
        # Termination condition
        self._collision_occurred: bool = False
        self._goal_reached: bool = False
        self._offroad_occurred: bool = False
        
        # Episode statistics (for training monitoring)
        self._episode_collision_occurred: bool = False
        self._episode_goal_reached: bool = False
        self._episode_offroad_occurred: bool = False
        self._episode_steps: int = 0
        self._episode_progress: float = 0.0  # Target progress [0, 1]
        
        # Level parameters vector (for adversary building)
        # [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilts...]
        if self.tilting_mode == 'per_vehicle':
            self.level_params_vec = list(DEFAULT_LEVEL_PARAMS) + [0] * PER_VEHICLE_TILTING_LENGTH
        else:
            self.level_params_vec = list(DEFAULT_LEVEL_PARAMS)
        
        # ========== Student observation config (from args) ==========
        # These parameters are used in make_agent, set default values here
        self._max_observable_agents = kwargs.get('student_num_neighbors', 16)
        self._top_k_road_points = kwargs.get('student_top_k_road', 64)
        
        # Cache road data (filled after _initialize_simulation)
        self._road_graph_cache: Optional[List[Dict]] = None
        
        # ========== Observation and action space (Student) ==========
        # Calculate Late Fusion observation dimension: ego(6) + partners(K×6) + road_graph(R×13)
        late_fusion_obs_dim = 6 + self._max_observable_agents * 6 + self._top_k_road_points * 13
        
        # Use config obs_dim or calculated dimension (take larger one for compatibility)
        self._obs_dim = max(obs_dim, late_fusion_obs_dim)
        self._action_dim = action_dim
        self.tilt_range = tilt_range if tilt_range is not None else list(DEFAULT_TILT_RANGE)
        self.tilt_mutation_std = tilt_mutation_std
        
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
        # Adversary building steps: scenario_id + tilt parameters
        if self.tilting_mode == 'per_vehicle':
            self.adversary_max_steps = 1 + PER_VEHICLE_TILTING_LENGTH
        else:
            self.adversary_max_steps = 4
        self.random_z_dim = random_z_dim
        self.passable = True  # Driving scenario default passable
        
        # Adversary action space: continuous action [-1, 1]
        # Step 0: map to scenario index
        # Step 1-3: map to tilt parameters
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Adversary observation space
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, 
            high=self.adversary_max_steps, 
            shape=(1,), 
            dtype='uint8'
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
            high=max(len(self.scenario_ids), 25.0),
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
        
        # ========== Random seed ==========
        self.seed_value = seed
    
    # ========== Basic environment interface ==========
    
    def seed(self, seed=None):
        """Set the random seed of the environment"""
        if seed is not None:
            self.level_seed = seed
            self.seed_value = seed
        return [self.level_seed]
    
    # ========== Adversary interface (PAIRED/ACCEL) ==========
    
    def reset(self) -> Dict:
        """
        Reset the environment, prepare adversary building process
        
        Returns:
            adversary observation dictionary: {'image', 'time_step', 'random_z'}
        """
        self.adversary_step_count = 0
        
        # Reset level parameters to default values
        if self.tilting_mode == 'per_vehicle':
            self.level_params_vec = list(DEFAULT_LEVEL_PARAMS) + [0] * PER_VEHICLE_TILTING_LENGTH
        else:
            self.level_params_vec = list(DEFAULT_LEVEL_PARAMS)
        
        # Generate new level seed
        self.level_seed = rand_int_seed()
        
        # Return adversary observation
        obs = {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z()
        }
        
        return obs
    
    def step_adversary(self, action) -> Tuple[Dict, float, bool, Dict]:
        """        
        Action mapping:
        - Step 0: action -> scenario_index (discretized to scenario pool size)
        - Step 1..: action -> tilt parameters ([-1,1] -> tilt_range)
        
        Args:
            action: continuous action [-1, 1]
        
        Returns:
            (obs, reward, done, info)
            - obs: adversary observation
            - reward: always 0 (adversary reward calculated after rollout)
            - done: whether the building is completed
            - info: additional information
        """
        import torch
        if torch.is_tensor(action):
            action = action.item()
        
        # Set parameters according to current step
        if self.adversary_step_count == 0:
            # Step 0: select scenario
            # Map [-1, 1] to [0, num_scenarios-1]
            num_scenarios = len(self.scenario_ids)
            scenario_idx = int((action + 1) / 2 * num_scenarios)
            scenario_idx = np.clip(scenario_idx, 0, num_scenarios - 1)
            self.level_params_vec[0] = scenario_idx
        else:
            # Step 1+: set tilt parameters
            # Map [-1, 1] to tilt_range
            tilt_scale = (self.tilt_range[1] - self.tilt_range[0]) / 2.0
            tilt_value = action * tilt_scale
            tilt_value = np.clip(tilt_value, self.tilt_range[0], self.tilt_range[1])
            if self.tilting_mode == 'per_vehicle':
                per_idx = self.adversary_step_count - 1
                if 0 <= per_idx < PER_VEHICLE_TILTING_LENGTH:
                    self.level_params_vec[4 + per_idx] = round(float(tilt_value))
            else:
                self.level_params_vec[self.adversary_step_count] = round(float(tilt_value))
        
        self.adversary_step_count += 1
        
        # Check if the building is completed
        done = self.adversary_step_count >= self.adversary_max_steps
        
        if done:
            # Building completed, create ScenarioLevel and initialize environment
            self._build_level_from_params()
        
        # Return adversary observation
        obs = {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z()
        }
        
        return obs, 0, done, {}
    
    def _build_level_from_params(self):
        """Build ScenarioLevel from level_params_vec and initialize environment"""

        # Check if the scenario pool mapping needs to be rebuilt
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        
        scenario_idx = int(self.level_params_vec[0])
        
        # Check if the scenario ID exists, if not, warning
        if scenario_idx not in self.index_to_scenario_id:
            import warnings
            warnings.warn(
                f"Scenario index {scenario_idx} not found in mapping. "
                f"Falling back to first scenario: {self.scenario_ids[0]}"
            )
        scenario_id = self.index_to_scenario_id.get(scenario_idx, self.scenario_ids[0])
        
        if self.tilting_mode == 'per_vehicle':
            per_vehicle_tilting = tuple(
                int(round(float(v))) for v in self.level_params_vec[4:4 + PER_VEHICLE_TILTING_LENGTH]
            )
            self.current_level = ScenarioLevel(
                scenario_id=scenario_id,
                seed=self.level_seed,
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=per_vehicle_tilting,
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
        
        # Reset episode statistics
        self._episode_collision_occurred = False
        self._episode_goal_reached = False
        self._episode_offroad_occurred = False
        self._episode_steps = 0
        self._episode_progress = 0.0
        
        # Set random seed
        np.random.seed(level.seed)
        
        # ⚠️ Important: must get GT data first, then load main scenario
        # Reason: get_ground_truth() internally creates a temporary Simulation and steps,
        # this will destroy the global state of Nocturne, causing the vehicle objects in the subsequent Simulation to become invalid (segmentation fault when setting attributes).
        # Solution: get GT data first (let the temporary Simulation complete and destroy),
        # then load main scenario.
        
        # Get ground truth data (need to add .json suffix)
        self._gt_data_dict = self.data_bridge.get_ground_truth(
            self.scenario_data_dir, 
            f"{level.scenario_id}.json"
        )
        
        # Load Nocturne scenario (must after getting GT data)
        self._load_scenario(level.scenario_id)
        
        # Select ego vehicle (need GT data to select interesting pair)
        self.ego_vehicle = self._select_ego_vehicle()
        
        # Load preprocessed data (with check)
        self._preproc_data, file_exists = self.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        if not file_exists:
            raise FileNotFoundError(
                f"Preprocessed data not found for scenario '{level.scenario_id}'. "
                f"Check preprocess_dir: {self.data_bridge.preprocess_dir}"
            )
        
        # Select opponent controlled vehicles (select the nearest k from moving vehicles)
        self._select_opponent_vehicles(k=self.opponent_k)
        
        # Initialize ego vehicle's goal and reward related states
        self._initialize_ego_goal_state()
        
        # Set opponent tilting based on tilting_mode
        if self.tilting_mode == 'global':
            # Global mode: all opponents share the same tilts
            self.opponent.set_tilting(
                level.goal_tilt, 
                level.veh_veh_tilt, 
                level.veh_edge_tilt
            )
        else:  # per_vehicle mode
            # Per-vehicle mode: each opponent has independent tilts
            # Sort opponent_vehicle_ids by veh_id (integer, numerical order)
            sorted_opponent_ids = sorted(self.opponent_vehicle_ids)
            
            # Build per-vehicle mapping: veh_id -> (goal_tilt, veh_veh_tilt, veh_edge_tilt)
            per_vehicle_mapping = {}
            per = level.per_vehicle_tilting
            for i, veh_id in enumerate(sorted_opponent_ids):
                if i * 3 + 2 < len(per):
                    base = 3 * i
                    per_vehicle_mapping[veh_id] = (per[base], per[base+1], per[base+2])
                else:
                    # If insufficient tilts, use (0, 0, 0)
                    per_vehicle_mapping[veh_id] = (0, 0, 0)
            
            # Set per-vehicle tilts via adapter
            self.opponent.set_per_vehicle_tilting(per_vehicle_mapping)
        
        self.opponent.reset(
            self.scenario,
            self.vehicles,
            self._gt_data_dict,
            self._preproc_data,
            self.opponent_vehicle_ids,
        )
        
        # Cache road data (for Student observation)
        self._road_graph_cache = self.data_bridge.get_road_data(self.scenario)
    
    def generate_random_z(self) -> np.ndarray:
        """Generate random condition vector (for adversary observation)"""
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
    
    @property
    def processed_action_dim(self) -> int:
        """Processed action dimension (compatible with AdversarialRunner)"""
        return 1
    
    # ========== PLR/DR interface ==========
    
    def reset_random(self) -> np.ndarray:
        """
        Randomly generate new level and reset
        
        Entry point for DCD Domain Randomization.
        Randomly sample all parameters.
        
        Returns:
            student initial observation
        """
        level = self._sample_random_level()
        return self.reset_to_level(level)
    
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
        # Convert to ScenarioLevel object
        if isinstance(level, str):
            level = ScenarioLevel.from_level_string(level)
        elif isinstance(level, np.ndarray):
            # Process string array format
            if level.dtype.kind == 'U': # string array unicode
                level = self._decode_string_encoding(level)
            else:
                level = ScenarioLevel.from_encoding(level, self.index_to_scenario_id)
        
        self.current_level = level
        
        # Update level_params_vec to keep consistent
        scenario_idx = self.scenario_id_to_index.get(level.scenario_id, 0)
        if self.tilting_mode == 'per_vehicle':
            self.level_params_vec = [
                scenario_idx,
                0,
                0,
                0,
                *level.per_vehicle_tilting,
            ]
        else:
            self.level_params_vec = [
                scenario_idx,
                level.goal_tilt,
                level.veh_veh_tilt,
                level.veh_edge_tilt,
            ]
        self.level_seed = level.seed
        
        # Initialize simulation
        self._initialize_simulation()
        
        # Return student observation
        return self._get_student_observation()
    
    def _decode_string_encoding(self, encoding: np.ndarray) -> ScenarioLevel:
        """Decode string array encoding to ScenarioLevel"""
        from envs.nocturne_ctrlsim.level import PER_VEHICLE_TILTING_LENGTH
        
        # Check if the scenario pool mapping needs to be rebuilt
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        
        # Handle backward compatibility: old format has length 5, new has length 26
        if len(encoding) >= 5 + PER_VEHICLE_TILTING_LENGTH:
            # New format: [scenario_idx, goal, veh_veh, veh_edge, per_vehicle(21), seed]
            per_vehicle_tilting = tuple(int(round(float(encoding[i]))) for i in range(4, 4 + PER_VEHICLE_TILTING_LENGTH))
            seed_idx = 4 + PER_VEHICLE_TILTING_LENGTH
        else:
            # Old format: [scenario_idx, goal, veh_veh, veh_edge, seed]
            per_vehicle_tilting = ()
            seed_idx = 4
        
        scenario_idx = int(float(encoding[0]))
        
        # Check if the scenario ID exists, if not, warning
        if scenario_idx not in self.index_to_scenario_id:
            import warnings
            warnings.warn(
                f"Scenario index {scenario_idx} not found in mapping. "
                f"Falling back to first scenario: {self.scenario_ids[0]}"
            )
        scenario_id = self.index_to_scenario_id.get(scenario_idx, self.scenario_ids[0])
        
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=int(float(encoding[seed_idx])),
            goal_tilt=float(encoding[1]),
            veh_veh_tilt=float(encoding[2]),
            veh_edge_tilt=float(encoding[3]),
            per_vehicle_tilting=per_vehicle_tilting,
        )
    
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
        
        return self._get_student_observation()
    
    def mutate_level(self, num_edits: int = 1) -> np.ndarray: # TODO: not implemented for per_vehicle tilting yet
        """
        Mutate current level and reset
        
        Level editing.
        
        Mutation strategy:
        - Randomly select parameters for perturbation
        - Perturbation direction: -1, 0, +1
        - Perturbation amplitude: Gaussian distribution
        
        Args:
            num_edits: number of mutations
        
        Returns:
            student initial observation after mutation
        """
        if self.current_level is None:
            raise ValueError("Must call reset_to_level first")
        
        if num_edits > 0:
            mutated = self._mutate_level_internal(self.current_level, num_edits)
            self.level_seed = rand_int_seed()  # new seed
            return self.reset_to_level(mutated)
        
        return self.reset_agent()
    
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
        self.current_step += 1  # TODO：use current step then +1
        
        # 1. Opponent policy inference
        opponent_actions = self.opponent.step(self.current_step - 1, self.vehicles)
        
        # 2. Apply student action to ego vehicle
        self._apply_student_action(action)
        
        # 3. Apply opponent action
        for veh_id, (accel, steer) in opponent_actions.items():
            veh = self._get_vehicle_by_id(veh_id)
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
                gt_action = self._get_gt_action(veh_id, self.current_step - 1)
                if gt_action is not None:
                    self.opponent.apply_action(veh, gt_action)
        
        # 5. Record all vehicle actions (for next update_state)
        self.opponent.record_all_actions(
            self.current_step - 1, 
            self.vehicles, 
            opponent_actions
        )
        
        # 6. Step simulation
        self.sim.step(self.dt)
        
        # 7. If recording is enabled, capture current frame
        if self.recording_video and self.video_recorder is not None:
            self.video_recorder.capture_frame(
                self.scenario,
                self.vehicles,
                highlight_vehicle_ids=[self.ego_vehicle.getID()] if self.ego_vehicle else None
            )
        
        # 8. Calculate reward and termination conditions
        obs = self._get_student_observation()
        reward = self._compute_reward()
        
        # Update episode statistics
        self._episode_steps += 1
        if self._collision_occurred:
            self._episode_collision_occurred = True
        if self._goal_reached:
            self._episode_goal_reached = True
        if self._offroad_occurred:
            self._episode_offroad_occurred = True
        
        # Calculate target progress (current distance vs initial distance)
        if self.ego_vehicle and self._ego_goal_dict and self._ego_goal_dist_normalizer > 0:
            ego_pos = self.ego_vehicle.getPosition()
            goal_pos = self._ego_goal_dict['pos']
            current_dist = np.linalg.norm(goal_pos - np.array([ego_pos.x, ego_pos.y]))
            self._episode_progress = max(0.0, 1.0 - current_dist / self._ego_goal_dist_normalizer)
        
        done = self._check_done()
        info = self._get_info()
        
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
        [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilts(21), seed]
        
        Used for PLR buffer storage (byte mode)
        """
        from envs.nocturne_ctrlsim.level import PER_VEHICLE_TILTING_LENGTH
        
        if self.current_level is None:
            enc = DEFAULT_LEVEL_PARAMS + [0] * PER_VEHICLE_TILTING_LENGTH + [self.level_seed]
        else:
            scenario_idx = self.scenario_id_to_index.get(
                self.current_level.scenario_id, 0
            )
            enc = [
                scenario_idx,
                self.current_level.goal_tilt,
                self.current_level.veh_veh_tilt,
                self.current_level.veh_edge_tilt,
                *self.current_level.per_vehicle_tilting,
                self.current_level.seed,
            ]
        
        # Convert to string array (compatible with BipedalWalker)
        enc_str = [str(x) for x in enc]
        return np.array(enc_str, dtype=self.encoding_u_chars)
    
    def get_encodings(self) -> List[np.ndarray]:
        """Return encoding list (compatible with vectorized env interface)"""
        return [self.encoding]
    
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
    
    # ========== Metrics and information ==========
    
    def reset_metrics(self):
        """Reset metrics tracking"""
        self.episode_reward = 0.0
        self.collision_count = 0
        self.goal_reached = False
    
    def get_complexity_info(self) -> Dict[str, Any]:
        """
        Return current level complexity information and episode statistics (for logging and analysis)
        
        Returns:
            Dictionary containing level parameters and episode statistics
        """
        if self.current_level is None:
            return {}
        
        info = {
            # Level parameters
            'scenario_id': self.current_level.scenario_id,
            'seed': self.current_level.seed,
            'opponent_k': self.opponent_k,
            'scenario_pool_size': len(self.scenario_ids),
            
            # Episode statistics (for training monitoring)
            'collision_rate': 1.0 if self._episode_collision_occurred else 0.0,
            'goal_reached_rate': 1.0 if self._episode_goal_reached else 0.0,
            'offroad_rate': 1.0 if self._episode_offroad_occurred else 0.0,
            'avg_progress': self._episode_progress,
            'episode_steps': self._episode_steps,
            'episode_reward': self.episode_reward,
        }

        if self.tilting_mode == 'global':
            info.update({
                'goal_tilt': self.current_level.goal_tilt,
                'veh_veh_tilt': self.current_level.veh_veh_tilt,
                'veh_edge_tilt': self.current_level.veh_edge_tilt,
            })
        else:
            per = self.current_level.per_vehicle_tilting
            for i in range(self.opponent_k):
                base = 3 * i
                info[f'per_vehicle_goal_tilt_{i}'] = per[base]
                info[f'per_vehicle_veh_tilt_{i}'] = per[base + 1]
                info[f'per_vehicle_edge_tilt_{i}'] = per[base + 2]
        
        return info
    
    # ========== Internal helper methods ==========
    
    def _sample_random_level(self) -> ScenarioLevel:
        """Randomly generate level"""
        from envs.nocturne_ctrlsim.level import PER_VEHICLE_TILTING_LENGTH
        
        if self.tilting_mode == 'global':
            # Global mode: sample 3 global tilts, per-vehicle区段置0
            per_vehicle_tilting = tuple([0] * PER_VEHICLE_TILTING_LENGTH)
            return ScenarioLevel(
                scenario_id=np.random.choice(self.scenario_ids),
                seed=rand_int_seed(),
                goal_tilt=round(float(np.random.uniform(*self.tilt_range))),
                veh_veh_tilt=round(float(np.random.uniform(*self.tilt_range))),
                veh_edge_tilt=round(float(np.random.uniform(*self.tilt_range))),
                per_vehicle_tilting=per_vehicle_tilting,
            )
        else:  # per_vehicle mode
            # Per-vehicle mode: global tilts置0, sample 21 per-vehicle tilts
            per_vehicle_tilts = [round(float(np.random.uniform(*self.tilt_range))) for _ in range(PER_VEHICLE_TILTING_LENGTH)]
            return ScenarioLevel(
                scenario_id=np.random.choice(self.scenario_ids),
                seed=rand_int_seed(),
                goal_tilt=0,
                veh_veh_tilt=0,
                veh_edge_tilt=0,
                per_vehicle_tilting=tuple(per_vehicle_tilts),
            )
    
    def _mutate_level_internal(
        self,
        level: ScenarioLevel,
        num_edits: int
    ) -> ScenarioLevel:
        """
        Execute level mutation
        
        Mutation strategy (see BipedalWalker):
        1. Randomly select parameters
        2. Randomly select direction: -1, 0, +1
        3. Apply Gaussian perturbation
        
        Note: only mutate tilt parameters, do not change scenario_id
        """
        from dataclasses import replace
        
        mutations = {}
        params = ['goal_tilt', 'veh_veh_tilt', 'veh_edge_tilt']
        
        for _ in range(num_edits):
            # Only mutate tilt parameters, do not change scenario_id
            param = np.random.choice(params)
            current_val = mutations.get(param, getattr(level, param))
            direction = np.random.randint(-1, 2)  # -1, 0, 1
            mutation = direction * np.random.uniform(0, self.tilt_mutation_std)
            new_val = np.clip(current_val + mutation, *self.tilt_range)
            mutations[param] = round(float(new_val))
        
        return replace(level, **mutations)
    
    def _load_scenario(self, scenario_id: str):
        """
        Load Nocturne scenario
        
        See: third_party/ctrl-sim/utils/sim.py get_sim() function
        """
        import os
        from nocturne import Simulation
        from omegaconf import OmegaConf
        
        scenario_path = os.path.join(self.scenario_data_dir, f"{scenario_id}.json")
        
        # Nocturne Simulation only needs scenario configuration dictionary
        # See cfgs/config.py get_scenario_dict() function
        if 'scenario' in self.cfg.nocturne:
            scenario_dict = OmegaConf.to_container(
                self.cfg.nocturne.scenario, resolve=True
            )
        else:
            # Fall back to basic configuration
            scenario_dict = {
                'start_time': 0,
                'allow_non_vehicles': False,
            }
        
        self.sim = Simulation(scenario_path, scenario_dict)
        self.scenario = self.sim.getScenario()
        self.vehicles = list(self.scenario.vehicles())
        
        # Set vehicle control flags (see ctrl-sim evaluator.py line 37-39)
        for veh in self.vehicles:
            veh.expert_control = False
            veh.physics_simulated = True
        
        # Note: ego selection is moved to _initialize_simulation() because it needs GT data to select interesting pair
    
    def _get_moving_vehicle_ids(self) -> List[int]:
        """
        Get all moving vehicles IDs in the scenario
        
        See: ctrl-sim utils/sim.py get_moving_vehicles() function
        """
        return [v.getID() for v in self.scenario.getObjectsThatMoved()]
    
    def _find_interesting_pair(self, moving_veh_ids: List[int]) -> Optional[Tuple[int, int]]:
        """
        Find interesting vehicle pairs (see ctrl-sim policy_evaluator.py line 362-412)
        
        Selection criteria:
        - Target position is close (<10 meters)
        - Target time step is close (<20 steps)
        - Trajectory is long enough (>=60 steps)
        
        Returns:
            (veh_id_1, veh_id_2) tuple, if no interesting pair is found, use first moving vehicle as ego
        """
        # Configuration thresholds (see ctrl-sim cfg.eval)
        goal_dist_threshold = 10.0  # meters
        timestep_diff_threshold = 20  # steps
        traj_len_threshold = 60  # steps
        history_steps = getattr(self.cfg.nocturne, 'history_steps', 10)
        
        goals = []
        goal_timesteps = []
        valid_traj_mask = []
        veh_ids = []
        
        for veh_id in moving_veh_ids:
            if veh_id not in self._gt_data_dict:
                continue
                
            gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj[:, 4]
            
            # Calculate target position and time step
            idx_goal = self.max_episode_steps - 1
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
            
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
                
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj[idx_goal, :2]
            
            # Check trajectory length
            has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
            
            goals.append(goal_pos)
            goal_timesteps.append(idx_goal - history_steps)
            valid_traj_mask.append(1 if has_valid_traj else 0)
            veh_ids.append(veh_id)
        
        if len(goals) < 2:
            return None
        
        goals = np.array(goals)
        goal_timesteps = np.array(goal_timesteps)
        valid_traj_mask = np.array(valid_traj_mask)
        
        # Calculate target distance matrix
        dists = np.linalg.norm(goals[:, np.newaxis] - goals[np.newaxis, :], axis=-1)
        
        # Build mask
        nearby_mask = dists < goal_dist_threshold
        not_same_mask = dists > 0
        valid_traj_both = np.outer(valid_traj_mask, valid_traj_mask)
        timestep_diff = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
        within_time_mask = timestep_diff < timestep_diff_threshold
        
        goal_mask = nearby_mask & not_same_mask & valid_traj_both.astype(bool) & within_time_mask
        
        indices = np.where(goal_mask)
        valid_pairs = list(zip(indices[0], indices[1]))
        
        if len(valid_pairs) == 0:
            return None
        
        # Deterministic selection: select first pair (sorted by index, to ensure consistency)
        pair_idx = valid_pairs[0]
        return (veh_ids[pair_idx[0]], veh_ids[pair_idx[1]])
    
    def _select_ego_vehicle(self):
        """
        Select ego vehicle (student controlled)
        
        Use find_interesting_pair logic to select two interesting vehicles,
        then deterministically select the vehicle with smaller veh_id as ego.
        
        If no interesting pair is found, use first moving vehicle as ego.
        """
        # 1. Get moving vehicles
        moving_veh_ids = self._get_moving_vehicle_ids()
        
        if len(moving_veh_ids) == 0:
            raise ValueError(
                f"No moving vehicles found in scenario {self.current_level.scenario_id}. "
                "Scenario will be skipped."
            )
        
        # 2. Find interesting pair
        interesting_pair = self._find_interesting_pair(moving_veh_ids)
        
        if interesting_pair is None:
            # If no interesting pair is found, downgrade to select first moving vehicle as ego
            print(
                f"Warning: No interesting vehicle pair found in scenario {self.current_level.scenario_id}. "
                f"Using first moving vehicle as ego."
            )
            if len(moving_veh_ids) > 0:
                ego_veh_id = moving_veh_ids[0]
            else:
                raise ValueError(
                    f"No moving vehicles found in scenario {self.current_level.scenario_id}."
                )
        else:
            # 3. Deterministic selection: select vehicle with smaller veh_id as ego
            ego_veh_id = min(interesting_pair)
        
        return self._get_vehicle_by_id(ego_veh_id)
    
    def _select_opponent_vehicles(self, k: int = 7):
        """
        Select opponent vehicles (k nearest moving vehicles to ego)
        
        See: ctrl-sim distance calculation
        """
        if self.ego_vehicle is None:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # 1. Get moving vehicles (excluding ego)
        moving_veh_ids = self._get_moving_vehicle_ids()
        ego_id = self.ego_vehicle.getID()
        candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]
        
        if len(candidate_ids) == 0:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # 2. Calculate distance to ego
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        
        distances = []
        for veh_id in candidate_ids:
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
            pos = veh.getPosition()
            dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
            distances.append((dist, veh_id, veh))
        
        # 3. Sort by distance, select k nearest vehicles
        distances.sort(key=lambda x: x[0])
        selected = distances[:k]
        
        self.opponent_vehicles = [item[2] for item in selected]
        self.opponent_vehicle_ids = [item[1] for item in selected]
    
    def _get_vehicle_by_id(self, veh_id: int):
        """Get vehicle object by ID"""
        for veh in self.vehicles:
            if veh.getID() == veh_id:
                return veh
        return None
    
    def _initialize_ego_goal_state(self):
        """
        Initialize ego vehicle's target and reward related state
        
        See: ctrl-sim evaluator.py initialize_goal_dict() and compute_goal_dist_normalizer()
        """
        if self.ego_vehicle is None:
            return
        
        ego_id = self.ego_vehicle.getID()
        
        # Get GT trajectory data
        if ego_id not in self._gt_data_dict:
            return
        
        gt_traj_data = np.array(self._gt_data_dict[ego_id]['traj'])
        
        # Calculate target position (see evaluator.py initialize_goal_dict)
        goal_pos = np.array([
            self.ego_vehicle.target_position.x,
            self.ego_vehicle.target_position.y
        ])
        goal_heading = self.ego_vehicle.target_heading
        goal_speed = self.ego_vehicle.target_speed
        
        # Check if vehicle disappears before trajectory ends, if so, use last valid position as target
        existence_mask = gt_traj_data[:, 4]
        idx_disappear = np.where(existence_mask == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1
            if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj_data[idx_goal, :2]
                goal_heading = gt_traj_data[idx_goal, 2]
                goal_speed = gt_traj_data[idx_goal, 3]
        
        self._ego_goal_dict = {
            'pos': goal_pos,
            'heading': goal_heading,
            'speed': goal_speed
        }
        
        # Calculate target distance normalization factor
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        dist = np.linalg.norm(ego_pos - goal_pos)
        self._ego_goal_dist_normalizer = dist if dist > 0 else 1.0
        
        # Initialize ego's vehicle_data_dict (for reward calculation)
        self._ego_vehicle_data_dict = {
            ego_id: {
                'reward': [],
                'position': [],
                'heading': [],
                'speed': [],
            }
        }
    
    def _get_gt_action(self, veh_id: int, t: int) -> Optional[Tuple[float, float]]:
        """
        Get vehicle's action from GT trajectory data at time step t
        
        See: ctrl-sim policy_evaluator.py apply_gt_action()
        
        Args:
            veh_id: vehicle ID
            t: time step
        
        Returns:
            (acceleration, steering) tuple, if data does not exist, return None
        """
        if veh_id not in self._gt_data_dict:
            return None
        
        gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
        
        # Check if time step is valid
        if t < 0 or t >= len(gt_traj) - 1:
            return (0.0, 0.0)
        
        # Check if vehicle exists in current and next time step
        veh_exists = gt_traj[t, 4] and gt_traj[t + 1, 4]
        if not veh_exists:
            return (0.0, 0.0)
        
        # Calculate acceleration (speed difference)
        accel = (gt_traj[t + 1, 3] - gt_traj[t, 3]) / self.dt
        
        # Calculate steering rate (heading difference)
        steer = (gt_traj[t + 1, 2] - gt_traj[t, 2]) / self.dt
        
        return (float(accel), float(steer))
    
    def _apply_student_action(self, action: np.ndarray):
        """
        Apply student action to ego vehicle
        
        Args:
            action: [acceleration, steering] normalized to [-1, 1]
        """
        if self.ego_vehicle is None:
            return
        
        # Convert normalized action to actual values
        # scaling depends on the simu
        accel = action[0] * 10.0  # max acc 10 m/s²
        steer = action[1] * 0.7  # max steer 0.7 rad
        
        if accel > 0:
            self.ego_vehicle.acceleration = accel
        else:
            self.ego_vehicle.brake(abs(accel))
        self.ego_vehicle.steering = steer
    
    def _build_road_graph_obs(
        self, 
        ego_pos, 
        ego_heading: float
    ) -> List[np.ndarray]:
        """
        Build Road Graph observation (in gpudrive)
        
        Road Graph features (13 dimensions):
        - pos_x, pos_y (2): position of road point relative to ego
        - length (1): length of road segment
        - scale_x, scale_y (2): scale of road point
        - orientation (1): road direction
        - type_onehot (7): road type one-hot encoding
        
        Args:
            ego_pos: ego vehicle position
            ego_heading: ego vehicle heading
        
        Returns:
            road_graph_states: List of road point features (R 13-dimensional vectors)
        """
        if self._road_graph_cache is None or len(self._road_graph_cache) == 0:
            # No road data, return empty road graph
            return [np.zeros(13, dtype=np.float32) for _ in range(self._top_k_road_points)]
        
        # Extract road point features
        road_points = []
        
        for road_item in self._road_graph_cache:
            road_type = road_item['type']
            geometry = road_item['geometry']
            
            # Process different types of geometry data
            if isinstance(geometry, list) and len(geometry) > 0:
                # Road line (multiple points)
                for i, pt in enumerate(geometry):
                    # Relative position
                    rel_x = pt['x'] - ego_pos.x
                    rel_y = pt['y'] - ego_pos.y
                    
                    # Calculate road segment length
                    if i < len(geometry) - 1:
                        next_pt = geometry[i + 1]
                        seg_length = np.sqrt(
                            (next_pt['x'] - pt['x'])**2 + 
                            (next_pt['y'] - pt['y'])**2
                        )
                        # Direction: points to next point
                        orientation = np.arctan2(
                            next_pt['y'] - pt['y'],
                            next_pt['x'] - pt['x']
                        )
                    else:
                        seg_length = 1.0  # Default value
                        orientation = 0.0
                    
                    # Road point scale (default value)
                    scale_x = 1.0
                    scale_y = 1.0
                    
                    # Road type one-hot (7 dimensions)
                    # ctrl-sim: {none:0, lane:1, road_line:2, road_edge:3, stop_sign:4, crosswalk:5, speed_bump:6, other:7}
                    # gpudrive: {None:0, RoadLine:1, RoadEdge:2, RoadLane:3, CrossWalk:4, SpeedBump:5, StopSign:6}
                    # Map ctrlsim road type to gpudrive order
                    type_mapping = {
                        'none': 0,
                        'road_line': 1,
                        'road_edge': 2,
                        'lane': 3,
                        'crosswalk': 4,
                        'speed_bump': 5,
                        'stop_sign': 6,
                        'other': 0, 
                    }
                    type_idx = type_mapping.get(road_type, 0)
                    type_onehot = np.zeros(7, dtype=np.float32)
                    type_onehot[type_idx] = 1.0
                    
                    # Concatenate features (13 dimensions)
                    road_feat = np.array([
                        rel_x,
                        rel_y,
                        seg_length,
                        scale_x,
                        scale_y,
                        orientation,
                        *type_onehot
                    ], dtype=np.float32)
                    
                    road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))
            
            elif isinstance(geometry, dict):
                # Static object (e.g. stop_sign)
                rel_x = geometry['x'] - ego_pos.x
                rel_y = geometry['y'] - ego_pos.y
                
                type_mapping = {
                    'stop_sign': 6,
                    'crosswalk': 4,
                    'speed_bump': 5,
                }
                type_idx = type_mapping.get(road_type, 0)
                type_onehot = np.zeros(7, dtype=np.float32)
                type_onehot[type_idx] = 1.0
                
                road_feat = np.array([
                    rel_x,
                    rel_y,
                    0.0,  # length
                    1.0,  # scale_x
                    1.0,  # scale_y
                    0.0,  # orientation
                    *type_onehot
                ], dtype=np.float32)
                
                road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))
        
        # Sort by distance, select top_k nearest points
        road_points.sort(key=lambda x: x[0])
        
        road_graph_states = []
        num_valid_points = min(len(road_points), self._top_k_road_points)
        
        for i in range(num_valid_points):
            road_graph_states.append(road_points[i][1])
        
        # Fill missing road points
        for _ in range(self._top_k_road_points - num_valid_points):
            road_graph_states.append(np.zeros(13, dtype=np.float32))
        
        return road_graph_states

    def _get_student_observation(self) -> np.ndarray:
        """
        Get student policy observation (consistent with gpudrive)
        
        Observation vector structure:
        - Ego state: [speed, length, width, rel_goal_x, rel_goal_y, collision_state] (6 dimensions)
        - Partners: K vehicles * [speed, rel_pos_x, rel_pos_y, rel_orientation, length, width] (K*6 dimensions)
        - Road graph: R points * [pos_x, pos_y, length, scale_x, scale_y, orientation, type_onehot(7)] (R*13 dimensions)
        
        Returns:
            Observation vector, shape (obs_dim,)
        """
        if self.ego_vehicle is None or self._ego_goal_dict is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        
        # ========== Ego state (6 dimensions) ==========
        ego_pos = self.ego_vehicle.getPosition()
        ego_heading = self.ego_vehicle.getHeading()
        ego_speed = self.ego_vehicle.getSpeed()
        
        # Relative target position (in ego coordinate system)
        goal_pos = self._ego_goal_dict['pos']
        rel_goal_x = goal_pos[0] - ego_pos.x
        rel_goal_y = goal_pos[1] - ego_pos.y
        
        # Collision state
        collision_state = 1.0 if self._collision_occurred else 0.0
        
        ego_state = np.array([
            ego_speed,
            self.ego_vehicle.getLength(),
            self.ego_vehicle.getWidth(),
            rel_goal_x,
            rel_goal_y,
            collision_state,
        ], dtype=np.float32)
        
        # ========== Partner state (K*6 dimensions) ==========
        # Use num_neighbors configured in args, get from __init__
        max_neighbors = getattr(self, '_max_observable_agents', 16)
        partner_states = []
        
        # Select nearest K neighboring vehicles
        num_neighbors = min(len(self.opponent_vehicles), max_neighbors)
        
        for i in range(num_neighbors):
            veh = self.opponent_vehicles[i]
            veh_pos = veh.getPosition()
            
            # Relative position to ego
            rel_pos_x = veh_pos.x - ego_pos.x
            rel_pos_y = veh_pos.y - ego_pos.y
            
            # Relative orientation
            rel_orientation = veh.getHeading() - ego_heading
            # Normalize to [-pi, pi]
            while rel_orientation > np.pi:
                rel_orientation -= 2 * np.pi
            while rel_orientation < -np.pi:
                rel_orientation += 2 * np.pi
            
            partner_state = np.array([
                veh.getSpeed(),
                rel_pos_x,
                rel_pos_y,
                rel_orientation,
                veh.getLength(),
                veh.getWidth(),
            ], dtype=np.float32)
            partner_states.append(partner_state)
        
        # Fill missing neighbors with zero vector
        for _ in range(max_neighbors - num_neighbors):
            partner_states.append(np.zeros(6, dtype=np.float32))
        
        # ========== Road Graph (R*13 dimensions) ==========
        road_graph_states = self._build_road_graph_obs(ego_pos, ego_heading)
        
        # ========== Concatenate all observations ==========
        obs_parts = [ego_state]
        obs_parts.extend(partner_states)
        obs_parts.extend(road_graph_states)
        
        obs_concat = np.concatenate(obs_parts)
        
        # Fill or truncate to obs_dim
        if len(obs_concat) < self._obs_dim:
            obs_final = np.zeros(self._obs_dim, dtype=np.float32)
            obs_final[:len(obs_concat)] = obs_concat
        else:
            obs_final = obs_concat[:self._obs_dim]
        
        return obs_final
    
    def _compute_reward(self) -> float:     
        """
        Compute student reward
        
        Reward components (see ctrl-sim compute_reward):
        - Goal achievement reward (shaped_goal_distance)
        - Collision penalty
        - Offroad penalty
        
        Returns:
            Scalar reward value
        """
        import nocturne
        
        if self.ego_vehicle is None or self._ego_goal_dict is None:
            return 0.0
        
        reward = 0.0
        ego_id = self.ego_vehicle.getID()
        
        # ========== Get current state ==========
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        ego_speed = self.ego_vehicle.getSpeed()
        ego_heading = self.ego_vehicle.getHeading()
        
        goal_pos = self._ego_goal_dict['pos']
        goal_speed = self._ego_goal_dict['speed']
        goal_heading = self._ego_goal_dict['heading']
        
        # ========== Goal achievement detection ==========
        dist_to_goal = np.linalg.norm(goal_pos - ego_pos)
        position_tolerance = 1.0  # meters
        speed_tolerance = 1.0  # m/s
        heading_tolerance = 0.3  # rad
        
        position_achieved = dist_to_goal < position_tolerance
        speed_achieved = abs(ego_speed - goal_speed) < speed_tolerance
        heading_achieved = abs(self._angle_diff(ego_heading, goal_heading)) < heading_tolerance
        
        # If goal already achieved, keep achieved state
        if self._goal_reached:
            position_achieved = True
        elif position_achieved and speed_achieved and heading_achieved:
            self._goal_reached = True
        
        # ========== Shaped Goal Distance Reward ==========
        # The closer to the goal, the higher the reward 
        # in ctrlsim: 0.2
        goal_dist_scaling = 0.2
        reward_scaling = 1.0
        
        if self._ego_goal_dist_normalizer > 0:
            # Normalize distance reward: [0, 1], the closer the higher
            if self._goal_reached:
                pos_goal_rew = goal_dist_scaling / reward_scaling
            else:
                pos_goal_rew = goal_dist_scaling * (1 - dist_to_goal / self._ego_goal_dist_normalizer) / reward_scaling
                pos_goal_rew = max(0.0, pos_goal_rew)  # 确保非负
        else:
            pos_goal_rew = 0.0
        
        reward += pos_goal_rew
        
        # ========== Collision penalty ==========
        try:
            veh_veh_collision = self.ego_vehicle.collision_type_veh == nocturne.CollisionType.VEHICLE_VEHICLE
            veh_edge_collision = self.ego_vehicle.collision_type_edge == nocturne.CollisionType.VEHICLE_ROAD
        except AttributeError:
            # If nocturne version does not support, use old API 
            try:
                veh_veh_collision = self.ego_vehicle.collision_type == nocturne.CollisionType.VEHICLE_VEHICLE
                veh_edge_collision = self.ego_vehicle.collision_type == nocturne.CollisionType.VEHICLE_ROAD
            except:
                veh_veh_collision = False
                veh_edge_collision = False
        
        collision_penalty = -1.0
        if veh_veh_collision:
            reward += collision_penalty
            self._collision_occurred = True
        
        if veh_edge_collision:
            reward += collision_penalty * 0.5  # Offroad penalty slightly lighter
            self._offroad_occurred = True
        
        # ========== Update vehicle_data_dict (for continuous tracking) ==========
        if ego_id in self._ego_vehicle_data_dict:
            self._ego_vehicle_data_dict[ego_id]['reward'].append([
                float(position_achieved),
                float(heading_achieved),
                float(speed_achieved),
                pos_goal_rew,
                0.0,  # speed_goal_rew
                0.0,  # heading_goal_rew
                float(veh_veh_collision),
                float(veh_edge_collision),
            ])
        
        self.episode_reward += reward
        return reward
    
    def _angle_diff(self, a: float, b: float) -> float:
        """Calculate the difference between two angles (handle wraparound)"""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _check_done(self) -> bool:
        """
        Check if episode is done
        
        终止条件：
        1. Max steps (timeout)
        2. Collision
        3. Goal reached
        4. Offroad
        
        Returns:
            Whether to terminate
        """
        # Max steps (timeout)
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Collision (vehicle-vehicle)
        if self._collision_occurred:
            return True
        
        # Goal reached
        if self._goal_reached:
            return True
        
        # Offroad - optional, some scenarios may not need immediate termination
        # if self._offroad_occurred:
        #     return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information"""
        # Calculate progress (distance to goal)
        progress = 0.0
        if self.ego_vehicle is not None and self._ego_goal_dict is not None:
            ego_pos = self.ego_vehicle.getPosition()
            dist_to_goal = np.linalg.norm(
                self._ego_goal_dict['pos'] - np.array([ego_pos.x, ego_pos.y])
            )
            if self._ego_goal_dist_normalizer > 0:
                progress = 1.0 - dist_to_goal / self._ego_goal_dist_normalizer
                progress = max(0.0, min(1.0, progress))
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            # Diagnostic information (参考 ctrl-sim metrics)
            'collision': self._collision_occurred,
            'goal_reached': self._goal_reached,
            'offroad': self._offroad_occurred,
            'progress': progress,
        }
        
        # Add statistics when episode ends
        if self._check_done():
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step,
            }
            info.update(self.get_complexity_info())
        
        return info
    
    def close(self):
        """Close environment"""
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
