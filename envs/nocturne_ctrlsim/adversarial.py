"""
Nocturne + CtRL-Sim environment

Support DCD framework requirements:
- PLR (Prioritized Level Replay) mechanism
- PAIRED / ACCEL etc. UED algorithms (through Adversary interface)
- dynamic scenario pool size
- Level mutation and editing
"""
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from dataclasses import replace

from .config import build_nocturne_ctrlsim_env_config
from .level import (
    ScenarioLevel,
    build_zero_tilt_level,
    normalize_per_vehicle_tilting,
)
from .services.bootstrap import (
    NocturneCtrlSimBootstrapDependencies,
    bootstrap_nocturne_ctrlsim_env,
)

from .services.scenario_helpers import (
    get_vehicle_by_id,
    load_scenario,
    remove_background_moving_vehicles,
)
from .services import level_manager as level_manager_service
from .services import scenario_pool as scenario_pool_service

from .services.gt_helpers import (
    get_goal_point_for_vehicle,
    initialize_ego_goal_state,
)
from .student_env_policy import get_student_observation
from .services.runtime import NocturneCtrlSimRuntime

from .services.simulation_info import (
    get_complexity_info,
    reset_metrics as sim_reset_metrics,
)

from .utils.vehicle_map_helpers import (
    is_retryable_vehicle_map_error,
    load_vehicle_ids_for_scenario,
)
from .utils.encoding_helpers import (
    encode_level_to_string_array,
)
from .utils.tilt_helpers import (
    apply_adversary_tilting_action,
    init_level_params_vec,
)

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
        """kwargs: runtime/environment settings."""
        super().__init__()

        config = build_nocturne_ctrlsim_env_config(
            scenario_index_path=scenario_index_path,
            opponent_checkpoint=opponent_checkpoint,
            scenario_data_dir=scenario_data_dir,
            preprocess_dir=preprocess_dir,
            kwargs=kwargs,
        )
        dependencies = NocturneCtrlSimBootstrapDependencies(
            scenario_index_cls=ScenarioIndex,
            create_minimal_config=create_minimal_config,
            data_bridge_cls=DataBridge,
            opponent_cls=CtrlSimOpponentAdapter,
            runtime_cls=NocturneCtrlSimRuntime,
            load_scenario=load_scenario,
            get_vehicle_by_id=get_vehicle_by_id,
            load_vehicle_ids_for_scenario=load_vehicle_ids_for_scenario,
            initialize_ego_goal_state=initialize_ego_goal_state,
            get_goal_point_for_vehicle=get_goal_point_for_vehicle,
            get_student_observation=get_student_observation,
        )
        bootstrap_nocturne_ctrlsim_env(self, config, dependencies)

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
        self.np_random = np.random.RandomState(seed)

    def _sample_level_seed(self) -> int:
        """Sample a level seed from process-local RNG in int32 range."""
        int32_max = np.iinfo(np.int32).max
        return int(self.np_random.randint(1, int32_max))

    def _set_level_seed(self, seed: int) -> int:
        """Set the current level seed tracked by the environment."""
        self.level_seed = int(seed)
        return self.level_seed

    def _set_current_level(self, level: ScenarioLevel) -> ScenarioLevel:
        """Set current level and synchronize its seed state."""
        self.current_level = level
        self._set_level_seed(level.seed)
        return self.current_level

    def _resolve_initial_ego_reweight_tilt(
        self,
        *,
        tilting_mode: str,
        opponent_runtime_mode: str,
    ) -> Tuple[float, float, float]:
        """Resolve the runtime tilt used for ego-side RTG mismatch computation."""
        if not self.use_policy_reweighting:
            return (0.0, 0.0, 0.0)
        if opponent_runtime_mode != 'normal':
            return (0.0, 0.0, 0.0)
        if tilting_mode != 'ego':
            return (0.0, 0.0, 0.0)

        current_tilt = getattr(self.opponent, 'current_tilt', None)
        if current_tilt is None:
            return (0.0, 0.0, 0.0)
        return (
            float(current_tilt.goal_tilt),
            float(current_tilt.veh_veh_tilt),
            float(current_tilt.veh_edge_tilt),
        )

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
        self.level_params_vec = init_level_params_vec(
            tilting_mode=self.tilting_mode,
            per_vehicle_tilting_length=self.per_vehicle_tilting_length,
        )
        
        # Generate new level seed
        self._set_level_seed(self._sample_level_seed())
        
        return self._build_adversary_obs()

    def _normalize_adversary_action(self, action: Any) -> np.ndarray:
        """Normalize adversary action into a clipped 1D float vector."""

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
        apply_adversary_tilting_action(
            level_params_vec=self.level_params_vec,
            tilting_mode=self.tilting_mode,
            per_vehicle_tilting_length=self.per_vehicle_tilting_length,
            runtime_mode=runtime_mode,
            action_vec=action_vec,
            tilt_range=self.tilt_range,
        )
        
        self.adversary_step_count += 1
        
        # Check if the building is completed
        done = self.adversary_step_count >= self.adversary_max_steps
        
        if done:
            # Building completed, create ScenarioLevel and initialize environment
            level_manager_service.build_level_from_params(self)
        
        return self._build_adversary_obs(), 0, done, {}

    def _build_adversary_obs(self) -> Dict[str, np.ndarray]:
        """Build adversary observation dictionary from current environment state."""
        return {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z(),
        }

    def generate_random_z(self) -> np.ndarray:
        """Generate random condition vector for adversary observation(not used in nocturne-ctrlsim)."""
        return np.zeros((self.random_z_dim,), dtype=np.float32)

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

    def _initialize_simulation(self):
        """Initialize Nocturne simulation (delegated to runtime)."""
        self.runtime.initialize_simulation()
    
    # ========== PLR/DR interface ==========

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
            level = level_manager_service.sample_random_level(self)
            last_scenario_id = level.scenario_id
            try:
                return self.reset_to_level(level)
            except Exception as e:
                if not is_retryable_vehicle_map_error(e):
                    raise
                last_error = e
                if attempt < max_retries:
                    print(
                        f"Warning: reset_random failed for scenario '{last_scenario_id}' "
                        f"({attempt}/{max_retries}) due to strict vehicle-map metadata error: {e}. "
                        "Retrying with another random scenario."
                    )

        raise RuntimeError(
            f"reset_random failed after {max_retries} retries due to strict vehicle-map metadata errors. "
            f"Last scenario: '{last_scenario_id}'. Last error: {last_error}"
        ) from last_error

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
        level = level_manager_service.coerce_level(self, level)
        runtime_mode = getattr(self, 'opponent_runtime_mode', 'normal')

        # Keep replay/disable behavior robust even when replayed levels
        # already carry non-zero tilting fields.
        if runtime_mode != 'normal':
            level = build_zero_tilt_level(
                scenario_id=level.scenario_id,
                seed=level.seed,
                tilting_mode=self.tilting_mode,
                per_vehicle_tilting_length=self.per_vehicle_tilting_length,
            )

        if self._scenario_pool_dirty:
            scenario_pool_service.rebuild_index_mappings(self)
        start_idx = self.scenario_id_to_index.get(level.scenario_id, 0)
        level_manager_service.initialize_level_with_fallback(
            self,
            start_idx=start_idx,
            create_level=lambda scenario_id: level.with_scenario_id(scenario_id),
            context="while loading a replayed or provided level",
        )
        
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
            base_level = level_manager_service.coerce_level(self, level)

        if self.tilting_mode == 'none':
            return self.reset_to_level(base_level)

        mutated = level_manager_service.mutate_level_internal(self, base_level)
        # Ensure the mutated level carries a fresh seed for subsequent resets.
        mutated = replace(mutated, seed=self._set_level_seed(self._sample_level_seed()))
        return self.reset_to_level(mutated)
    
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
        return encode_level_to_string_array(
            current_level=self.current_level,
            scenario_id_to_index=self.scenario_id_to_index,
            normalize_per_vehicle_tilting=lambda values: normalize_per_vehicle_tilting(
                values,
                self.per_vehicle_tilting_length,
            ),
            per_vehicle_tilting_length=self.per_vehicle_tilting_length,
            level_seed=self.level_seed,
            dtype=self.encoding_u_chars,
        )
    
    def get_encodings(self) -> List[np.ndarray]:
        """Return encoding list (compatible with vectorized env interface)"""
        return [self.encoding]

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
