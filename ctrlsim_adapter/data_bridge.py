"""
负责在 DCD 运行环境与 ctrl-sim 所需数据格式之间做双向桥接。
该模块统一提供 ground truth、预处理特征、道路信息和场景索引等访问入口。
Bridges the DCD runtime environment with the data format expected by ctrl-sim.
Centralizes access to ground truth, preprocessed features, road data, and scenario indexing.
"""

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim

from ._data_bridge import ground_truth as _ground_truth
from ._data_bridge import preprocessed as _preprocessed
from ._data_bridge import scenario as _scenario


class DataBridge:
    """
    Data bridge: Get the data format required by ctrl-sim in the DCD environment
    
    Main functions:
    1. Get ground truth state data
    2. Load preprocessed data (RTG, road information)
    3. Identify moving vehicles
    4. Get road data
    
    - evaluators/evaluator.py: load_preprocessed_data()
    - evaluators/policy_evaluator.py: load test_filenames.pkl to index scenarios
    """
    
    def __init__(
        self,
        cfg: Any,
        preprocess_dir: str,
        preproc_cache_size: int = 64,
    ):
        """
        Args:
            cfg: Hydra
            preprocess_dir
        """
        self.cfg = cfg
        self.preprocess_dir = preprocess_dir
        self.preproc_cache_size = max(0, int(preproc_cache_size))
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.cfg_dataset = cfg.dataset.waymo

        # ctrl-sim preprocessed dataset
        # preprocess_dir must directly point to the directory containing *_physics.pkl files.
        self.preprocessed_dset: Optional[RLWaymoDatasetCtRLSim] = None
        if preprocess_dir:
            try:
                self.cfg.dataset.waymo.preprocess_dir = preprocess_dir
            except Exception:
                pass
            # Use empty split_name so RLWaymoDataset loads files from preprocess_dir/*.pkl.
            self.preprocessed_dset = RLWaymoDatasetCtRLSim(cfg, split_name='', mode='eval')
        
        # Preprocessed files cache
        self._preprocessed_files_cache: Optional[Dict[str, str]] = None
        self._preprocessed_indices_cache: Optional[Dict[str, int]] = None
        self._preproc_data_cache: OrderedDict[str, Dict] = OrderedDict()
    
    def _ensure_preprocessed_files_cache(self):
        _preprocessed.ensure_preprocessed_files_cache(self)
    
    def get_ground_truth(
        self, 
        scenario_path: str,
        scenario_filename: str
    ) -> Dict:
        """
        utils/sim.py get_ground_truth_states()
        
        Args:
            scenario_path
            scenario_filename
        
        Returns:
            gt_data_dict: {veh_id: {"traj": [...], "type": [...]}}
                - traj: shape (steps+1, 8) 
                    [pos_x, pos_y, heading, speed, existence, goal_x, goal_y, length]
                - type: one-hot
        """
        return _ground_truth.get_ground_truth(self, scenario_path, scenario_filename)
    
    def get_ground_truth_from_sim(
        self,
        sim,
        scenario_filename: str
    ) -> Dict:
        """
        From the loaded simulator, get the ground truth state
        
        This is a more efficient version, avoiding reloading the scene
        
        Args:
            sim: Nocturne Simulation object
            scenario_filename: Scene file name (for logging)
        
        Returns:
            gt_data_dict: Same as get_ground_truth
        """
        return _ground_truth.get_ground_truth_from_sim(self, sim, scenario_filename)
    
    def load_preprocessed_data(
        self, 
        scenario_filename: str
    ) -> Tuple[Optional[Dict], bool]:
        """
        Load preprocessed data (including RTG and road information)
        
        Refer to evaluators/evaluator.py lines 47-59 load_preprocessed_data()
        and datasets/rl_waymo/dataset_ctrl_sim.py get_data() method
        
        Args:
            scenario_filename: Scenario filename (without extension, e.g., 'tfrecord-00011-of-00150_131')
        
        Returns:
            preproc_data: Preprocessed data dictionary, including:
                - 'rtgs': shape (num_agents, steps+1, 5) - RTG values
                - 'road_points': Road points information
                - 'road_types': Road types information
            file_exists: Whether the file exists
        """
        return _preprocessed.load_preprocessed_data(self, scenario_filename)
    
    def _process_preprocessed_data(self, raw_data: Dict) -> Dict:
        """
        Process original preprocessed data, calculate RTG
        
        Refer to dataset_ctrl_sim.py get_data() method (mode='eval' branch)
        """
        return _preprocessed.process_preprocessed_data(self, raw_data)
    
    def _compute_rewards(
        self, 
        ag_data: np.ndarray, 
        ag_rewards: np.ndarray,
        veh_edge_dist_rewards: np.ndarray, 
        veh_veh_dist_rewards: np.ndarray
    ) -> np.ndarray:
        """
        Compute comprehensive rewards
        
        Refer to dataset.py compute_rewards() method
        
        Reward dimensions:
        - 0: pos_target_achieved (0 or 1)
        - 1: heading_target_achieved (0 or 1)  
        - 2: speed_target_achieved (0 or 1)
        - 3: pos_goal_shaped
        - 4: veh_veh_dist
        - 5: veh_edge_dist
        """
        return _preprocessed.compute_rewards(
            self,
            ag_data,
            ag_rewards,
            veh_edge_dist_rewards,
            veh_veh_dist_rewards,
        )
    
    def get_available_scenario_ids(self) -> List[str]:
        """Get the list of scenario IDs with preprocessed data"""
        return _preprocessed.get_available_scenario_ids(self)
    
    def load_preprocessed_data_direct(
        self,
        scenario_filename: str
    ) -> Tuple[Optional[Dict], bool]:
        """
        Load preprocessed data directly from file (without dataset object)
        
        Args:
            scenario_filename: Scenario filename (without extension)
        
        Returns:
            preproc_data: Preprocessed data
            file_exists: Whether the file exists
        """
        return _preprocessed.load_preprocessed_data_direct(self, scenario_filename)
    
    def get_moving_vehicle_ids(self, scenario) -> List[int]:
        """
        Get the list of IDs of the moving vehicles in the scenario
        
        utils/sim.py get_moving_vehicles()
        
        Args:
            scenario: Nocturne scenario object
        
        Returns:
            moving_ids: List of IDs of the moving vehicles
        """
        return _scenario.get_moving_vehicle_ids(self, scenario)
    
    def get_road_data(self, scenario) -> List[Dict]:
        """
        Get the road data (utils/sim.py get_road_data())
        
        Args:
            scenario: Nocturne scenario object
        
        Returns:
            road_data: List of road data, each element contains:
                - 'geometry': List of geometry points [{'x': float, 'y': float}, ...]
                - 'type': Road type ('road_line', 'road_edge', 'lane', etc.)
        """
        return _scenario.get_road_data_for_scenario(self, scenario)
    
    def extract_road_edge_polylines(self, road_data: List[Dict]) -> List[np.ndarray]:
        """
        Extract the polylines from the road data
        
        Args:
            road_data: The road data returned by get_road_data()
        
        Returns:
            polylines: List of polylines, each element shape (N, 2)
        """
        return _scenario.extract_road_edge_polylines(self, road_data)
    
    def create_simulation(
        self,
        scenario_path: str,
        scenario_filename: str
    ):
        """
        Create a Nocturne simulation instance
        
        utils/sim.py get_sim()
        
        Args:
            scenario_path: The directory of the scenario file
            scenario_filename: The name of the scenario file
        
        Returns:
            sim: Nocturne Simulation object
        """
        return _scenario.create_simulation(self, scenario_path, scenario_filename)


class ScenarioDataLoader:
    """
    Scenario data loader: Simplify the batch loading of scenario data
    
    Used for batch processing of scenarios in the DCD environment
    """
    
    def __init__(
        self,
        cfg: Any,
        scenario_dir: str,
        preprocess_dir: str,
    ):
        """
        Args:
            cfg: Hydra configuration object
            scenario_dir: The directory of the scenario files
            preprocess_dir: The directory of the preprocessed data
        """
        self.cfg = cfg
        self.scenario_dir = scenario_dir
        self.preprocess_dir = preprocess_dir
        self.bridge = DataBridge(cfg, preprocess_dir)
    
    def load_scenario(
        self,
        scenario_id: str
    ) -> Tuple[Any, Dict, Optional[Dict], List[int]]:
        """
        Load all data of a single scenario (utils/sim.py get_sim())
        
        Args:
            scenario_id: The ID of the scenario (file name, without extension)
        
        Returns:
            sim: Nocturne Simulation object
            gt_data_dict: Ground truth data
            preproc_data: Preprocessed data (may be None)
            moving_ids: List of IDs of the moving vehicles
        """
        return _scenario.load_scenario(self, scenario_id)
    
    def get_scenario_list(self) -> List[str]:
        """
        Get the list of available scenarios
        
        Returns:
            scenario_ids: List of scenario IDs
        """
        return _scenario.get_scenario_list(self)
