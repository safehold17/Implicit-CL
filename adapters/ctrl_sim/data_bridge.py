"""
Data bridge tool

Bridge the DCD environment with the ctrl-sim data format

- utils/sim.py: get_ground_truth_states(), get_road_data(), get_moving_vehicles()
- evaluators/evaluator.py: load_preprocessed_data()
- evaluators/policy_evaluator.py  lines 427-460
"""
import os
import sys
import glob
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from utils.sim import get_ground_truth_states, get_road_data, get_moving_vehicles, get_sim
from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim


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
    
    def __init__(self, cfg: Any, preprocess_dir: str):
        """
        Args:
            cfg: Hydra
            preprocess_dir
        """
        self.cfg = cfg
        self.preprocess_dir = preprocess_dir
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.cfg_dataset = cfg.dataset.waymo

        # ctrl-sim preprocessed dataset (align evaluator behavior)
        self._preprocess_split = 'test'
        self._preprocess_root: Optional[str] = None
        self.preprocessed_dset: Optional[RLWaymoDatasetCtRLSim] = None
        if preprocess_dir:
            preprocess_root = preprocess_dir
            if os.path.basename(os.path.normpath(preprocess_dir)) == self._preprocess_split:
                preprocess_root = os.path.dirname(preprocess_dir)
            self._preprocess_root = preprocess_root
            try:
                self.cfg.dataset.waymo.preprocess_dir = preprocess_root
            except Exception:
                pass
            self.preprocessed_dset = RLWaymoDatasetCtRLSim(cfg, split_name=self._preprocess_split, mode='eval')
        
        # Preprocessed files cache
        self._preprocessed_files_cache: Optional[Dict[str, str]] = None
        self._preproc_data_cache: Dict[str, Dict] = {}
    
    def _ensure_preprocessed_files_cache(self):
        """Lazy initialize preprocessed files cache, mapping scenario_id -> file_path"""
        if self._preprocessed_files_cache is None:
            self._preprocessed_files_cache = {}
            if self.preprocessed_dset is None:
                return
            for filepath in self.preprocessed_dset.files:
                basename = os.path.basename(filepath)
                if basename.endswith('_physics.pkl'):
                    scenario_id = basename.replace('_physics.pkl', '')
                    self._preprocessed_files_cache[scenario_id] = filepath
    
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
        # Construct file list (compatible with ctrl-sim interface)
        files = [scenario_filename]
        file_id = 0
        
        return get_ground_truth_states(
            self.cfg,
            scenario_path,
            files,
            file_id,
            self.dt,
            self.steps
        )
    
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
        from utils.data import get_agent_type_onehot
        
        def get_state(veh):
            pos = veh.getPosition()
            heading = veh.getHeading()
            target = veh.getGoalPosition()
            speed = veh.getSpeed()
            agent_type = get_agent_type_onehot(veh.getType().value)
            existence = 1 if pos.x != -10000 else 0
            length = veh.getLength()
            
            veh_state = [pos.x, pos.y, heading, speed, existence, target.x, target.y, length]
            return veh_state, agent_type
        
        scenario = sim.getScenario()
        vehicles = scenario.vehicles()
        state_dict = {veh.getID(): {"traj": [], "type": None} for veh in vehicles}
        
        # Save the current state
        for veh in vehicles:
            veh.expert_control = True
        
        # Collect the state of all time steps
        for s in range(self.steps):
            for veh in vehicles:
                veh_state, veh_type = get_state(veh)
                state_dict[veh.getID()]["traj"].append(veh_state)
                state_dict[veh.getID()]["type"] = veh_type
            sim.step(self.dt)
        
        # The last state
        for veh in vehicles:
            veh_state, veh_type = get_state(veh)
            state_dict[veh.getID()]["traj"].append(veh_state)
            state_dict[veh.getID()]["type"] = veh_type
        
        # Reset the simulator
        sim.reset()
        
        return state_dict
    
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
        if self.preprocessed_dset is None or self._preprocess_root is None:
            return None, False
        
        scenario_id = os.path.splitext(scenario_filename)[0]
        self._ensure_preprocessed_files_cache()
        
        # Check cache
        if scenario_id in self._preproc_data_cache:
            return self._preproc_data_cache[scenario_id], True
        
        # Look for preprocessed file
        if scenario_id not in self._preprocessed_files_cache:
            return None, False
        
        filepath = self._preprocessed_files_cache[scenario_id]
        
        try:
            # align with ctrl-sim evaluator: use dataset indexing to load preprocessed data
            idx = self.preprocessed_dset.files.index(filepath)
            preproc_data = self.preprocessed_dset[idx]
            
            # cache results
            self._preproc_data_cache[scenario_id] = preproc_data
            
            return preproc_data, True
            
        except Exception as e:
            print(f"Warning: Failed to load preprocessed data for {scenario_id}: {e}")
            return None, False
    
    def _process_preprocessed_data(self, raw_data: Dict) -> Dict:
        """
        Process original preprocessed data, calculate RTG
        
        Refer to dataset_ctrl_sim.py get_data() method (mode='eval' branch)
        """
        ag_data = raw_data['ag_data']
        ag_rewards = raw_data['ag_rewards']
        veh_edge_dist_rewards = raw_data['veh_edge_dist_rewards']
        veh_veh_dist_rewards = raw_data['veh_veh_dist_rewards']
        road_points = raw_data['road_points']
        road_types = raw_data['road_types']
        
        # compute comprehensive rewards (refer to dataset.py compute_rewards)
        all_rewards = self._compute_rewards(
            ag_data, ag_rewards, 
            veh_edge_dist_rewards, veh_veh_dist_rewards
        )
        
        # calculate RTG (Return-To-Go): accumulated future rewards
        # shape: (num_agents, steps+1, num_reward_components)
        rtgs = np.cumsum(all_rewards[:, ::-1], axis=1)[:, ::-1]
        
        return {
            'rtgs': rtgs,
            'road_points': road_points,
            'road_types': road_types,
            # keep original data for later use
            'ag_data': ag_data,
            'ag_rewards': ag_rewards,
            'filtered_ag_ids': raw_data.get('filtered_ag_ids', []),
        }
    
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
        cfg = self.cfg_dataset
        
        # ag_rewards shape: (num_agents, steps+1, 6)
        # contains: pos_target, heading_target, speed_target, pos_shaped, speed_shaped, heading_shaped
        
        # build complete reward array
        num_agents, num_steps, _ = ag_rewards.shape
        all_rewards = np.zeros((num_agents, num_steps, 6), dtype=np.float32)
        
        # copy base rewards
        all_rewards[:, :, :3] = ag_rewards[:, :, :3]  # target achieved
        all_rewards[:, :, 3] = ag_rewards[:, :, 3]    # pos_goal_shaped
        
        # add distance rewards (refer to dataset.py lines 267-280)
        all_rewards[:, :, 4] = veh_veh_dist_rewards * cfg.veh_veh_collision_rew_multiplier
        all_rewards[:, :, 5] = veh_edge_dist_rewards * cfg.veh_edge_collision_rew_multiplier
        
        return all_rewards
    
    def get_available_scenario_ids(self) -> List[str]:
        """Get the list of scenario IDs with preprocessed data"""
        self._ensure_preprocessed_files_cache()
        return list(self._preprocessed_files_cache.keys())
    
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
        # use new load_preprocessed_data method
        return self.load_preprocessed_data(scenario_filename)
    
    def get_moving_vehicle_ids(self, scenario) -> List[int]:
        """
        Get the list of IDs of the moving vehicles in the scenario
        
        utils/sim.py get_moving_vehicles()
        
        Args:
            scenario: Nocturne scenario object
        
        Returns:
            moving_ids: List of IDs of the moving vehicles
        """
        return get_moving_vehicles(scenario)
    
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
        return get_road_data(scenario)
    
    def extract_road_edge_polylines(self, road_data: List[Dict]) -> List[np.ndarray]:
        """
        Extract the polylines from the road data
        
        Args:
            road_data: The road data returned by get_road_data()
        
        Returns:
            polylines: List of polylines, each element shape (N, 2)
        """
        road_edge_polylines = []
        for road in road_data:
            if road['type'] == 'road_edge':
                geometry = road['geometry']
                if isinstance(geometry, list):
                    polyline = np.array([[pt['x'], pt['y']] for pt in geometry])
                    road_edge_polylines.append(polyline)
        return road_edge_polylines
    
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
        files = [scenario_filename]
        file_id = 0
        return get_sim(self.cfg, scenario_path, files, file_id)


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
        scenario_filename = f"{scenario_id}.json"
        
        # Create a simulation
        sim = self.bridge.create_simulation(
            self.scenario_dir, 
            scenario_filename
        )
        scenario = sim.getScenario()
        
        # Get the ground truth
        gt_data_dict = self.bridge.get_ground_truth(
            self.scenario_dir,
            scenario_filename
        )
        
        # Load the preprocessed data
        preproc_data, _ = self.bridge.load_preprocessed_data(scenario_id)
        
        # Get the moving vehicle IDs
        moving_ids = self.bridge.get_moving_vehicle_ids(scenario)
        
        return sim, gt_data_dict, preproc_data, moving_ids
    
    def get_scenario_list(self) -> List[str]:
        """
        Get the list of available scenarios
        
        Returns:
            scenario_ids: List of scenario IDs
        """
        import glob
        files = glob.glob(os.path.join(self.scenario_dir, "*.json"))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]
