"""
Data bridge tool

Bridge the DCD environment with the ctrl-sim data format

- utils/sim.py: get_ground_truth_states(), get_road_data(), get_moving_vehicles()
- evaluators/evaluator.py: load_preprocessed_data()
- evaluators/policy_evaluator.py  lines 427-460
"""
import os
import sys
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
    
    Example:
    ```python
    bridge = DataBridge(cfg, preprocess_dir)
    gt_data = bridge.get_ground_truth(scenario_path, filename)
    preproc_data, exists = bridge.load_preprocessed_data(filename)
    moving_ids = bridge.get_moving_vehicle_ids(scenario)
    ```
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
        
        # For loading preprocessed data
        self._preprocessed_dset: Optional[RLWaymoDatasetCtRLSim] = None
        self._preprocessed_files_cache: Optional[List[str]] = None
    
    def _ensure_preprocessed_dset(self):
        """Delay initialization of preprocessed dataset"""
        if self._preprocessed_dset is None:
            self._preprocessed_dset = RLWaymoDatasetCtRLSim(
                self.cfg, split_name='test', mode='eval'
            )
            self._preprocessed_files_cache = self._preprocessed_dset.files
    
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
        Load preprocessed data (contains RTG and road information)
        
        evaluators/evaluator.py lines 47-59 load_preprocessed_data()
        
        Args:
            scenario_filename: Scene file name (without extension, e.g. 'scenario_001')
        
        Returns:
            preproc_data: Preprocessed data, containing:
                - 'rtgs': shape (num_agents, steps, num_reward_components)
                - 'road_points': Road point information
                - 'road_types': Road type information
            file_exists: Whether the file exists
        """
        self._ensure_preprocessed_dset()
        
        # Construct the path of the preprocessed file
        filename = os.path.join(
            self.preprocess_dir, 
            f'{scenario_filename}_physics.pkl'
        )
        
        file_exists = filename in self._preprocessed_files_cache
        
        if file_exists:
            idx = self._preprocessed_files_cache.index(filename)
            preproc_data = self._preprocessed_dset[idx]
        else:
            preproc_data = None
        
        return preproc_data, file_exists
    
    def load_preprocessed_data_direct(
        self,
        scenario_filename: str
    ) -> Tuple[Optional[Dict], bool]:
        """
        Load preprocessed data directly from the file (without depending on the dataset object)
        
        Args:
            scenario_filename: Scene file name (without extension)
        
        Returns:
            preproc_data: Preprocessed data
            file_exists: Whether the file exists
        """
        filename = os.path.join(
            self.preprocess_dir,
            f'{scenario_filename}_physics.pkl'
        )
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                preproc_data = pickle.load(f)
            return preproc_data, True
        else:
            return None, False
    
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
