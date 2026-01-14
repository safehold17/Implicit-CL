"""
Find Interesting Pairs in Nocturne Scenarios

This script scans JSON scenario files and identifies which scenarios contain
"interesting pairs" of vehicles using the same logic as adversarial.py.

Usage:
    python find_interesting_pairs.py --data_dir data/nocturne_waymo/formatted_json_v2_no_tl_valid
    
Optional arguments:
    --goal_dist_threshold: Maximum goal distance (default: 10.0 meters)
    --timestep_diff_threshold: Maximum timestep difference (default: 20 steps)
    --traj_len_threshold: Minimum trajectory length (default: 60 steps)
    --save_results: Save results to JSON file (default: results_interesting_pairs.json)
"""

import os
import sys
import glob
import json
import argparse
import pickle
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_party', 'ctrl-sim'))

from adapters.ctrl_sim.data_bridge import DataBridge


class InterestingPairFinder:
    """
    Find interesting pairs in Nocturne scenarios using the same logic as adversarial.py
    """
    
    def __init__(
        self,
        cfg,
        goal_dist_threshold: float = 20.0,
        timestep_diff_threshold: int = 40,
        traj_len_threshold: int = 30,
        history_steps: int = 10,
        max_episode_steps: int = 90
    ):
        """
        Args:
            cfg: Hydra configuration object
            goal_dist_threshold: Maximum distance between goal positions (meters)
            timestep_diff_threshold: Maximum difference in goal timesteps
            traj_len_threshold: Minimum trajectory length
            history_steps: Number of history steps
            max_episode_steps: Maximum episode steps
        """
        self.cfg = cfg
        self.goal_dist_threshold = goal_dist_threshold
        self.timestep_diff_threshold = timestep_diff_threshold
        self.traj_len_threshold = traj_len_threshold
        self.history_steps = history_steps
        self.max_episode_steps = max_episode_steps
        
        # Initialize data bridge
        # Use a dummy preprocess_dir since we only need ground truth data
        self.data_bridge = DataBridge(cfg, preprocess_dir="")
    
    def _get_moving_vehicle_ids(self, scenario) -> List[int]:
        """Get moving vehicle IDs from scenario"""
        return [v.getID() for v in scenario.getObjectsThatMoved()]
    
    def _find_interesting_pair(
        self, 
        moving_veh_ids: List[int],
        gt_data_dict: Dict,
        vehicles: List
    ) -> Optional[Tuple[int, int]]:
        """
        Find an interesting pair of vehicles
        
        Same logic as adversarial.py _find_interesting_pair()
        
        Filtering criteria:
        - Goal positions are close (< goal_dist_threshold meters)
        - Goal timesteps are close (< timestep_diff_threshold steps)
        - Trajectories are long enough (>= traj_len_threshold steps)
        
        Args:
            moving_veh_ids: List of moving vehicle IDs
            gt_data_dict: Ground truth data dictionary
            vehicles: List of vehicle objects
        
        Returns:
            (veh_id_1, veh_id_2) tuple if found, None otherwise
        """
        goals = []
        goal_timesteps = []
        valid_traj_mask = []
        veh_ids = []
        
        # Build a vehicle ID to vehicle object mapping
        veh_dict = {v.getID(): v for v in vehicles}
        
        for veh_id in moving_veh_ids:
            if veh_id not in gt_data_dict:
                continue
            
            gt_traj = np.array(gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj[:, 4]
            
            # Calculate goal position and timestep
            idx_goal = self.max_episode_steps - 1
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
            
            veh = veh_dict.get(veh_id)
            if veh is None:
                continue
            
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj[idx_goal, :2]
            
            # Check trajectory length
            has_valid_traj = existence_mask[self.history_steps:].sum() >= self.traj_len_threshold
            
            goals.append(goal_pos)
            goal_timesteps.append(idx_goal - self.history_steps)
            valid_traj_mask.append(1 if has_valid_traj else 0)
            veh_ids.append(veh_id)
        
        if len(goals) < 2:
            return None
        
        goals = np.array(goals)
        goal_timesteps = np.array(goal_timesteps)
        valid_traj_mask = np.array(valid_traj_mask)
        
        # Calculate goal distance matrix
        dists = np.linalg.norm(goals[:, np.newaxis] - goals[np.newaxis, :], axis=-1)
        
        # Build masks
        nearby_mask = dists < self.goal_dist_threshold
        not_same_mask = dists > 0
        valid_traj_both = np.outer(valid_traj_mask, valid_traj_mask)
        timestep_diff = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
        within_time_mask = timestep_diff < self.timestep_diff_threshold
        
        goal_mask = nearby_mask & not_same_mask & valid_traj_both.astype(bool) & within_time_mask
        
        indices = np.where(goal_mask)
        valid_pairs = list(zip(indices[0], indices[1]))
        
        if len(valid_pairs) == 0:
            return None
        
        # Deterministic selection: choose first pair
        pair_idx = valid_pairs[0]
        return (veh_ids[pair_idx[0]], veh_ids[pair_idx[1]])
    
    def check_scenario(
        self, 
        scenario_dir: str, 
        scenario_filename: str
    ) -> Tuple[bool, Optional[Tuple[int, int]], int]:
        """
        Check if a scenario contains an interesting pair
        
        Args:
            scenario_dir: Directory containing scenario files
            scenario_filename: Scenario filename (with .json extension)
        
        Returns:
            (has_pair, pair_ids, num_moving_vehicles)
        """
        try:
            # Get ground truth data
            gt_data_dict = self.data_bridge.get_ground_truth(
                scenario_dir,
                scenario_filename
            )
            
            # Create simulation to get scenario and vehicles
            sim = self.data_bridge.create_simulation(scenario_dir, scenario_filename)
            scenario = sim.getScenario()
            vehicles = list(scenario.vehicles())
            
            # Get moving vehicles
            moving_veh_ids = self._get_moving_vehicle_ids(scenario)
            
            if len(moving_veh_ids) < 2:
                return False, None, len(moving_veh_ids)
            
            # Find interesting pair
            interesting_pair = self._find_interesting_pair(
                moving_veh_ids, 
                gt_data_dict,
                vehicles
            )
            
            # Clean up
            sim.reset()
            
            has_pair = interesting_pair is not None
            return has_pair, interesting_pair, len(moving_veh_ids)
            
        except Exception as e:
            print(f"Error processing {scenario_filename}: {e}")
            return False, None, 0
    
    def scan_directory(
        self, 
        data_dir: str,
        max_scenarios: Optional[int] = None
    ) -> Dict:
        """
        Scan all scenarios in a directory
        
        Args:
            data_dir: Directory containing JSON scenario files
            max_scenarios: Maximum number of scenarios to process (None = all)
        
        Returns:
            Dictionary with results
        """
        # Find all JSON files
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        if len(json_files) == 0:
            print(f"No JSON files found in {data_dir}")
            return {}
        
        print(f"Found {len(json_files)} scenario files")
        
        if max_scenarios is not None:
            json_files = json_files[:max_scenarios]
            print(f"Processing first {max_scenarios} scenarios")
        
        # Process scenarios
        results = {
            'scenarios_with_pair': [],
            'scenarios_without_pair': [],
            'pair_details': {},
            'statistics': {
                'total': len(json_files),
                'with_pair': 0,
                'without_pair': 0,
                'error': 0
            }
        }
        
        for json_file in tqdm(json_files, desc="Scanning scenarios"):
            scenario_filename = os.path.basename(json_file)
            scenario_id = scenario_filename.replace('.json', '')
            
            has_pair, pair_ids, num_moving = self.check_scenario(
                data_dir, 
                scenario_filename
            )
            
            if has_pair:
                results['scenarios_with_pair'].append(scenario_id)
                results['pair_details'][scenario_id] = {
                    'pair': pair_ids,
                    'num_moving_vehicles': num_moving
                }
                results['statistics']['with_pair'] += 1
            else:
                results['scenarios_without_pair'].append(scenario_id)
                results['statistics']['without_pair'] += 1
        
        return results


def create_minimal_config():
    """Create minimal configuration for DataBridge"""
    config = {
        'nocturne': {
            'dt': 0.1,
            'steps': 90,
            'scenario': {
                'start_time': 0,
                'allow_non_vehicles': False,
            }
        },
        'dataset': {
            'waymo': {
                'veh_veh_collision_rew_multiplier': 1.0,
                'veh_edge_collision_rew_multiplier': 1.0,
            }
        }
    }
    return OmegaConf.create(config)


def main():
    parser = argparse.ArgumentParser(
        description="Find interesting pairs in Nocturne scenarios"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/nocturne_waymo/formatted_json_v2_no_tl_valid',
        help='Directory containing JSON scenario files'
    )
    parser.add_argument(
        '--goal_dist_threshold',
        type=float,
        default=10.0,
        help='Maximum goal distance threshold (meters)'
    )
    parser.add_argument(
        '--timestep_diff_threshold',
        type=int,
        default=20,
        help='Maximum timestep difference threshold'
    )
    parser.add_argument(
        '--traj_len_threshold',
        type=int,
        default=60,
        help='Minimum trajectory length threshold'
    )
    parser.add_argument(
        '--max_scenarios',
        type=int,
        default=None,
        help='Maximum number of scenarios to process (default: all)'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default='results_interesting_pairs.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Finding Interesting Pairs in Nocturne Scenarios")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Goal distance threshold: {args.goal_dist_threshold} meters")
    print(f"Timestep diff threshold: {args.timestep_diff_threshold} steps")
    print(f"Trajectory length threshold: {args.traj_len_threshold} steps")
    print("=" * 80)
    print()
    
    # Create configuration
    cfg = create_minimal_config()
    
    # Create finder
    finder = InterestingPairFinder(
        cfg,
        goal_dist_threshold=args.goal_dist_threshold,
        timestep_diff_threshold=args.timestep_diff_threshold,
        traj_len_threshold=args.traj_len_threshold
    )
    
    # Scan directory
    results = finder.scan_directory(args.data_dir, args.max_scenarios)
    
    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total scenarios: {results['statistics']['total']}")
    print(f"Scenarios WITH interesting pair: {results['statistics']['with_pair']} "
          f"({results['statistics']['with_pair'] / results['statistics']['total'] * 100:.1f}%)")
    print(f"Scenarios WITHOUT interesting pair: {results['statistics']['without_pair']} "
          f"({results['statistics']['without_pair'] / results['statistics']['total'] * 100:.1f}%)")
    print()
    
    # Show some examples
    if results['scenarios_with_pair']:
        print("Examples of scenarios WITH interesting pairs:")
        for scenario_id in results['scenarios_with_pair'][:5]:
            pair = results['pair_details'][scenario_id]['pair']
            num_moving = results['pair_details'][scenario_id]['num_moving_vehicles']
            print(f"  - {scenario_id}: pair={pair}, num_moving={num_moving}")
        if len(results['scenarios_with_pair']) > 5:
            print(f"  ... and {len(results['scenarios_with_pair']) - 5} more")
        print()
    
    if results['scenarios_without_pair']:
        print("Examples of scenarios WITHOUT interesting pairs:")
        for scenario_id in results['scenarios_without_pair'][:5]:
            print(f"  - {scenario_id}")
        if len(results['scenarios_without_pair']) > 5:
            print(f"  ... and {len(results['scenarios_without_pair']) - 5} more")
        print()
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.save_results}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
