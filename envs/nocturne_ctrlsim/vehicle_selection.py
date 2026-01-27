"""
Vehicle selection helpers for Nocturne CtrlSim environment.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np


class VehicleSelectionMixin:
    """Mixin providing vehicle selection logic (dynamic fallback only).
    
    Note: JSON-based vehicle ID loading is handled in adversarial.py's 
    _load_vehicle_ids_for_scenario() method. This mixin only provides
    the dynamic selection fallback logic.
    """

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
            (veh_id_1, veh_id_2) tuple. If no interesting pair is found, caller
            should fall back to dense vehicle selection.
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
    
    def _select_dense_vehicle(
        self,
        moving_veh_ids: List[int],
        k_neighbors: int = 7,
        traj_len_threshold: int = 30,
    ) -> Optional[int]:
        """Select vehicle with the smallest average distance to its nearest neighbors."""
        history_steps = getattr(self.cfg.nocturne, 'history_steps', 10)
        positions = {}
        for veh_id in moving_veh_ids:
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
            if veh_id not in self._gt_data_dict:
                continue
            gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj[:, 4]
            has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
            if not has_valid_traj:
                continue
            pos = veh.getPosition()
            positions[veh_id] = np.array([pos.x, pos.y], dtype=np.float32)

        if len(positions) == 0:
            return None
        if len(positions) == 1:
            return next(iter(positions.keys()))

        best_vid = None
        best_avg = None
        for vid, pos in positions.items():
            dists = []
            for other_id, other_pos in positions.items():
                if other_id == vid:
                    continue
                dists.append(np.linalg.norm(pos - other_pos))
            if len(dists) == 0:
                continue
            dists.sort()
            k = min(k_neighbors, len(dists))
            avg_dist = float(np.mean(dists[:k]))
            if best_avg is None or avg_dist < best_avg or (avg_dist == best_avg and vid < best_vid):
                best_avg = avg_dist
                best_vid = vid

        return best_vid

    def _select_ego_vehicle(self):
        """
        Select ego vehicle using dynamic fallback logic.
        
        Use find_interesting_pair logic to select two interesting vehicles,
        then deterministically select the vehicle with smaller veh_id as ego.
        
        If no interesting pair is found, select the dense vehicle (smallest
        average distance to neighbors). If dense selection fails, fall back
        to the first moving vehicle.
        
        Note: This is the fallback method. Primary vehicle ID loading from JSON
        is handled in adversarial.py's _load_vehicle_ids_for_scenario().
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
            # If no interesting pair is found, downgrade to dense vehicle selection
            print(
                f"Warning: No interesting vehicle pair found in scenario {self.current_level.scenario_id}. "
                f"Using dense vehicle selection for ego."
            )
            ego_veh_id = self._select_dense_vehicle(moving_veh_ids)
            if ego_veh_id is None:
                ego_veh_id = moving_veh_ids[0]
        else:
            # 3. Deterministic selection: select vehicle with smaller veh_id as ego
            ego_veh_id = min(interesting_pair)
        
        return self._get_vehicle_by_id(ego_veh_id)
    
    def _select_opponent_vehicles(self, k: int = 7, traj_len_threshold: int = 10):
        """
        Select opponent vehicles using dynamic fallback logic.
        
        Selects k nearest moving vehicles to ego with valid trajectory length.
        
        Note: This is the fallback method. Primary vehicle ID loading from JSON
        is handled in adversarial.py's _load_vehicle_ids_for_scenario().
        
        Args:
            k: Maximum number of opponent vehicles to select
            traj_len_threshold: Minimum trajectory length (default 10 steps)
        """
        if self.ego_vehicle is None:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # Dynamic selection: get moving vehicles (excluding ego)
        moving_veh_ids = self._get_moving_vehicle_ids()
        ego_id = self.ego_vehicle.getID()
        candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]
        
        if len(candidate_ids) == 0:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # 2. Calculate distance to ego, with trajectory length filter
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        history_steps = getattr(self.cfg.nocturne, 'history_steps', 10)
        
        distances = []
        for veh_id in candidate_ids:
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
            
            # Check trajectory length constraint
            if veh_id not in self._gt_data_dict:
                continue
            gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj[:, 4]
            has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
            if not has_valid_traj:
                continue
            
            pos = veh.getPosition()
            dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
            distances.append((dist, veh_id, veh))
        
        # 3. Sort by distance, select k nearest vehicles
        distances.sort(key=lambda x: x[0])
        selected = distances[:k]
        
        self.opponent_vehicles = [item[2] for item in selected]
        self.opponent_vehicle_ids = [item[1] for item in selected]
