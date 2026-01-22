#!/usr/bin/env python3
"""
Build ego vehicle map for scenarios listed in scenarios_index.json.

Output format:
    {
        "scenario_id": ego_vehicle_id_or_null,
        ...
    }
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from adapters.ctrl_sim import DataBridge, create_minimal_config


def _load_config_defaults(config_path: str) -> Optional[str]:
    """Load scenario_dir from config.yaml."""
    if not config_path or not os.path.exists(config_path):
        return None
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    scenario_dir = OmegaConf.select(cfg, "nocturne_env.scenario_data_dir")
    return scenario_dir


def _load_scenario_index(index_path: str) -> Tuple[List[str], Optional[str]]:
    """Load scenario_ids list and source_dir from scenarios_index.json."""
    with open(index_path, "r") as f:
        data = json.load(f)
    scenario_ids = data.get("scenario_ids")
    source_dir = data.get("source_dir")
    if not isinstance(scenario_ids, list):
        raise ValueError("scenarios_index.json missing 'scenario_ids' list")
    return scenario_ids, source_dir


def _find_interesting_pair(
    moving_veh_ids: List[int],
    gt_data_dict: Dict,
    vehicles: List,
    max_episode_steps: int,
    history_steps: int,
    goal_dist_threshold: float = 10.0,
    timestep_diff_threshold: int = 20,
    traj_len_threshold: int = 60,
) -> Optional[Tuple[int, int]]:
    """Find an interesting pair using the same logic as VehicleSelectionMixin."""
    goals = []
    goal_timesteps = []
    valid_traj_mask = []
    veh_ids = []

    veh_dict = {v.getID(): v for v in vehicles}

    for veh_id in moving_veh_ids:
        if veh_id not in gt_data_dict:
            continue

        gt_traj = np.array(gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]

        idx_goal = max_episode_steps - 1
        idx_disappear = np.where(existence_mask == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1

        veh = veh_dict.get(veh_id)
        if veh is None:
            continue

        goal_pos = np.array([veh.target_position.x, veh.target_position.y])
        if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
            goal_pos = gt_traj[idx_goal, :2]

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

    dists = np.linalg.norm(goals[:, np.newaxis] - goals[np.newaxis, :], axis=-1)

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

    pair_idx = valid_pairs[0]
    return (veh_ids[pair_idx[0]], veh_ids[pair_idx[1]])


def _select_dense_vehicle(
    moving_veh_ids: List[int],
    vehicles: List,
    k_neighbors: int = 7,
) -> Optional[int]:
    """Select vehicle with smallest average distance to its nearest neighbors."""
    veh_dict = {v.getID(): v for v in vehicles}
    positions = {}
    for veh_id in moving_veh_ids:
        veh = veh_dict.get(veh_id)
        if veh is None:
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


def _select_ego_vehicle_id(
    moving_veh_ids: List[int],
    gt_data_dict: Dict,
    vehicles: List,
    max_episode_steps: int,
    history_steps: int,
) -> Optional[int]:
    """Select ego vehicle id using interesting pair or dense fallback."""
    if len(moving_veh_ids) == 0:
        return None
    if len(moving_veh_ids) == 1:
        return moving_veh_ids[0]

    interesting_pair = _find_interesting_pair(
        moving_veh_ids,
        gt_data_dict,
        vehicles,
        max_episode_steps,
        history_steps,
    )
    if interesting_pair is not None:
        return min(interesting_pair)

    return _select_dense_vehicle(moving_veh_ids, vehicles, k_neighbors=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ego vehicle map for scenarios")
    parser.add_argument(
        "--scenario_index_json",
        type=str,
        default="/home/chen/workspace/dcd-ctrlsim/data/scenarios_index_valid.json",
        help="Path to scenarios_index.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/chen/workspace/dcd-ctrlsim/cfgs/config.yaml",
        help="Path to config.yaml for defaults",
    )
    parser.add_argument(
        "--scenario_dir",
        type=str,
        default="/home/chen/workspace/dcd-ctrlsim/data/nocturne_waymo/formatted_json_v2_no_tl_valid",
        help="Scenario directory (overrides config.yaml)",
    )
    args = parser.parse_args()

    cfg_scenario_dir = _load_config_defaults(args.config)
    scenario_ids, index_source_dir = _load_scenario_index(args.scenario_index_json)
    if args.scenario_dir is None:
        args.scenario_dir = index_source_dir or cfg_scenario_dir

    if not args.scenario_dir:
        raise ValueError("scenario_dir is required (set via --scenario_dir or config.yaml).")
    if not os.path.exists(args.scenario_index_json):
        raise FileNotFoundError(
            f"scenarios_index.json not found: {args.scenario_index_json}"
        )
    if len(scenario_ids) == 0:
        raise ValueError("No scenario_ids found in scenarios_index.json")

    cfg = create_minimal_config(
        checkpoint_path="",
        scenario_dir=args.scenario_dir,
        preprocess_dir=None,
    )
    data_bridge = DataBridge(cfg, preprocess_dir="")
    history_steps = int(getattr(cfg.nocturne, "history_steps", 10))
    max_episode_steps = 90

    ego_map: Dict[str, Optional[int]] = {}
    for scenario_id in scenario_ids:
        scenario_filename = f"{scenario_id}.json"
        scenario_path = os.path.join(args.scenario_dir, scenario_filename)
        if not os.path.exists(scenario_path):
            print(f"Warning: scenario not found: {scenario_path}")
            ego_map[scenario_id] = None
            continue

        try:
            gt_data_dict = data_bridge.get_ground_truth(args.scenario_dir, scenario_filename)
            sim = data_bridge.create_simulation(args.scenario_dir, scenario_filename)
            scenario = sim.getScenario()
            vehicles = list(scenario.vehicles())
            moving_veh_ids = [v.getID() for v in scenario.getObjectsThatMoved()]

            ego_id = _select_ego_vehicle_id(
                moving_veh_ids,
                gt_data_dict,
                vehicles,
                max_episode_steps,
                history_steps,
            )
            ego_map[scenario_id] = ego_id
            sim.reset()
        except Exception as e:
            print(f"Warning: failed to process {scenario_id}: {e}")
            ego_map[scenario_id] = None

    output_path = os.path.join("data", "ego_vehicle_valid.json")
    if not os.path.isdir("data"):
        raise FileNotFoundError("data directory not found for output")
    with open(output_path, "w") as f:
        json.dump(ego_map, f, indent=2)
    print(f"Saved ego vehicle map to: {output_path}")


if __name__ == "__main__":
    main()
