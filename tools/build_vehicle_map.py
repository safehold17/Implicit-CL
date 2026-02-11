#!/usr/bin/env python3
"""
Build vehicle map for scenarios listed in scenarios_index.json.

Output format:
    {
        "scenario_id": {
            "ego_vehicle_id": int or null,
            "ego_selection_mode": "interesting" | "dense" | "unknown",
            "opponent_vehicle_ids": [int, ...] or []
        },
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
from tqdm import tqdm

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
    return veh_ids[pair_idx[0]], veh_ids[pair_idx[1]]


def _select_dense_vehicle(
    moving_veh_ids: List[int],
    vehicles: List,
    gt_data_dict: Dict,
    history_steps: int,
    k_neighbors: int = 7,
    traj_len_threshold: int = 60,
) -> Optional[int]:
    """Select vehicle with the smallest average distance to its nearest neighbors.
    
    Only considers vehicles with valid trajectory length >= traj_len_threshold.
    """
    veh_dict = {v.getID(): v for v in vehicles}
    positions = {}
    for veh_id in moving_veh_ids:
        veh = veh_dict.get(veh_id)
        if veh is None:
            continue
        
        # Check trajectory length constraint (same as _find_interesting_pair)
        if veh_id not in gt_data_dict:
            continue
        gt_traj = np.array(gt_data_dict[veh_id]["traj"])
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


def _get_preproc_vehicle_ids(preproc_data, gt_data_dict: Dict) -> Optional[List[int]]:
    """
    从预处理数据中获取车辆ID列表
    
    Args:
        preproc_data: 预处理数据（dict 或对象）
        gt_data_dict: Ground truth 数据字典，用于推断车辆ID
        
    Returns:
        车辆ID列表，如果无法确定则返回None
    """
    if preproc_data is None:
        return None
    
    # 方法1: 尝试从 filtered_ag_ids 获取（最准确）
    filtered_ids = None
    if isinstance(preproc_data, dict):
        filtered_ids = preproc_data.get('filtered_ag_ids')
    else:
        filtered_ids = getattr(preproc_data, 'filtered_ag_ids', None)
    
    if filtered_ids is not None and len(filtered_ids) > 0:
        return list(filtered_ids)
    
    # 方法2: 从 RTG shape 推断 + 使用 gt_data_dict 获取实际ID
    rtgs = None
    if isinstance(preproc_data, dict):
        rtgs = preproc_data.get('rtgs')
    else:
        rtgs = getattr(preproc_data, 'rtgs', None)
    
    if rtgs is not None and gt_data_dict:
        import torch
        if isinstance(rtgs, torch.Tensor):
            rtgs = rtgs.cpu().numpy()
        
        # RTG shape: (num_agents, steps, reward_components)
        if hasattr(rtgs, 'shape') and len(rtgs.shape) >= 1:
            num_agents_in_rtg = rtgs.shape[0]
            
            # 从 gt_data_dict 获取所有车辆ID，取前 num_agents_in_rtg 个
            # 假设预处理数据按照 ID 顺序排列
            all_veh_ids = sorted(list(gt_data_dict.keys()))
            if len(all_veh_ids) >= num_agents_in_rtg:
                return all_veh_ids[:num_agents_in_rtg]
    
    return None


def _select_ego_vehicle_id(
    moving_veh_ids: List[int],
    gt_data_dict: Dict,
    vehicles: List,
    max_episode_steps: int,
    history_steps: int,
) -> Tuple[Optional[int], str]:
    """Select ego vehicle id using interesting pair or dense fallback."""
    if len(moving_veh_ids) == 0:
        return None, "unknown"
    if len(moving_veh_ids) == 1:
        return moving_veh_ids[0], "dense"

    interesting_pair = _find_interesting_pair(
        moving_veh_ids,
        gt_data_dict,
        vehicles,
        max_episode_steps,
        history_steps,
    )
    if interesting_pair is not None:
        return min(interesting_pair), "interesting"

    ego_id = _select_dense_vehicle(
        moving_veh_ids,
        vehicles,
        gt_data_dict,
        history_steps,
        k_neighbors=7,
    )
    if ego_id is None:
        return None, "unknown"
    return ego_id, "dense"


def _select_opponent_vehicle_ids(
    moving_veh_ids: List[int],
    vehicles: List,
    gt_data_dict: Dict,
    ego_id: Optional[int],
    history_steps: int,
    k: int = 7,
    traj_len_threshold: int = 60,
) -> List[int]:
    """Select opponent vehicle ids (k nearest moving vehicles to ego).
    
    Only considers vehicles with valid trajectory length >= traj_len_threshold.
    
    Args:
        moving_veh_ids: List of moving vehicle IDs
        vehicles: List of vehicle objects
        gt_data_dict: Ground truth data dictionary
        ego_id: Ego vehicle ID (to exclude from selection)
        history_steps: Number of history steps
        k: Maximum number of opponents to select
        traj_len_threshold: Minimum trajectory length (default 10)
        
    Returns:
        List of selected opponent vehicle IDs, sorted by distance to ego
    """
    if ego_id is None:
        return []
    
    veh_dict = {v.getID(): v for v in vehicles}
    ego_veh = veh_dict.get(ego_id)
    if ego_veh is None:
        return []
    
    ego_pos = ego_veh.getPosition()
    ego_pos = np.array([ego_pos.x, ego_pos.y], dtype=np.float32)
    
    # Filter candidates (excluding ego)
    candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]
    
    if len(candidate_ids) == 0:
        return []
    
    distances = []
    for veh_id in candidate_ids:
        veh = veh_dict.get(veh_id)
        if veh is None:
            continue
        
        # Check trajectory length constraint
        if veh_id not in gt_data_dict:
            continue
        gt_traj = np.array(gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]
        has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
        if not has_valid_traj:
            continue
        
        pos = veh.getPosition()
        dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
        distances.append((dist, veh_id))
    
    # Sort by distance, select k nearest
    distances.sort(key=lambda x: x[0])
    selected = distances[:k]
    
    return [item[1] for item in selected]


def _resolve_vehicle_goal_position(
    veh_id: int,
    veh,
    gt_data_dict: Dict,
    max_episode_steps: int,
) -> Optional[np.ndarray]:
    """
    Resolve vehicle goal position using the same fallback logic as runtime env:
    use target_position by default, and if GT indicates disappearance, use the
    last valid GT point when it differs from target.
    """
    if veh is None:
        return None

    goal_pos = np.array([veh.target_position.x, veh.target_position.y], dtype=np.float32)
    if veh_id not in gt_data_dict:
        return goal_pos

    gt_traj = np.array(gt_data_dict[veh_id].get("traj", []))
    if gt_traj.ndim != 2 or gt_traj.shape[0] == 0 or gt_traj.shape[1] < 5:
        return goal_pos

    existence_mask = gt_traj[:, 4]
    idx_goal = min(max_episode_steps - 1, gt_traj.shape[0] - 1)
    idx_disappear = np.where(existence_mask == 0)[0]
    has_disappear = len(idx_disappear) > 0
    if has_disappear:
        idx_goal = idx_disappear[0] - 1

    if has_disappear and idx_goal >= 0:
        gt_goal_pos = gt_traj[idx_goal, :2]
        if np.linalg.norm(gt_goal_pos - goal_pos) > 0.0:
            goal_pos = gt_goal_pos.astype(np.float32)

    return goal_pos


def _compute_start_goal_distance(
    ego_id: Optional[int],
    vehicles: List,
    gt_data_dict: Dict,
    max_episode_steps: int,
) -> Optional[float]:
    """Compute ego start->goal distance in meters."""
    if ego_id is None:
        return None

    veh_dict = {v.getID(): v for v in vehicles}
    ego_veh = veh_dict.get(ego_id)
    if ego_veh is None:
        return None

    start = ego_veh.getPosition()
    start_pos = np.array([start.x, start.y], dtype=np.float32)
    goal_pos = _resolve_vehicle_goal_position(
        ego_id,
        ego_veh,
        gt_data_dict,
        max_episode_steps=max_episode_steps,
    )
    if goal_pos is None:
        return None

    return float(np.linalg.norm(goal_pos - start_pos))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vehicle map for scenarios")
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
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default=None,
        help="Preprocessed data directory (required to filter vehicles by preprocessed data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/vehicle_map_valid.json or vehicle_map_train.json)",
    )
    parser.add_argument(
        "--filter_close_goal_point",
        action="store_true",
        help="Filter scenarios where ego start-goal distance is smaller than threshold.",
    )
    parser.add_argument(
        "--close_goal_threshold_m",
        type=float,
        default=5.0,
        help="Threshold in meters for --filter_close_goal_point (default: 5.0).",
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
        preprocess_dir=args.preprocess_dir,
    )
    data_bridge = DataBridge(cfg, preprocess_dir=args.preprocess_dir or "")
    history_steps = int(getattr(cfg.nocturne, "history_steps", 10))
    max_episode_steps = 90

    # 统计信息
    total_scenarios = 0
    scenarios_with_preproc = 0
    scenarios_filtered = 0
    close_goal_filtered_scenarios = 0
    
    vehicle_map: Dict[str, Dict] = {}
    kept_scenario_ids: List[str] = []
    for scenario_id in tqdm(
        scenario_ids,
        desc="Building vehicle map",
        unit="scenario",
        dynamic_ncols=True,
    ):
        total_scenarios += 1
        scenario_filename = f"{scenario_id}.json"
        scenario_path = os.path.join(args.scenario_dir, scenario_filename)
        if not os.path.exists(scenario_path):
            print(f"Warning: scenario not found: {scenario_path}")
            vehicle_map[scenario_id] = {
                "ego_vehicle_id": None,
                "ego_selection_mode": "unknown",
                "opponent_vehicle_ids": [],
            }
            kept_scenario_ids.append(scenario_id)
            continue

        try:
            gt_data_dict = data_bridge.get_ground_truth(args.scenario_dir, scenario_filename)
            sim = data_bridge.create_simulation(args.scenario_dir, scenario_filename)
            scenario = sim.getScenario()
            vehicles = list(scenario.vehicles())
            moving_veh_ids = [v.getID() for v in scenario.getObjectsThatMoved()]
            
            # 加载预处理数据并过滤车辆
            preproc_data = None
            preproc_veh_ids = None
            if args.preprocess_dir:
                try:
                    preproc_data, file_exists = data_bridge.load_preprocessed_data(scenario_id)
                    if file_exists and preproc_data is not None:
                        scenarios_with_preproc += 1
                        preproc_veh_ids = _get_preproc_vehicle_ids(preproc_data, gt_data_dict)
                        if preproc_veh_ids is not None and len(preproc_veh_ids) > 0:
                            # 只保留预处理数据中存在的车辆
                            original_count = len(moving_veh_ids)
                            moving_veh_ids = [vid for vid in moving_veh_ids if vid in preproc_veh_ids]
                            if len(moving_veh_ids) < original_count:
                                scenarios_filtered += 1
                                filtered_count = original_count - len(moving_veh_ids)
                                print(f"  {scenario_id}: Filtered {filtered_count} vehicles (from {original_count} to {len(moving_veh_ids)}), preproc has {len(preproc_veh_ids)} agents")
                            elif len(moving_veh_ids) == 0:
                                print(f"  {scenario_id}: Warning - All vehicles filtered out! Preproc has {len(preproc_veh_ids)} agents: {preproc_veh_ids}")
                        else:
                            print(f"  {scenario_id}: Warning - Cannot extract vehicle IDs from preprocessed data, using all moving vehicles")
                except Exception as e:
                    print(f"  {scenario_id}: Warning - Failed to load preprocessed data: {e}")
                    import traceback
                    traceback.print_exc()

            ego_id, ego_selection_mode = _select_ego_vehicle_id(
                moving_veh_ids,
                gt_data_dict,
                vehicles,
                max_episode_steps,
                history_steps,
            )

            if args.filter_close_goal_point:
                start_goal_dist = _compute_start_goal_distance(
                    ego_id=ego_id,
                    vehicles=vehicles,
                    gt_data_dict=gt_data_dict,
                    max_episode_steps=max_episode_steps,
                )
                if (
                    start_goal_dist is not None
                    and start_goal_dist < args.close_goal_threshold_m
                ):
                    close_goal_filtered_scenarios += 1
                    sim.reset()
                    continue
            
            opponent_ids = _select_opponent_vehicle_ids(
                moving_veh_ids,
                vehicles,
                gt_data_dict,
                ego_id,
                history_steps,
                k=7,
                traj_len_threshold=60,
            )
            
            vehicle_map[scenario_id] = {
                "ego_vehicle_id": ego_id,
                "ego_selection_mode": ego_selection_mode,
                "opponent_vehicle_ids": opponent_ids,
            }
            kept_scenario_ids.append(scenario_id)
            sim.reset()
        except Exception as e:
            print(f"Warning: failed to process {scenario_id}: {e}")
            vehicle_map[scenario_id] = {
                "ego_vehicle_id": None,
                "ego_selection_mode": "unknown",
                "opponent_vehicle_ids": [],
            }
            kept_scenario_ids.append(scenario_id)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-detect based on input filename
        if "valid" in args.scenario_index_json:
            output_path = os.path.join("data", "vehicle_map_valid.json")
        elif "train" in args.scenario_index_json:
            output_path = os.path.join("data", "vehicle_map_train.json")
        else:
            output_path = os.path.join("data", "vehicle_map.json")
    
    if not os.path.isdir("data"):
        raise FileNotFoundError("data directory not found for output")
    with open(output_path, "w") as f:
        json.dump(vehicle_map, f, indent=2)

    if args.filter_close_goal_point:
        filtered_vehicle_map_path = os.path.join("data", "vehicle_map_filtered.json")
        filtered_scenario_index_path = os.path.join("data", "scenarios_index_filtered.json")

        filtered_vehicle_map = {
            sid: vehicle_map[sid] for sid in kept_scenario_ids if sid in vehicle_map
        }
        filtered_index_data = {
            "version": "1.0",
            "source_dir": index_source_dir or args.scenario_dir,
            "total_scenarios": len(kept_scenario_ids),
            "scenario_ids": kept_scenario_ids,
        }

        with open(filtered_vehicle_map_path, "w") as f:
            json.dump(filtered_vehicle_map, f, indent=2)
        with open(filtered_scenario_index_path, "w") as f:
            json.dump(filtered_index_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved vehicle map to: {output_path}")
    if args.filter_close_goal_point:
        print(f"Saved filtered vehicle map to: {filtered_vehicle_map_path}")
        print(f"Saved filtered scenario index to: {filtered_scenario_index_path}")
    print(f"\nStatistics:")
    print(f"  Total scenarios: {total_scenarios}")
    if args.filter_close_goal_point:
        print(f"  Close-goal filtered scenarios: {close_goal_filtered_scenarios}")
        print(f"  Remaining scenarios after close-goal filter: {len(kept_scenario_ids)}")
    if args.preprocess_dir:
        print(f"  Scenarios with preprocessed data: {scenarios_with_preproc}")
        print(f"  Scenarios with filtered vehicles: {scenarios_filtered}")
        print(f"  Coverage: {scenarios_with_preproc}/{total_scenarios} = {100*scenarios_with_preproc/total_scenarios:.1f}%")
    else:
        print(f"  Note: No preprocess_dir provided, no vehicle filtering applied")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
