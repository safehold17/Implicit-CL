#!/usr/bin/env python3
"""
Build vehicle map for scenarios listed in scenarios_index.json.

Output format:
    {
        "scenario_id": {
            "ego_vehicle_id": int or null,
            "ego_selection_mode": "interesting" | "dense" | "unknown",
            "opponent_vehicle_ids": [int, ...] or [],
            "opponent_vehicle_num": int
        },
        ...
    }
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Reduce per-process thread oversubscription in multiprocess mode.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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


def _find_degenerate_road_edge_segment(
    scenario_path: str,
    eps: float = 1e-12,
) -> Optional[Tuple[int, int, Tuple[float, float], Tuple[float, float], float]]:
    """
    Find the first degenerate road_edge segment in one scenario file.

    A segment is considered degenerate when two adjacent geometry points are
    identical (or near-identical within eps).
    """
    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)

    roads = scenario_data.get("roads")
    if not isinstance(roads, list):
        return None

    eps_sq = eps * eps
    for road_idx, road in enumerate(roads):
        if not isinstance(road, dict) or road.get("type") != "road_edge":
            continue
        geometry = road.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 2:
            continue

        for seg_idx in range(len(geometry) - 1):
            start = geometry[seg_idx]
            end = geometry[seg_idx + 1]
            if not isinstance(start, dict) or not isinstance(end, dict):
                continue
            if "x" not in start or "y" not in start or "x" not in end or "y" not in end:
                continue

            sx = float(start["x"])
            sy = float(start["y"])
            ex = float(end["x"])
            ey = float(end["y"])
            len_sq = (ex - sx) ** 2 + (ey - sy) ** 2
            if len_sq <= eps_sq:
                return road_idx, seg_idx, (sx, sy), (ex, ey), float(np.sqrt(len_sq))

    return None


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


def _default_output_filename(scenario_index_json: str) -> str:
    """Return default output filename by scenario split."""
    if "valid" in scenario_index_json:
        return "vehicle_map_filtered_valid.json"
    if "train" in scenario_index_json:
        return "vehicle_map_filtered_train.json"
    return "vehicle_map_filtered.json"


def _determine_output_path(scenario_index_json: str, output: Optional[str]) -> str:
    """Determine output file path from args.

    Supports both:
    - output as a file path (e.g. /tmp/vehicle_map.json)
    - output as a directory path (e.g. /tmp/out_dir or /tmp/out_dir/)
    """
    default_filename = _default_output_filename(scenario_index_json)
    if output:
        output_abs = os.path.abspath(output)
        output_base = os.path.basename(output.rstrip(os.sep))
        has_extension = bool(os.path.splitext(output_base)[1])
        is_directory_like = (
            output.endswith(os.sep)
            or os.path.isdir(output_abs)
            or not has_extension
        )
        if is_directory_like:
            return os.path.join(output, default_filename)
        return output
    return os.path.join("data", default_filename)


def _save_json(path: str, payload: Dict) -> None:
    """Save json payload to path."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    """Save checkpoint atomically."""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint if exists."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _process_single_scenario(
    scenario_id: str,
    data_bridge: DataBridge,
    scenario_dir: str,
    preprocess_dir: str,
    history_steps: int,
    max_episode_steps: int,
    close_goal_threshold_m: float,
) -> Dict[str, Any]:
    """Process one scenario and return structured result."""
    scenario_filename = f"{scenario_id}.json"
    scenario_path = os.path.join(scenario_dir, scenario_filename)
    if not os.path.exists(scenario_path):
        return {
            "status": "missing_scenario",
            "scenario_id": scenario_id,
            "detail": f"scenario not found: {scenario_path}",
            "warnings": [],
            "with_preproc": False,
            "vehicle_filtered": False,
        }

    degenerate_segment = _find_degenerate_road_edge_segment(scenario_path)
    if degenerate_segment is not None:
        road_idx, seg_idx, start, end, seg_len = degenerate_segment
        return {
            "status": "degenerate_road_edge",
            "scenario_id": scenario_id,
            "detail": (
                f"Skip degenerate road_edge segment (road[{road_idx}] "
                f"seg[{seg_idx}] start={start} end={end} len={seg_len})"
            ),
            "warnings": [],
            "with_preproc": False,
            "vehicle_filtered": False,
        }

    sim = None
    warnings: List[str] = []
    with_preproc = False
    vehicle_filtered = False

    try:
        sim = data_bridge.create_simulation(scenario_dir, scenario_filename)
        gt_data_dict = data_bridge.get_ground_truth_from_sim(sim, scenario_filename)
        scenario = sim.getScenario()
        vehicles = list(scenario.vehicles())
        moving_veh_ids = [v.getID() for v in scenario.getObjectsThatMoved()]

        if preprocess_dir:
            try:
                preproc_data, file_exists = data_bridge.load_preprocessed_data(scenario_id)
                if file_exists and preproc_data is not None:
                    with_preproc = True
                    preproc_veh_ids = _get_preproc_vehicle_ids(preproc_data, gt_data_dict)
                    if preproc_veh_ids is not None and len(preproc_veh_ids) > 0:
                        preproc_veh_id_set = set(preproc_veh_ids)
                        original_count = len(moving_veh_ids)
                        moving_veh_ids = [
                            vid for vid in moving_veh_ids if vid in preproc_veh_id_set
                        ]
                        if len(moving_veh_ids) < original_count:
                            vehicle_filtered = True
                            filtered_count = original_count - len(moving_veh_ids)
                            warnings.append(
                                f"{scenario_id}: Filtered {filtered_count} vehicles "
                                f"(from {original_count} to {len(moving_veh_ids)}), "
                                f"preproc has {len(preproc_veh_ids)} agents"
                            )
                        if len(moving_veh_ids) == 0:
                            warnings.append(
                                f"{scenario_id}: Warning - All vehicles filtered out! "
                                f"Preproc has {len(preproc_veh_ids)} agents"
                            )
                    else:
                        warnings.append(
                            f"{scenario_id}: Warning - Cannot extract vehicle IDs from "
                            "preprocessed data, using all moving vehicles"
                        )
            except Exception as e:
                warnings.append(
                    f"{scenario_id}: Warning - Failed to load preprocessed data: {e}"
                )

        ego_id, ego_selection_mode = _select_ego_vehicle_id(
            moving_veh_ids,
            gt_data_dict,
            vehicles,
            max_episode_steps,
            history_steps,
        )

        start_goal_dist = _compute_start_goal_distance(
            ego_id=ego_id,
            vehicles=vehicles,
            gt_data_dict=gt_data_dict,
            max_episode_steps=max_episode_steps,
        )
        if (
            start_goal_dist is not None
            and start_goal_dist < close_goal_threshold_m
        ):
            return {
                "status": "close_goal_filtered",
                "scenario_id": scenario_id,
                "detail": (
                    f"start_goal_dist={start_goal_dist:.3f} < "
                    f"{close_goal_threshold_m:.3f}"
                ),
                "warnings": warnings,
                "with_preproc": with_preproc,
                "vehicle_filtered": vehicle_filtered,
            }

        opponent_ids = _select_opponent_vehicle_ids(
            moving_veh_ids,
            vehicles,
            gt_data_dict,
            ego_id,
            history_steps,
            k=7,
            traj_len_threshold=60,
        )

        return {
            "status": "ok",
            "scenario_id": scenario_id,
            "detail": "",
            "warnings": warnings,
            "with_preproc": with_preproc,
            "vehicle_filtered": vehicle_filtered,
            "vehicle_map_item": {
                "ego_vehicle_id": ego_id,
                "ego_selection_mode": ego_selection_mode,
                "opponent_vehicle_ids": opponent_ids,
                "opponent_vehicle_num": len(opponent_ids),
            },
        }
    except Exception as e:
        return {
            "status": "failed",
            "scenario_id": scenario_id,
            "detail": str(e),
            "warnings": warnings,
            "with_preproc": with_preproc,
            "vehicle_filtered": vehicle_filtered,
        }
    finally:
        if sim is not None:
            try:
                sim.reset()
            except Exception:
                pass


def _worker_loop(
    worker_cfg: Dict[str, Any],
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Worker process loop for isolated scenario execution."""
    cfg = create_minimal_config(
        checkpoint_path="",
        scenario_dir=worker_cfg["scenario_dir"],
        preprocess_dir=worker_cfg["preprocess_dir"],
    )
    data_bridge = DataBridge(cfg, preprocess_dir=worker_cfg["preprocess_dir"] or "")
    history_steps = int(getattr(cfg.nocturne, "history_steps", 10))
    max_episode_steps = int(worker_cfg["max_episode_steps"])

    while True:
        scenario_id = task_queue.get()
        if scenario_id is None:
            break
        result = _process_single_scenario(
            scenario_id=scenario_id,
            data_bridge=data_bridge,
            scenario_dir=worker_cfg["scenario_dir"],
            preprocess_dir=worker_cfg["preprocess_dir"],
            history_steps=history_steps,
            max_episode_steps=max_episode_steps,
            close_goal_threshold_m=worker_cfg["close_goal_threshold_m"],
        )
        result_queue.put(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vehicle map for scenarios")
    parser.add_argument(
        "--scenario_index_json",
        type=str,
        default="/home/chen/workspace/dcd-ctrlsim/data/scenarios_index_train.json",
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
        default="/home/chen/Downloads/data/nocturne_mini/formatted_json_v2_no_tl_train/formatted_json_v2_no_tl_train",
        help="Scenario directory (overrides config.yaml)",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="/home/chen/Downloads/data/processed/training",
        help="Preprocessed data directory (required to filter vehicles by preprocessed data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/vehicle_map_valid.json or vehicle_map_train.json)",
    )
    parser.add_argument(
        "--close_goal_threshold_m",
        type=float,
        default=5.0,
        help="Threshold in meters for close-goal filtering (default: 5.0).",
    )
    parser.add_argument(
        "--isolate_scenario_process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Process scenarios in an isolated worker process to survive native crashes. "
            "Use --isolate_scenario_process / --no-isolate_scenario_process."
        ),
    )
    parser.add_argument(
        "--scenario_timeout_s",
        type=float,
        default=120.0,
        help="Timeout (seconds) for one scenario when isolation is enabled.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker processes in isolation mode (default: 16).",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1000,
        help="Checkpoint interval by processed scenario count. <=0 disables checkpointing.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint file path (default: <output>.checkpoint.json).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from checkpoint if exists. "
            "Use --resume_from_checkpoint / --no-resume_from_checkpoint."
        ),
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

    output_path = _determine_output_path(args.scenario_index_json, args.output)
    checkpoint_path = args.checkpoint_path or f"{output_path}.checkpoint.json"
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    max_episode_steps = 90

    # 统计信息与结果容器
    vehicle_map: Dict[str, Dict] = {}
    kept_scenario_ids: List[str] = []
    stats: Dict[str, int] = {
        "total_scenarios": 0,
        "scenarios_with_preproc": 0,
        "scenarios_filtered": 0,
        "close_goal_filtered_scenarios": 0,
        "degenerate_road_edge_scenarios": 0,
        "failed_scenarios": 0,
        "timeout_scenarios": 0,
        "worker_crash_scenarios": 0,
    }
    failed_scenario_ids: List[str] = []
    start_index = 0

    if args.resume_from_checkpoint:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            if (
                checkpoint.get("scenario_index_json") == args.scenario_index_json
                and checkpoint.get("scenario_dir") == args.scenario_dir
            ):
                start_index = int(checkpoint.get("next_index", 0))
                vehicle_map = checkpoint.get("vehicle_map", {})
                kept_scenario_ids = checkpoint.get("kept_scenario_ids", [])
                loaded_stats = checkpoint.get("stats", {})
                for key in stats:
                    stats[key] = int(loaded_stats.get(key, stats[key]))
                failed_scenario_ids = checkpoint.get("failed_scenario_ids", [])
                print(
                    f"Resuming from checkpoint: index={start_index}/"
                    f"{len(scenario_ids)}, path={checkpoint_path}"
                )
            else:
                print(
                    f"Warning: checkpoint mismatch, ignored: {checkpoint_path}"
                )

    data_bridge: Optional[DataBridge] = None
    history_steps = 10
    if not args.isolate_scenario_process:
        cfg = create_minimal_config(
            checkpoint_path="",
            scenario_dir=args.scenario_dir,
            preprocess_dir=args.preprocess_dir,
        )
        data_bridge = DataBridge(cfg, preprocess_dir=args.preprocess_dir or "")
        history_steps = int(getattr(cfg.nocturne, "history_steps", 10))
    completed_flags = [False] * len(scenario_ids)
    for done_idx in range(min(start_index, len(scenario_ids))):
        completed_flags[done_idx] = True
    next_done_index = min(start_index, len(scenario_ids))
    last_checkpoint_done = next_done_index

    def handle_result(scenario_id: str, result: Dict[str, Any]) -> None:
        stats["total_scenarios"] += 1
        if result.get("with_preproc"):
            stats["scenarios_with_preproc"] += 1
        if result.get("vehicle_filtered"):
            stats["scenarios_filtered"] += 1

        for warning in result.get("warnings", []):
            print(f"  {warning}")

        status = result.get("status")
        detail = result.get("detail", "")
        if status == "ok":
            vehicle_map[scenario_id] = result["vehicle_map_item"]
        elif status == "degenerate_road_edge":
            stats["degenerate_road_edge_scenarios"] += 1
            print(f"  {scenario_id}: {detail}")
        elif status == "close_goal_filtered":
            stats["close_goal_filtered_scenarios"] += 1
        elif status == "missing_scenario":
            print(f"Warning: {detail}")
        else:
            stats["failed_scenarios"] += 1
            failed_scenario_ids.append(scenario_id)
            print(f"Warning: failed to process {scenario_id}: {detail}")

    def mark_completed(idx: int) -> None:
        nonlocal next_done_index
        completed_flags[idx] = True
        while next_done_index < len(scenario_ids) and completed_flags[next_done_index]:
            next_done_index += 1

    def maybe_save_checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_done
        if args.checkpoint_every <= 0:
            return
        should_save = force or (
            next_done_index - last_checkpoint_done >= args.checkpoint_every
        )
        if not should_save:
            return
        checkpoint_payload = {
            "version": 1,
            "scenario_index_json": args.scenario_index_json,
            "scenario_dir": args.scenario_dir,
            "next_index": next_done_index,
            "vehicle_map": vehicle_map,
            "kept_scenario_ids": [
                sid for sid in scenario_ids[:next_done_index] if sid in vehicle_map
            ],
            "stats": stats,
            "failed_scenario_ids": failed_scenario_ids,
        }
        _save_checkpoint(checkpoint_path, checkpoint_payload)
        last_checkpoint_done = next_done_index

    if args.isolate_scenario_process:
        worker_count = max(1, int(args.num_workers))
        worker_ctx = mp.get_context("spawn")
        worker_cfg = {
            "scenario_dir": args.scenario_dir,
            "preprocess_dir": args.preprocess_dir,
            "max_episode_steps": max_episode_steps,
            "close_goal_threshold_m": args.close_goal_threshold_m,
        }
        workers: List[Dict[str, Any]] = []

        def stop_worker(worker_idx: int) -> None:
            worker = workers[worker_idx]
            proc = worker.get("process")
            task_q = worker.get("task_queue")
            if proc is None:
                return
            if proc.is_alive():
                try:
                    task_q.put_nowait(None)
                except Exception:
                    pass
                proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
            worker["process"] = None
            worker["task_queue"] = None
            worker["result_queue"] = None
            worker["active_idx"] = None
            worker["active_scenario_id"] = None
            worker["start_time"] = None

        def start_worker(worker_idx: int) -> None:
            stop_worker(worker_idx)
            task_q = worker_ctx.Queue(maxsize=1)
            result_q = worker_ctx.Queue(maxsize=1)
            proc = worker_ctx.Process(
                target=_worker_loop,
                args=(worker_cfg, task_q, result_q),
                daemon=True,
            )
            proc.start()
            workers[worker_idx] = {
                "process": proc,
                "task_queue": task_q,
                "result_queue": result_q,
                "active_idx": None,
                "active_scenario_id": None,
                "start_time": None,
            }

        for _ in range(worker_count):
            workers.append(
                {
                    "process": None,
                    "task_queue": None,
                    "result_queue": None,
                    "active_idx": None,
                    "active_scenario_id": None,
                    "start_time": None,
                }
            )
        for worker_idx in range(worker_count):
            start_worker(worker_idx)

        next_dispatch_idx = start_index
        progress_total = len(scenario_ids) - start_index
        try:
            with tqdm(
                total=progress_total,
                desc="Building vehicle map",
                unit="scenario",
                dynamic_ncols=True,
            ) as pbar:
                while next_done_index < len(scenario_ids):
                    progress_made = False

                    for worker_idx, worker in enumerate(workers):
                        if worker["active_idx"] is not None:
                            continue
                        if next_dispatch_idx >= len(scenario_ids):
                            break
                        scenario_id = scenario_ids[next_dispatch_idx]
                        worker["task_queue"].put(scenario_id)
                        worker["active_idx"] = next_dispatch_idx
                        worker["active_scenario_id"] = scenario_id
                        worker["start_time"] = time.monotonic()
                        next_dispatch_idx += 1
                        progress_made = True

                    for worker_idx, worker in enumerate(workers):
                        active_idx = worker["active_idx"]
                        if active_idx is None:
                            continue
                        scenario_id = worker["active_scenario_id"]
                        try:
                            result = worker["result_queue"].get_nowait()
                            handle_result(scenario_id, result)
                            mark_completed(active_idx)
                            worker["active_idx"] = None
                            worker["active_scenario_id"] = None
                            worker["start_time"] = None
                            pbar.update(1)
                            maybe_save_checkpoint()
                            progress_made = True
                            continue
                        except queue.Empty:
                            pass
                        except Exception as e:
                            stats["worker_crash_scenarios"] += 1
                            result = {
                                "status": "worker_crash",
                                "scenario_id": scenario_id,
                                "detail": f"Worker communication failed: {e}",
                                "warnings": [],
                                "with_preproc": False,
                                "vehicle_filtered": False,
                            }
                            handle_result(scenario_id, result)
                            mark_completed(active_idx)
                            pbar.update(1)
                            maybe_save_checkpoint()
                            start_worker(worker_idx)
                            progress_made = True
                            continue

                        proc = worker["process"]
                        elapsed = time.monotonic() - worker["start_time"]
                        if proc is None or not proc.is_alive():
                            stats["worker_crash_scenarios"] += 1
                            result = {
                                "status": "worker_crash",
                                "scenario_id": scenario_id,
                                "detail": "Worker process crashed (possible native SIGFPE)",
                                "warnings": [],
                                "with_preproc": False,
                                "vehicle_filtered": False,
                            }
                            handle_result(scenario_id, result)
                            mark_completed(active_idx)
                            pbar.update(1)
                            maybe_save_checkpoint()
                            start_worker(worker_idx)
                            progress_made = True
                        elif elapsed > args.scenario_timeout_s:
                            stats["timeout_scenarios"] += 1
                            result = {
                                "status": "timeout",
                                "scenario_id": scenario_id,
                                "detail": (
                                    f"Scenario timed out after "
                                    f"{args.scenario_timeout_s:.1f}s"
                                ),
                                "warnings": [],
                                "with_preproc": False,
                                "vehicle_filtered": False,
                            }
                            handle_result(scenario_id, result)
                            mark_completed(active_idx)
                            pbar.update(1)
                            maybe_save_checkpoint()
                            start_worker(worker_idx)
                            progress_made = True

                    if not progress_made:
                        time.sleep(0.01)
        finally:
            for worker_idx in range(worker_count):
                stop_worker(worker_idx)
            maybe_save_checkpoint(force=True)
    else:
        with tqdm(
            range(start_index, len(scenario_ids)),
            desc="Building vehicle map",
            unit="scenario",
            dynamic_ncols=True,
        ) as pbar_range:
            for idx in pbar_range:
                scenario_id = scenario_ids[idx]
                result = _process_single_scenario(
                    scenario_id=scenario_id,
                    data_bridge=data_bridge,
                    scenario_dir=args.scenario_dir,
                    preprocess_dir=args.preprocess_dir,
                    history_steps=history_steps,
                    max_episode_steps=max_episode_steps,
                    close_goal_threshold_m=args.close_goal_threshold_m,
                )
                handle_result(scenario_id, result)
                mark_completed(idx)
                maybe_save_checkpoint()
        maybe_save_checkpoint(force=True)
    
    kept_scenario_ids = [sid for sid in scenario_ids if sid in vehicle_map]
    filtered_scenario_index_path = os.path.join(
        output_dir, "scenarios_index_filtered.json"
    )
    filtered_index_data = {
        "version": "1.0",
        "source_dir": index_source_dir or args.scenario_dir,
        "total_scenarios": len(kept_scenario_ids),
        "scenario_ids": kept_scenario_ids,
    }
    _save_json(output_path, vehicle_map)
    _save_json(filtered_scenario_index_path, filtered_index_data)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Saved filtered vehicle map to: {output_path}")
    print(f"Saved filtered scenario index to: {filtered_scenario_index_path}")
    print(f"\nStatistics:")
    print(f"  Total scenarios: {stats['total_scenarios']}")
    print(f"  Degenerate-road-edge skipped scenarios: {stats['degenerate_road_edge_scenarios']}")
    print(f"  Processed scenarios: {len(kept_scenario_ids)}")
    print(f"  Close-goal filtered scenarios: {stats['close_goal_filtered_scenarios']}")
    print(f"  Remaining scenarios after close-goal filter: {len(kept_scenario_ids)}")
    if args.preprocess_dir:
        print(f"  Scenarios with preprocessed data: {stats['scenarios_with_preproc']}")
        print(f"  Scenarios with filtered vehicles: {stats['scenarios_filtered']}")
        if stats["total_scenarios"] > 0:
            coverage = 100 * stats["scenarios_with_preproc"] / stats["total_scenarios"]
            print(
                f"  Coverage: {stats['scenarios_with_preproc']}/"
                f"{stats['total_scenarios']} = {coverage:.1f}%"
            )
    else:
        print(f"  Note: No preprocess_dir provided, no vehicle filtering applied")
    print(f"  Failed scenarios: {stats['failed_scenarios']}")
    if args.isolate_scenario_process:
        print(f"  Timeout scenarios: {stats['timeout_scenarios']}")
        print(f"  Worker-crash scenarios: {stats['worker_crash_scenarios']}")
    if len(failed_scenario_ids) > 0:
        failed_preview = ", ".join(failed_scenario_ids[:10])
        print(f"  Failed scenario sample (up to 10): {failed_preview}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
