"""
负责 DataBridge 所需的场景级辅助能力，包括仿真创建、道路访问和场景枚举。
该模块封装了对 `utils.sim` 的薄适配，保持桥接层调用接口整洁。
Provides scenario-level helpers for DataBridge, including simulation creation, road access, and scenario enumeration.
Acts as a thin adapter over `utils.sim` so the bridge layer can keep a clean interface.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.sim import get_moving_vehicles, get_road_data, get_sim

# Road type index mapping (must match ROAD_TYPE_MAPPING in student_env_policy.py)
_ROAD_TYPE_TO_IDX = {
    "none": 0, "road_line": 1, "road_edge": 2, "lane": 3,
    "crosswalk": 4, "speed_bump": 5, "stop_sign": 6, "other": 0,
}
_STATIC_ROAD_TYPE_TO_IDX = {"stop_sign": 6, "crosswalk": 4, "speed_bump": 5}


def get_moving_vehicle_ids(bridge: Any, scenario: Any) -> List[int]:
    return get_moving_vehicles(scenario)


def get_road_data_for_scenario(bridge: Any, scenario: Any) -> List[Dict]:
    del bridge
    return get_road_data(scenario)


def road_data_to_numpy(road_data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Flatten road_data (list-of-dicts) into compact numpy arrays.

    Called once per scenario reset; eliminates the per-step Python loop
    over road geometry in build_road_graph_obs().

    Returns:
        pts_xy     float32 (N, 2)  world-space [x, y] for each road point
        seg_len    float32 (N,)    segment length (0 for static objects)
        seg_orient float32 (N,)    segment orientation in world frame (radians)
        type_idx   int32   (N,)    road type index matching ROAD_TYPE_MAPPING
        is_line    bool    (N,)    True for polyline pts, False for static objects
        point_indices int64 (N,)   Stable insertion-order index for each point.
                                   Used when multiple road points have identical distance
                                   to the ego vehicle during top-k selection.
    """
    pts_xy = []
    seg_len = []
    seg_orient = []
    type_idx = []
    is_line = []

    for road_item in road_data:
        road_type = road_item["type"]
        geometry = road_item["geometry"]

        if isinstance(geometry, list) and len(geometry) > 0:
            tidx = _ROAD_TYPE_TO_IDX.get(road_type, 0)
            last_idx = len(geometry) - 1
            for i, pt in enumerate(geometry):
                pts_xy.append((pt["x"], pt["y"]))
                type_idx.append(tidx)
                is_line.append(True)
                if i < last_idx:
                    nxt = geometry[i + 1]
                    dx = nxt["x"] - pt["x"]
                    dy = nxt["y"] - pt["y"]
                    seg_len.append(math.sqrt(dx * dx + dy * dy))
                    seg_orient.append(math.atan2(dy, dx))
                else:
                    seg_len.append(1.0)
                    seg_orient.append(0.0)

        elif isinstance(geometry, dict):
            tidx = _STATIC_ROAD_TYPE_TO_IDX.get(road_type, 0)
            pts_xy.append((geometry["x"], geometry["y"]))
            seg_len.append(0.0)
            seg_orient.append(0.0)
            type_idx.append(tidx)
            is_line.append(False)

    if not pts_xy:
        empty = np.empty(0, dtype=np.float32)
        return {
            "pts_xy": np.empty((0, 2), dtype=np.float32),
            "seg_len": empty,
            "seg_orient": empty,
            "type_idx": np.empty(0, dtype=np.int32),
            "is_line": np.empty(0, dtype=bool),
            "point_indices": np.empty(0, dtype=np.int64),
        }

    point_indices = np.arange(len(pts_xy), dtype=np.int64)
    return {
        "pts_xy":      np.array(pts_xy,    dtype=np.float32),
        "seg_len":     np.array(seg_len,   dtype=np.float32),
        "seg_orient":  np.array(seg_orient, dtype=np.float32),
        "type_idx":    np.array(type_idx,  dtype=np.int32),
        "is_line":     np.array(is_line,   dtype=bool),
        "point_indices": point_indices,
    }


def extract_road_edge_polylines(bridge: Any, road_data: List[Dict]) -> List[np.ndarray]:
    del bridge
    road_edge_polylines = []
    for road in road_data:
        if road["type"] == "road_edge":
            geometry = road["geometry"]
            if isinstance(geometry, list) and len(geometry) >= 2:
                polyline = np.array([[pt["x"], pt["y"]] for pt in geometry])
                road_edge_polylines.append(polyline)
    return road_edge_polylines


def create_simulation(
    bridge: Any,
    scenario_path: str,
    scenario_filename: str,
) -> Any:
    files = [scenario_filename]
    file_id = 0
    return get_sim(bridge.cfg, scenario_path, files, file_id)


def load_scenario(loader: Any, scenario_id: str) -> Tuple[Any, Dict, Optional[Dict], List[int]]:
    scenario_filename = f"{scenario_id}.json"
    sim = loader.bridge.create_simulation(loader.scenario_dir, scenario_filename)
    scenario = sim.getScenario()
    gt_data_dict = loader.bridge.get_ground_truth(loader.scenario_dir, scenario_filename)
    preproc_data, _ = loader.bridge.load_preprocessed_data(scenario_id)
    moving_ids = loader.bridge.get_moving_vehicle_ids(scenario)
    return sim, gt_data_dict, preproc_data, moving_ids


def get_scenario_list(loader: Any) -> List[str]:
    files = glob.glob(os.path.join(loader.scenario_dir, "*.json"))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]
