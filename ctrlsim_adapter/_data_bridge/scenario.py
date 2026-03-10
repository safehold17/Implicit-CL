from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.sim import get_moving_vehicles, get_road_data, get_sim


def get_moving_vehicle_ids(bridge: Any, scenario: Any) -> List[int]:
    return get_moving_vehicles(scenario)


def get_road_data_for_scenario(bridge: Any, scenario: Any) -> List[Dict]:
    del bridge
    return get_road_data(scenario)


def extract_road_edge_polylines(bridge: Any, road_data: List[Dict]) -> List[np.ndarray]:
    del bridge
    road_edge_polylines = []
    for road in road_data:
        if road["type"] == "road_edge":
            geometry = road["geometry"]
            if isinstance(geometry, list):
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

