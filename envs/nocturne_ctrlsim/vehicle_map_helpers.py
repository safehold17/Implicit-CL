"""
Vehicle-map helper functions for Nocturne CtrlSim adversarial env.
"""

import json
import os
from typing import Dict, List, Optional, Tuple


def load_vehicle_map(env) -> Optional[Dict]:
    """Load vehicle map JSON file (cached)."""
    if env._vehicle_map_cache is not None:
        return env._vehicle_map_cache

    if not env.vehicle_map_path or not os.path.exists(env.vehicle_map_path):
        return None

    try:
        with open(env.vehicle_map_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            env._vehicle_map_cache = data
            return data
    except Exception as e:
        print(f"Warning: Failed to load vehicle map {env.vehicle_map_path}: {e}")

    return None


def load_vehicle_ids_for_scenario(
    env, scenario_id: str
) -> Tuple[int, List[int], str]:
    """
    Load ego vehicle ID and opponent vehicle IDs for a scenario.

    Strict mode: vehicle map and scenario entries must exist; no dynamic fallback.
    """
    vehicle_map = load_vehicle_map(env)

    if vehicle_map is None:
        raise FileNotFoundError(
            f"Vehicle map is unavailable: {env.vehicle_map_path}. "
            "Strict mode requires a valid vehicle map."
        )

    scenario_data = vehicle_map.get(scenario_id)

    if scenario_data is None or not isinstance(scenario_data, dict):
        raise KeyError(
            f"Scenario '{scenario_id}' not found in vehicle map '{env.vehicle_map_path}'. "
            "Strict mode requires all scenarios to be present."
        )

    ego_id = scenario_data.get("ego_vehicle_id")
    opponent_ids = scenario_data.get("opponent_vehicle_ids", [])
    ego_selection_mode = scenario_data.get("ego_selection_mode", "unknown")
    if ego_selection_mode not in ("interesting", "dense"):
        ego_selection_mode = "unknown"

    # Validate ego_id
    if ego_id is None:
        raise ValueError(
            f"ego_vehicle_id is missing for scenario '{scenario_id}' "
            f"in vehicle map '{env.vehicle_map_path}'."
        )

    if opponent_ids is None:
        opponent_ids = []
    if not isinstance(opponent_ids, list):
        raise ValueError(
            f"opponent_vehicle_ids for scenario '{scenario_id}' must be a list, got {type(opponent_ids)}."
        )

    return int(ego_id), [int(vid) for vid in opponent_ids], ego_selection_mode
