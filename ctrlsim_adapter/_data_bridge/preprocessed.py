"""
负责定位、缓存、加载并整理 ctrl-sim 预处理数据。
该模块将原始预处理文件转换成适配器直接可用的 RTG、道路和奖励相关结构。
Locates, caches, loads, and normalizes ctrl-sim preprocessed data for the bridge layer.
Turns raw preprocessed files into adapter-ready RTG, road, and reward-related structures.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


COMPRESSED_PREPROCESS_FORMAT = "ctrlsim_preprocessed_compressed"


def update_preprocessed_data_cache(
    bridge: Any,
    scenario_id: str,
    preproc_data: Dict[str, Any],
) -> None:
    if bridge.preproc_cache_size <= 0:
        return

    if scenario_id in bridge._preproc_data_cache:
        bridge._preproc_data_cache.pop(scenario_id)
    bridge._preproc_data_cache[scenario_id] = preproc_data
    if len(bridge._preproc_data_cache) > bridge.preproc_cache_size:
        bridge._preproc_data_cache.popitem(last=False)


def ensure_preprocessed_files_cache(bridge: Any) -> None:
    if bridge._preprocessed_files_cache is not None:
        return

    bridge._preprocessed_files_cache = {}
    bridge._preprocessed_indices_cache = {}
    if bridge.preprocessed_dset is None:
        return

    for idx, filepath in enumerate(bridge.preprocessed_dset.files):
        basename = os.path.basename(filepath)
        if basename.endswith("_physics.pkl"):
            scenario_id = basename.replace("_physics.pkl", "")
            bridge._preprocessed_files_cache[scenario_id] = filepath
            bridge._preprocessed_indices_cache[scenario_id] = idx


def load_preprocessed_data(
    bridge: Any,
    scenario_filename: str,
) -> Tuple[Optional[Dict], bool]:
    if bridge.preprocessed_dset is None:
        return None, False

    scenario_id = os.path.splitext(scenario_filename)[0]
    ensure_preprocessed_files_cache(bridge)

    if bridge.preproc_cache_size > 0 and scenario_id in bridge._preproc_data_cache:
        preproc_data = bridge._preproc_data_cache.pop(scenario_id)
        bridge._preproc_data_cache[scenario_id] = preproc_data
        return preproc_data, True

    if scenario_id not in bridge._preprocessed_files_cache:
        return None, False

    file_path = bridge._preprocessed_files_cache[scenario_id]
    idx = bridge._preprocessed_indices_cache.get(scenario_id)
    if idx is None:
        return None, False

    try:
        compressed_data = load_compressed_preprocessed_data(file_path)
        if compressed_data is not None:
            update_preprocessed_data_cache(bridge, scenario_id, compressed_data)
            return compressed_data, True

        preproc_data = bridge.preprocessed_dset[idx]
        update_preprocessed_data_cache(bridge, scenario_id, preproc_data)
        return preproc_data, True
    except Exception as exc:
        print(f"Warning: Failed to load preprocessed data for {scenario_id}: {exc}")
        return None, False


def load_compressed_preprocessed_data(file_path: str) -> Optional[Dict[str, Any]]:
    with open(file_path, "rb") as file:
        raw_data = pickle.load(file)

    if raw_data.get("format") != COMPRESSED_PREPROCESS_FORMAT:
        return None

    return {
        "rtgs": raw_data["rtgs"],
        "road_points": raw_data["road_points"],
        "road_types": raw_data["road_types"],
    }


def compute_rewards(
    bridge: Any,
    ag_data: np.ndarray,
    ag_rewards: np.ndarray,
    veh_edge_dist_rewards: np.ndarray,
    veh_veh_dist_rewards: np.ndarray,
) -> np.ndarray:
    del ag_data
    cfg = bridge.cfg_dataset
    num_agents, num_steps, _ = ag_rewards.shape
    all_rewards = np.zeros((num_agents, num_steps, 6), dtype=np.float32)
    all_rewards[:, :, :3] = ag_rewards[:, :, :3]
    all_rewards[:, :, 3] = ag_rewards[:, :, 3]
    all_rewards[:, :, 4] = veh_veh_dist_rewards * cfg.veh_veh_collision_rew_multiplier
    all_rewards[:, :, 5] = veh_edge_dist_rewards * cfg.veh_edge_collision_rew_multiplier
    return all_rewards


def process_preprocessed_data(bridge: Any, raw_data: Dict) -> Dict:
    ag_data = raw_data["ag_data"]
    ag_rewards = raw_data["ag_rewards"]
    veh_edge_dist_rewards = raw_data["veh_edge_dist_rewards"]
    veh_veh_dist_rewards = raw_data["veh_veh_dist_rewards"]
    road_points = raw_data["road_points"]
    road_types = raw_data["road_types"]

    all_rewards = compute_rewards(
        bridge,
        ag_data,
        ag_rewards,
        veh_edge_dist_rewards,
        veh_veh_dist_rewards,
    )
    rtgs = np.cumsum(all_rewards[:, ::-1], axis=1)[:, ::-1]
    return {
        "rtgs": rtgs,
        "road_points": road_points,
        "road_types": road_types,
        "ag_data": ag_data,
        "ag_rewards": ag_rewards,
        "filtered_ag_ids": raw_data.get("filtered_ag_ids", []),
    }


def get_available_scenario_ids(bridge: Any) -> List[str]:
    ensure_preprocessed_files_cache(bridge)
    return list(bridge._preprocessed_files_cache.keys())


def load_preprocessed_data_direct(
    bridge: Any,
    scenario_filename: str,
) -> Tuple[Optional[Dict], bool]:
    return load_preprocessed_data(bridge, scenario_filename)
