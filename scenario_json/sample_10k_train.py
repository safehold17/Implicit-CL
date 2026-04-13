#!/usr/bin/env python3
"""Sample 10k train scenarios and build aligned JSON files."""

import json
import random
from pathlib import Path
from typing import Any


SOURCE_SCENARIO_INDEX_PATH = Path(
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/"
    "scenarios_index_filtered_train.json"
)
SOURCE_VEHICLE_MAP_PATH = Path(
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/"
    "vehicle_map_filtered_train.json"
)
OUTPUT_SCENARIO_INDEX_PATH = Path(__file__).with_name(
    "scenario_index_10k_train.json"
)
OUTPUT_VEHICLE_MAP_PATH = Path(__file__).with_name("vehicle_map_10k_train.json")
SAMPLE_SIZE = 10_000
RANDOM_SEED = 26


def has_ego_vehicle_id(scene_info: dict[str, Any]) -> bool:
    """Return whether a scene has a usable ego vehicle ID."""
    ego_vehicle_id = scene_info.get("ego_vehicle_id")
    return ego_vehicle_id is not None and ego_vehicle_id != "" and ego_vehicle_id != -1


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    """Save a payload as pretty JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def sample_scenario_ids(
    scenario_ids: list[str],
    sample_size: int,
    seed: int,
) -> list[str]:
    """Sample scenario IDs reproducibly while preserving source order."""
    sampled_ids = set(random.Random(seed).sample(scenario_ids, sample_size))
    return [scenario_id for scenario_id in scenario_ids if scenario_id in sampled_ids]


def build_sampled_files(
    scenario_index: dict[str, Any],
    vehicle_map: dict[str, Any],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build aligned sampled scenario index and vehicle map payloads."""
    scenario_ids = [
        scenario_id
        for scenario_id in scenario_index["scenario_ids"]
        if scenario_id in vehicle_map and has_ego_vehicle_id(vehicle_map[scenario_id])
    ]
    sampled_scenario_ids = sample_scenario_ids(scenario_ids, sample_size, seed)
    sampled_vehicle_map = {
        scenario_id: vehicle_map[scenario_id]
        for scenario_id in sampled_scenario_ids
    }
    sampled_scenario_index = {
        "version": scenario_index["version"],
        "source_dir": scenario_index["source_dir"],
        "total_scenarios": len(sampled_scenario_ids),
        "scenario_ids": sampled_scenario_ids,
    }
    return sampled_scenario_index, sampled_vehicle_map


def main() -> None:
    """Generate 10k train scenario index and vehicle map JSON files."""
    scenario_index = load_json(SOURCE_SCENARIO_INDEX_PATH)
    vehicle_map = load_json(SOURCE_VEHICLE_MAP_PATH)

    sampled_scenario_index, sampled_vehicle_map = build_sampled_files(
        scenario_index=scenario_index,
        vehicle_map=vehicle_map,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED,
    )

    save_json(OUTPUT_SCENARIO_INDEX_PATH, sampled_scenario_index)
    save_json(OUTPUT_VEHICLE_MAP_PATH, sampled_vehicle_map)

    print(f"Saved scenario index: {OUTPUT_SCENARIO_INDEX_PATH}")
    print(f"Saved vehicle map: {OUTPUT_VEHICLE_MAP_PATH}")
    print(f"Sampled scenarios: {len(sampled_scenario_index['scenario_ids'])}")


if __name__ == "__main__":
    main()
