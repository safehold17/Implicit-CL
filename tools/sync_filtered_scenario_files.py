#!/usr/bin/env python3
"""
Prune filtered scenario index and vehicle map files to available scenarios.

For each split, a scenario is kept only if it:
- exists in the preprocess directory
- exists in the scenario JSON directory
- exists in the current scenario index JSON
- exists in the current vehicle map JSON

The script updates the JSON files in place.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitPaths:
    name: str
    scenario_index_json: Path
    vehicle_map_json: Path
    preprocess_dir: Path
    scenario_dir: Path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _collect_preprocess_scenario_ids(preprocess_dir: Path) -> set[str]:
    physics_files = sorted(preprocess_dir.glob("*_physics.pkl"))
    if physics_files:
        return {path.name[: -len("_physics.pkl")] for path in physics_files}
    return {path.stem for path in preprocess_dir.iterdir() if path.is_file()}


def _collect_scenario_json_ids(scenario_dir: Path) -> set[str]:
    return {
        path.stem
        for path in scenario_dir.glob("*.json")
        if not path.name.startswith("valid_") and not path.name.endswith("_index.json")
    }


def _prune_split(split: SplitPaths) -> dict[str, int]:
    index_data = _load_json(split.scenario_index_json)
    vehicle_map = _load_json(split.vehicle_map_json)

    scenario_ids = index_data.get("scenario_ids")
    if not isinstance(scenario_ids, list):
        raise ValueError(
            f"{split.scenario_index_json} is missing a valid 'scenario_ids' list"
        )
    if not isinstance(vehicle_map, dict):
        raise ValueError(f"{split.vehicle_map_json} must contain a JSON object")

    preprocess_ids = _collect_preprocess_scenario_ids(split.preprocess_dir)
    scenario_file_ids = _collect_scenario_json_ids(split.scenario_dir)
    vehicle_map_ids = set(vehicle_map.keys())
    keep_ids = [
        scenario_id
        for scenario_id in scenario_ids
        if scenario_id in preprocess_ids
        and scenario_id in scenario_file_ids
        and scenario_id in vehicle_map_ids
    ]

    pruned_vehicle_map = {
        scenario_id: vehicle_map[scenario_id]
        for scenario_id in keep_ids
    }
    pruned_index = {
        "version": index_data.get("version", "1.0"),
        "source_dir": str(split.scenario_dir.resolve()),
        "total_scenarios": len(keep_ids),
        "scenario_ids": keep_ids,
    }

    _save_json(split.scenario_index_json, pruned_index)
    _save_json(split.vehicle_map_json, pruned_vehicle_map)

    return {
        "original_index_count": len(scenario_ids),
        "original_vehicle_map_count": len(vehicle_map),
        "preprocess_count": len(preprocess_ids),
        "scenario_file_count": len(scenario_file_ids),
        "kept_count": len(keep_ids),
        "removed_index_count": len(scenario_ids) - len(keep_ids),
        "removed_vehicle_map_count": len(vehicle_map) - len(pruned_vehicle_map),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update filtered scenario index and vehicle map JSON files so they only "
            "contain scenarios present in both the preprocess and scenario directories."
        )
    )
    parser.add_argument(
        "--train-scenario-index-json",
        type=Path,
        default=Path("/home/chen/data/scenarios_index_filtered_train.json"),
        help="Path to scenario_index_filtered_train.json",
    )
    parser.add_argument(
        "--train-vehicle-map-json",
        type=Path,
        default=Path("/home/chen/data/vehicle_map_filtered_train.json"),
        help="Path to vehicle_map_filtered_train.json",
    )
    parser.add_argument(
        "--train-preprocess-dir",
        type=Path,
        default=Path("/home/chen/data/preprocess_compressed/train"),
        help="Train preprocess directory",
    )
    parser.add_argument(
        "--train-scenario-dir",
        type=Path,
        default=Path("/home/chen/data/nocturne_waymo/formatted_json_v2_no_tl_train"),
        help="Train scenario JSON directory",
    )
    parser.add_argument(
        "--valid-scenario-index-json",
        type=Path,
        default=Path("/home/chen/data/scenarios_index_filtered_valid.json"),
        help="Path to scenario_index_filtered_valid.json",
    )
    parser.add_argument(
        "--valid-vehicle-map-json",
        type=Path,
        default=Path("/home/chen/data/vehicle_map_filtered_valid.json"),
        help="Path to vehicle_map_filtered_valid.json",
    )
    parser.add_argument(
        "--valid-preprocess-dir",
        type=Path,
        default=Path("/home/chen/data/preprocess_compressed/test"),
        help="Valid preprocess directory",
    )
    parser.add_argument(
        "--valid-scenario-dir",
        type=Path,
        default=Path("/home/chen/data/nocturne_waymo/formatted_json_v2_no_tl_valid"),
        help="Valid scenario JSON directory",
    )
    return parser.parse_args(argv)


def _validate_paths(split: SplitPaths) -> None:
    for path in (split.scenario_index_json, split.vehicle_map_json):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    for path in (split.preprocess_dir, split.scenario_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Required directory not found: {path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    splits = (
        SplitPaths(
            name="train",
            scenario_index_json=args.train_scenario_index_json,
            vehicle_map_json=args.train_vehicle_map_json,
            preprocess_dir=args.train_preprocess_dir,
            scenario_dir=args.train_scenario_dir,
        ),
        SplitPaths(
            name="valid",
            scenario_index_json=args.valid_scenario_index_json,
            vehicle_map_json=args.valid_vehicle_map_json,
            preprocess_dir=args.valid_preprocess_dir,
            scenario_dir=args.valid_scenario_dir,
        ),
    )

    for split in splits:
        _validate_paths(split)
        stats = _prune_split(split)
        print(
            f"[{split.name}] kept {stats['kept_count']} scenarios, "
            f"filtered {stats['removed_index_count']} scenarios "
            f"(index -{stats['removed_index_count']}, "
            f"vehicle_map -{stats['removed_vehicle_map_count']})"
        )


if __name__ == "__main__":
    main()
