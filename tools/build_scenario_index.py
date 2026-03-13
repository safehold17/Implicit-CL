#!/usr/bin/env python3
"""
Build Nocturne scenario index.

Usage:
    python tools/build_scenario_index.py --data_dir ... --output ...
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional

class ScenarioIndex:
    """
    Scenario index manager.

    Provides bidirectional mapping between scenario_id and index.
    """

    def __init__(self, index_path: str):
        """
        Load index from file.

        Args:
            index_path: Path to the index JSON file.
        """
        with open(index_path, "r") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0")
        self.scenario_ids = data["scenario_ids"]

        self.scenario_id_to_index = {
            scenario_id: i for i, scenario_id in enumerate(self.scenario_ids)
        }
        self.index_to_scenario_id = {
            i: scenario_id for i, scenario_id in enumerate(self.scenario_ids)
        }

    def __len__(self) -> int:
        return len(self.scenario_ids)


def build_scenario_index(
    nocturne_data_dir: str,
    output_path: str,
    valid_files_json: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Scan Nocturne data directory and build scenario index.

    Args:
        nocturne_data_dir: Nocturne scenario file directory.
        output_path: Output JSON file path.
        valid_files_json: Optional valid files list JSON path.
        verbose: Whether to print progress.

    Returns:
        Index data dict.
    """
    data_path = Path(nocturne_data_dir)

    if valid_files_json and os.path.exists(valid_files_json):
        with open(valid_files_json, "r") as f:
            valid_files = json.load(f)
        scenario_ids = sorted([Path(f).stem for f in valid_files])
        if verbose:
            print(f"📋 Using valid_files.json: {len(scenario_ids)} scenarios")
    else:
        scenario_files = [
            f
            for f in data_path.glob("*.json")
            if not f.name.startswith("valid_") and not f.name.endswith("_index.json")
        ]
        scenario_ids = sorted([f.stem for f in scenario_files])
        if verbose:
            print(f"🔍 Scanning directory: {len(scenario_ids)} scenarios found")

    index_data = {
        "version": "1.0",
        "source_dir": str(data_path.absolute()),
        "total_scenarios": len(scenario_ids),
        "scenario_ids": scenario_ids,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index_data, f, indent=2)

    if verbose:
        print(f"✅ Built index with {len(scenario_ids)} scenarios")
        print(f"   Saved to: {output_path}")

    return index_data


def _get_nocturne_ctrlsim_defaults() -> dict:
    from arguments import NOCTURNE_CTRLSIM_DEFAULTS

    return NOCTURNE_CTRLSIM_DEFAULTS


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    defaults = _get_nocturne_ctrlsim_defaults()
    parser = argparse.ArgumentParser(description="Build Nocturne scenario index")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=defaults["scenario_data_dir"],
        help="Nocturne scenario data directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=defaults["scenario_index_path"],
        help="Output JSON path for the scenario index.",
    )
    parser.add_argument(
        "--valid_files_json",
        type=str,
        default=None,
        help="Optional valid_files.json path used to filter scenario ids.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress prints.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    build_scenario_index(
        nocturne_data_dir=args.data_dir,
        output_path=args.output,
        valid_files_json=args.valid_files_json,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
