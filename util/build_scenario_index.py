#!/usr/bin/env python3
"""
Build Nocturne scenario index.

Usage:
    python util/build_scenario_index.py
"""
import json
import os
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import OmegaConf


CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "cfgs" / "data")
CONFIG_NAME = "scenario_index"


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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    scenario_cfg = cfg.scenario_index
    build_scenario_index(
        nocturne_data_dir=scenario_cfg.data_dir,
        output_path=scenario_cfg.output,
    )


if __name__ == "__main__":
    main()
