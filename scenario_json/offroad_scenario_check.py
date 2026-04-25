#!/usr/bin/env python3
# Run:
# source /home/chen/miniconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate dcd-ctrlsim
# python scenario_json/offroad_scenario_check.py --process 8
"""Check offroad scenarios by replaying GT state in Nocturne."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch


def disable_torch_compile() -> None:
    """Replace torch.compile with an identity wrapper for worker safety."""
    if getattr(torch, "_offroad_scenario_check_compile_disabled", False):
        return

    def _identity_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        return model

    torch.compile = _identity_compile
    torch._offroad_scenario_check_compile_disabled = True


disable_torch_compile()


DEFAULT_LOG_DIR = Path(
    "/media/chen/Dataset/logs/"
    "dcd/steps4096000-proc16-roll256-plr1-edit1-tiltper_vehicle-kl1-prw0_0"
)
DEFAULT_META_PATH = DEFAULT_LOG_DIR / "meta.json"
DEFAULT_SCENARIO_INDEX_PATH = Path(
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/scenarios_index_filtered_valid.json"
)
DEFAULT_VEHICLE_MAP_PATH = Path(
    "/media/chen/Dataset/ctrlsim_dataset/preparation_file/vehicle_map_filtered_valid.json"
)
DEFAULT_SCENARIO_DATA_DIR = Path(
    "/media/chen/Dataset/ctrlsim_dataset/scenario_data/formatted_json_v2_no_tl_valid"
)
DEFAULT_PREPROCESS_DIR = Path(
    "/media/chen/Dataset/ctrlsim_dataset/compressed_preprocessed_data/test"
)
DEFAULT_OUTPUT_JSON_PATH = Path(__file__).with_name("offroad_scenario_check_result_valid.json")
PARTIAL_SAVE_INTERVAL = 96

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CTRLSIM_ROOT = PROJECT_ROOT / "ctrlsim"
for _path in (PROJECT_ROOT, CTRLSIM_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from envs.nocturne_ctrlsim.adversarial import NocturneCtrlSimAdversarial
from envs.nocturne_ctrlsim.core.episode_runtime import _update_step_ego_cache
from envs.nocturne_ctrlsim.core.level import build_zero_tilt_level
from envs.nocturne_ctrlsim.student.observation_action import refresh_student_vehicle_cache
from envs.nocturne_ctrlsim.student.student_reward import compute_student_reward


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    """Save a payload as pretty JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_meta_args(meta_path: Path) -> dict[str, Any]:
    """Load run arguments from meta.json."""
    meta = load_json(meta_path)
    return meta.get("args", meta)


def build_gt_replay_env(
    meta_path: Path,
    scenario_index_path: Path,
    vehicle_map_path: Path,
    scenario_data_dir: Path,
    preprocess_dir: Path,
) -> NocturneCtrlSimAdversarial:
    """Build one CPU env for GT-state replay."""
    args = load_meta_args(meta_path)
    return NocturneCtrlSimAdversarial(
        scenario_index_path=str(scenario_index_path),
        opponent_checkpoint=args["opponent_checkpoint"],
        scenario_data_dir=str(scenario_data_dir),
        preprocess_dir=str(preprocess_dir),
        vehicle_map_path=str(vehicle_map_path),
        device="cpu",
        tilting_mode=args["tilting_mode"],
        mutation_mode=args["mutation_mode"],
        student_accel_discretization=args["student_accel_discretization"],
        student_steer_discretization=args["student_steer_discretization"],
        obs_dim=None,
        remove_background_vehicles=args["remove_background_vehicles"],
        show_vehicle_ids=args["show_vehicle_ids"],
        show_tilting_params=args["show_tilting_params"],
        show_ego_vehicle_selection=args["show_ego_vehicle_selection"],
        opponent_runtime_mode="disable",
        inference_precision="fp32",
    )


def get_max_replay_steps(env: NocturneCtrlSimAdversarial) -> int:
    """Return the longest GT trajectory length in the current env."""
    gt_data_dict = getattr(env, "_gt_data_dict", {})
    if not gt_data_dict:
        return 0
    return max(len(data["traj"]) for data in gt_data_dict.values())


def replay_all_vehicles_with_gt_state(
    env: NocturneCtrlSimAdversarial,
    max_replay_steps: int,
) -> int:
    """Replay one scenario by aligning every vehicle to GT state."""
    veh_by_id = {veh.getID(): veh for veh in env.vehicles}
    for step in range(max_replay_steps):
        env.current_step = step
        env._collision_occurred = False
        env._offroad_occurred = False

        for veh_id, veh in veh_by_id.items():
            gt_traj = env._gt_data_dict[veh_id]["traj"]
            if step >= len(gt_traj) or not bool(gt_traj[step][4]):
                veh.setPosition(-1000000.0, -1000000.0)
                veh.setSpeed(0.0)
                continue

            veh.setPosition(float(gt_traj[step][0]), float(gt_traj[step][1]))
            veh.setHeading(float(gt_traj[step][2]))
            veh.setSpeed(float(gt_traj[step][3]))

        refresh_student_vehicle_cache(env)
        _update_step_ego_cache(env)
        compute_student_reward(env)

        if env._collision_occurred:
            env._episode_collision_occurred = True
        if env._goal_reached:
            env._episode_goal_reached = True
        if env._offroad_occurred:
            env._episode_offroad_occurred = True
            return step

    return max_replay_steps - 1 if max_replay_steps > 0 else 0


def analyze_one_scenario(
    scene_id: str,
    meta_path: Path,
    scenario_index_path: Path,
    vehicle_map_path: Path,
    scenario_data_dir: Path,
    preprocess_dir: Path,
) -> dict[str, Any]:
    """Replay one scenario and report episode offroad occurrence."""
    env = build_gt_replay_env(
        meta_path=meta_path,
        scenario_index_path=scenario_index_path,
        vehicle_map_path=vehicle_map_path,
        scenario_data_dir=scenario_data_dir,
        preprocess_dir=preprocess_dir,
    )
    try:
        level = build_zero_tilt_level(
            scenario_id=scene_id,
            seed=1,
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
        )
        env.reset_to_level(level)
        max_replay_steps = get_max_replay_steps(env)
        final_step = replay_all_vehicles_with_gt_state(env, max_replay_steps)
        return {
            "scene_id": scene_id,
            "ego_vehicle_id": int(env.ego_vehicle.getID()) if env.ego_vehicle else None,
            "replay_step_count": int(max_replay_steps),
            "max_replay_steps": int(max_replay_steps),
            "offroad_occurred": bool(getattr(env, "_episode_offroad_occurred", False)),
            "final_step": int(final_step),
        }
    finally:
        env.close()


def analyze_one_scenario_worker(
    task: tuple[str, str, str, str, str, str],
) -> dict[str, Any]:
    """Run one scenario analysis in a worker process."""
    (
        scene_id,
        meta_path_str,
        index_path_str,
        vehicle_map_path_str,
        scenario_dir_str,
        preprocess_dir_str,
    ) = task
    return analyze_one_scenario(
        scene_id=scene_id,
        meta_path=Path(meta_path_str),
        scenario_index_path=Path(index_path_str),
        vehicle_map_path=Path(vehicle_map_path_str),
        scenario_data_dir=Path(scenario_dir_str),
        preprocess_dir=Path(preprocess_dir_str),
    )


def analyze_offroad_scenarios(
    meta_path: Path,
    scenario_index_path: Path,
    vehicle_map_path: Path,
    scenario_data_dir: Path,
    preprocess_dir: Path,
    output_json_path: Path,
    process: int,
    max_scenarios: int | None,
) -> dict[str, Any]:
    """Compute offroad stats for the selected scenario set."""
    if process < 1:
        raise ValueError("process must be at least 1")

    scenario_index = load_json(scenario_index_path)
    vehicle_map = load_json(vehicle_map_path)
    scenario_ids = list(scenario_index["scenario_ids"])
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]

    missing_vehicle_map_ids: list[str] = []
    missing_ego_ids: list[str] = []
    scenario_tasks: list[tuple[str, str, str, str, str, str]] = []

    for scene_id in scenario_ids:
        scene_vehicle_info = vehicle_map.get(scene_id)
        if scene_vehicle_info is None:
            missing_vehicle_map_ids.append(scene_id)
            continue
        if scene_vehicle_info.get("ego_vehicle_id") is None:
            missing_ego_ids.append(scene_id)
            continue
        scenario_tasks.append(
            (
                scene_id,
                str(meta_path),
                str(scenario_index_path),
                str(vehicle_map_path),
                str(scenario_data_dir),
                str(preprocess_dir),
            )
        )

    executor: ProcessPoolExecutor | None = None
    if process == 1:
        results_iter = (
            analyze_one_scenario_worker(task)
            for task in tqdm(scenario_tasks, desc="Scanning scenarios")
        )
    else:
        mp_context = multiprocessing.get_context("fork")
        executor = ProcessPoolExecutor(max_workers=process, mp_context=mp_context)
        futures = [
            executor.submit(analyze_one_scenario_worker, task)
            for task in scenario_tasks
        ]
        results_iter = (
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Scanning scenarios",
            )
        )

    flagged_results: list[dict[str, Any]] = []
    completed_results = 0

    def flush_flagged_ids() -> None:
        """Persist current flagged scenario IDs to JSON."""
        save_json(
            output_json_path,
            [result["scene_id"] for result in flagged_results],
        )

    try:
        for result in results_iter:
            completed_results += 1
            if result["offroad_occurred"]:
                flagged_results.append(result)
            if completed_results % PARTIAL_SAVE_INTERVAL == 0:
                flush_flagged_ids()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    flush_flagged_ids()

    flagged_scenario_ids = [result["scene_id"] for result in flagged_results]
    total_scenarios = len(scenario_ids)
    return {
        "meta_path": str(meta_path),
        "scenario_index_path": str(scenario_index_path),
        "vehicle_map_path": str(vehicle_map_path),
        "scenario_data_dir": str(scenario_data_dir),
        "preprocess_dir": str(preprocess_dir),
        "scanned_scenarios": total_scenarios,
        "flagged_scenarios": len(flagged_results),
        "flagged_ratio": len(flagged_results) / total_scenarios if total_scenarios else 0.0,
        "flagged_scenario_ids": flagged_scenario_ids,
        "flagged_results": flagged_results,
        "missing_vehicle_map_ids": missing_vehicle_map_ids,
        "missing_ego_ids": missing_ego_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Check which scenarios become offroad under GT-state replay."
    )
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument(
        "--scenario-index-path",
        type=Path,
        default=DEFAULT_SCENARIO_INDEX_PATH,
    )
    parser.add_argument("--vehicle-map-path", type=Path, default=DEFAULT_VEHICLE_MAP_PATH)
    parser.add_argument(
        "--scenario-data-dir",
        type=Path,
        default=DEFAULT_SCENARIO_DATA_DIR,
    )
    parser.add_argument(
        "--preprocess-dir",
        type=Path,
        default=DEFAULT_PREPROCESS_DIR,
    )
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--max-scenarios", type=int, default=None)
    return parser


def main() -> None:
    """Run the offroad scenario checker."""
    args = build_parser().parse_args()
    report = analyze_offroad_scenarios(
        meta_path=args.meta_path,
        scenario_index_path=args.scenario_index_path,
        vehicle_map_path=args.vehicle_map_path,
        scenario_data_dir=args.scenario_data_dir,
        preprocess_dir=args.preprocess_dir,
        output_json_path=DEFAULT_OUTPUT_JSON_PATH,
        process=args.process,
        max_scenarios=args.max_scenarios,
    )
    print(f"Meta path: {report['meta_path']}")
    print(f"Scenario index: {report['scenario_index_path']}")
    print(f"Vehicle map: {report['vehicle_map_path']}")
    print(f"Scenario data dir: {report['scenario_data_dir']}")
    print(f"Preprocess dir: {report['preprocess_dir']}")
    print(f"Scanned scenarios: {report['scanned_scenarios']}")
    print(f"Flagged scenarios: {report['flagged_scenarios']}")
    print(f"Flagged ratio: {report['flagged_ratio']:.4%}")
    print(f"Missing vehicle-map entries: {len(report['missing_vehicle_map_ids'])}")
    print(f"Missing ego ids: {len(report['missing_ego_ids'])}")
    print(f"Saved flagged scenario ids: {DEFAULT_OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
