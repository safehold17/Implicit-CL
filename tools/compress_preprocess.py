from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tqdm import tqdm


COMPRESSED_FORMAT = "ctrlsim_preprocessed_compressed"
DEFAULT_INPUT_DATA_DIR = "/home/chen/data/preprocess/train"
DEFAULT_OUTPUT_DATA_DIR = "/home/chen/data/preprocess_compressed/train"


@dataclass(frozen=True)
class CompressionConfig:
    pos_target_achieved_rew_multiplier: float = 10.0
    pos_goal_shaped_min: float = 0.0
    pos_goal_shaped_max: float = 0.2
    veh_veh_collision_rew_multiplier: float = 10.0
    veh_edge_collision_rew_multiplier: float = 10.0
    dist_to_road_edge_scaling_factor: float = 15.0
    remove_shaped_goal: bool = True
    remove_shaped_veh_reward: bool = False
    remove_shaped_edge_reward: bool = False


def compute_eval_rewards(
    ag_data: np.ndarray,
    ag_rewards: np.ndarray,
    veh_edge_dist_rewards: np.ndarray,
    veh_veh_dist_rewards: np.ndarray,
    config: CompressionConfig,
) -> np.ndarray:
    """Match the reward aggregation used by RLWaymoDatasetCtRLSim(mode='eval').

    The legacy preprocess files do not store RTG directly. Instead, the eval path
    reconstructs RTG from per-step reward terms and then applies a reverse-time
    cumulative sum. This helper mirrors that reward reconstruction exactly so the
    compressed file keeps the same initial RTG as the legacy reader.

    Inputs:
    - ``ag_data`` provides the per-agent existence flag in the last channel.
    - ``ag_rewards`` stores the original 8 reward terms generated during
    preprocess:
        0: position target achieved
        1: heading target achieved
        2: speed target achieved
        3: shaped position-to-goal reward
        4: shaped speed-to-goal reward
        5: shaped heading-to-goal reward
        6: vehicle-vehicle collision indicator
        7: vehicle-edge collision indicator
    - ``veh_edge_dist_rewards`` and ``veh_veh_dist_rewards`` are the additional
        shaped distance terms computed from geometry after preprocess.

    Output:
    - A 5-D reward tensor per timestep with the same semantics used by the eval
        dataset code:
        0: goal position reward
        1: goal heading reward
        2: goal speed reward
        3: vehicle-vehicle reward
        4: vehicle-edge reward
    """
    # ``ag_data[..., -1]`` is the existence mask. Legacy code zeroes out every
    # reward term after an agent disappears so missing timesteps do not
    # contribute to RTG.
    ag_existence = ag_data[:, :, -1:]

    # processed_rewards dimensions: (num_agents, num_steps, 8)
    processed_rewards = np.asarray(ag_rewards, dtype=np.float32)

    # Goal position reward always includes the sparse target-achieved term.
    # When ``remove_shaped_goal`` is enabled, this matches the common project
    # setting and drops the dense distance-to-goal shaping entirely.
    # Otherwise, it reuses the same clipping and normalization as the legacy
    # dataset reader to fold shaped goal distance into the position reward.
    if config.remove_shaped_goal:
        goal_pos_rewards = (
            processed_rewards[:, :, 0] * config.pos_target_achieved_rew_multiplier
        )
    else:
        goal_pos_rewards = (
            processed_rewards[:, :, 0] * config.pos_target_achieved_rew_multiplier
            + (
                np.clip(
                    processed_rewards[:, :, 3],     #shaped position-to-goal reward
                    a_min=config.pos_goal_shaped_min,
                    a_max=config.pos_goal_shaped_max,
                )
                - config.pos_goal_shaped_max
            )
            * (1.0 / config.pos_goal_shaped_max)
        )
    goal_pos_rewards = goal_pos_rewards[:, :, np.newaxis] * ag_existence

    # Heading and speed rewards are each composed of:
    # - the sparse "target achieved" indicator
    # - the corresponding shaped term from ``ag_rewards``
    # Then they are masked by existence for the same reason as above.
    goal_heading_rewards = (
        processed_rewards[:, :, 1] + processed_rewards[:, :, 5]
    )[:, :, np.newaxis] * ag_existence
    goal_velocity_rewards = (
        processed_rewards[:, :, 2] + processed_rewards[:, :, 4]
    )[:, :, np.newaxis] * ag_existence

    # Vehicle-vehicle reward has two variants:
    # - sparse-only: keep only the collision penalty
    # - shaped: use the precomputed nearest-vehicle shaping term and subtract the
    #   collision penalty on top of it
    # This mirrors ``RLWaymoDataset.compute_rewards`` exactly.
    if config.remove_shaped_veh_reward:
        veh_veh_collision_rewards = (
            -1.0 * processed_rewards[:, :, 6] * config.veh_veh_collision_rew_multiplier
        )
    else:
        veh_veh_collision_rewards = (
            np.asarray(veh_veh_dist_rewards, dtype=np.float32)
            - processed_rewards[:, :, 6] * config.veh_veh_collision_rew_multiplier
        )
    veh_veh_collision_rewards = veh_veh_collision_rewards[:, :, np.newaxis] * ag_existence

    # Vehicle-edge reward follows the same pattern:
    # - sparse-only: only apply the edge collision penalty
    # - shaped: convert the geometric distance term into a bounded [0, 1] style
    #   shaping reward, then subtract the collision penalty
    # The distance term is multiplied by ``dist_to_road_edge_scaling_factor``
    # before clipping because the stored preprocess value is still in its raw
    # normalized form.
    if config.remove_shaped_edge_reward:
        veh_edge_collision_rewards = (
            -1.0 * processed_rewards[:, :, 7] * config.veh_edge_collision_rew_multiplier
        )
    else:
        veh_edge_collision_rewards = (
            np.clip(
                np.abs(np.asarray(veh_edge_dist_rewards, dtype=np.float32))
                * config.dist_to_road_edge_scaling_factor,
                a_min=0.0,
                a_max=5.0,
            )
            / 5.0
            - processed_rewards[:, :, 7] * config.veh_edge_collision_rew_multiplier
        )
    veh_edge_collision_rewards = veh_edge_collision_rewards[:, :, np.newaxis] * ag_existence

    # Concatenate into the exact 5-D layout expected by the eval RTG path:
    # [goal_pos, goal_heading, goal_speed, veh_veh, veh_edge]
    return np.concatenate(
        (
            goal_pos_rewards,
            goal_heading_rewards,
            goal_velocity_rewards,
            veh_veh_collision_rewards,
            veh_edge_collision_rewards,
        ),
        axis=-1,
    ).astype(np.float32)


def compute_eval_rtgs(raw_data: dict[str, Any], config: CompressionConfig) -> np.ndarray:
    rewards = compute_eval_rewards(
        ag_data=np.asarray(raw_data["ag_data"], dtype=np.float32),
        ag_rewards=np.asarray(raw_data["ag_rewards"], dtype=np.float32),
        veh_edge_dist_rewards=np.asarray(raw_data["veh_edge_dist_rewards"], dtype=np.float32),
        veh_veh_dist_rewards=np.asarray(raw_data["veh_veh_dist_rewards"], dtype=np.float32),
        config=config,
    )
    return np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1].astype(np.float32)


def compress_preprocessed_data(
    scenario_id: str,
    raw_data: dict[str, Any],
    config: CompressionConfig,
) -> dict[str, Any]:
    rtgs = compute_eval_rtgs(raw_data, config)

    return {
        "format": COMPRESSED_FORMAT,
        "scenario_id": scenario_id,
        "road_points": np.asarray(raw_data["road_points"], dtype=np.float32),
        "road_types": np.asarray(raw_data["road_types"]),
        "rtgs": rtgs[:, :1].astype(np.float32),
    }


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("*_physics.pkl"))


def compress_preprocess_file(
    input_path: Path,
    output_path: Path,
    config: CompressionConfig,
) -> tuple[int, int]:
    with open(input_path, "rb") as file:
        raw_data = pickle.load(file)

    compressed = compress_preprocessed_data(
        scenario_id=input_path.stem,
        raw_data=raw_data,
        config=config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(compressed, file, protocol=pickle.HIGHEST_PROTOCOL)

    return input_path.stat().st_size, output_path.stat().st_size


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress ctrl-sim preprocess pickle files.")
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default=DEFAULT_INPUT_DATA_DIR,
        help="Directory containing legacy *_physics.pkl files.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory for compressed *_physics.pkl files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of files to process.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_data_dir)
    output_dir = Path(args.output_data_dir)
    config = CompressionConfig()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input preprocess directory does not exist: {input_dir}")

    input_files = list(iter_input_files(input_dir))
    if args.limit is not None:
        input_files = input_files[: args.limit]
    pending_input_files = [
        input_path
        for input_path in input_files
        if not (output_dir / input_path.name).exists()
    ]
    skipped_existing_count = len(input_files) - len(pending_input_files)

    total_input_bytes = 0
    total_output_bytes = 0
    processed_count = 0
    failed_files: list[Path] = []
    for input_path in tqdm(pending_input_files):
        output_path = output_dir / input_path.name
        try:
            input_bytes, output_bytes = compress_preprocess_file(
                input_path=input_path,
                output_path=output_path,
                config=config,
            )
        except (OSError, EOFError, pickle.UnpicklingError) as exc:
            print(f"Warning: skipping invalid preprocess file {input_path}: {exc}")
            failed_files.append(input_path)
            continue
        total_input_bytes += input_bytes
        total_output_bytes += output_bytes
        processed_count += 1

    print(
        "Compressed "
        f"{processed_count} files from {input_dir} to {output_dir}. "
        f"size: {total_input_bytes} -> {total_output_bytes} bytes. "
        f"skipped existing: {skipped_existing_count}. "
        f"skipped invalid: {len(failed_files)}."
    )
    if failed_files:
        print("Skipped files:")
        for failed_path in failed_files:
            print(f"  {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
