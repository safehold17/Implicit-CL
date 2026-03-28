"""Encoding helper functions for Nocturne CtrlSim levels."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..level import ScenarioLevel


def encode_level_to_string_array(
    current_level: Optional[ScenarioLevel],
    scenario_id_to_index: dict[str, int],
    normalize_per_vehicle_tilting: Callable[[tuple[int, ...]], tuple[int, ...]],
    per_vehicle_tilting_length: int,
    level_seed: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Encode the current level into the string-array format used by PLR buffers."""
    if current_level is None:
        values = [0, 0, 0, 0] + [0] * per_vehicle_tilting_length + [level_seed]
    else:
        scenario_idx = scenario_id_to_index.get(current_level.scenario_id, 0)
        normalized_per_vehicle_tilting = normalize_per_vehicle_tilting(
            current_level.per_vehicle_tilting
        )
        values = [
            scenario_idx,
            current_level.goal_tilt,
            current_level.veh_veh_tilt,
            current_level.veh_edge_tilt,
            *normalized_per_vehicle_tilting,
            current_level.seed,
        ]

    return np.array([str(value) for value in values], dtype=dtype)


def decode_level_from_string_array(
    encoding: np.ndarray,
    resolve_scenario_id: Callable[[int], str],
    normalize_per_vehicle_tilting: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> ScenarioLevel:
    """Decode the string-array format used by PLR buffers into a level."""
    (
        scenario_idx,
        goal_tilt,
        veh_veh_tilt,
        veh_edge_tilt,
        per_vehicle_tilting,
        seed,
    ) = ScenarioLevel.decode_encoding_fields(encoding)
    scenario_id = resolve_scenario_id(scenario_idx)
    normalized_per_vehicle_tilting = normalize_per_vehicle_tilting(per_vehicle_tilting)

    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=goal_tilt,
        veh_veh_tilt=veh_veh_tilt,
        veh_edge_tilt=veh_edge_tilt,
        per_vehicle_tilting=normalized_per_vehicle_tilting,
    )
