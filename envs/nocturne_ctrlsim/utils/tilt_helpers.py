"""Tilt and level-parameter helper functions for Nocturne CtrlSim."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..core.level import ScenarioLevel


def round_clipped_tilt(value: float, tilt_range: tuple[float, float]) -> int:
    """Round a tilt value after clipping it into the configured tilt range."""
    return int(round(float(np.clip(value, *tilt_range))))


def map_action_to_tilt(
    tilt_action: float,
    tilt_range: tuple[float, float],
) -> int:
    """Map an action value from [-1, 1] into the configured tilt range."""
    tilt_scale = (tilt_range[1] - tilt_range[0]) / 2.0
    tilt_value = float(tilt_action) * tilt_scale
    return round_clipped_tilt(tilt_value, tilt_range)


def sample_policy_reweighting_ego_tilt(
    *,
    use_policy_reweighting: bool,
    opponent_runtime_mode: str,
    tilt_range: tuple[float, float],
    rng: Any,
) -> tuple[int, int, int]:
    """为 policy reweighting 采样 ego-only 三维 tilt；该链路与 tilting_mode 无关。

    Sample the ego-only 3D tilt used by policy reweighting. This helper encodes
    the exact project semantics for the reweighting path.
    """
    if not use_policy_reweighting or str(opponent_runtime_mode) != "normal":
        return (0, 0, 0)
    goal_tilt, veh_veh_tilt, veh_edge_tilt, _ = sample_random_tilt_components(
        tilting_mode="global",
        per_vehicle_tilting_length=0,
        tilt_range=tilt_range,
        rng=rng,
    )
    return (goal_tilt, veh_veh_tilt, veh_edge_tilt)


def init_level_params_vec(
    tilting_mode: str,
    per_vehicle_tilting_length: int,
) -> list[int]:
    """Build the default ``level_params_vec`` layout for the tilting mode."""
    if tilting_mode == "per_vehicle":
        return [0, 0, 0, 0] + [0] * per_vehicle_tilting_length
    if tilting_mode == "none":
        return [0]
    return [0, 0, 0, 0]


def apply_adversary_tilting_action(
    level_params_vec: list[int],
    tilting_mode: str,
    per_vehicle_tilting_length: int,
    runtime_mode: str,
    action_vec: np.ndarray,
    tilt_range: tuple[float, float],
) -> None:
    """Apply adversary tilt action into ``level_params_vec``."""
    if tilting_mode == "global":
        tilt_values = [0, 0, 0]
        if runtime_mode == "normal":
            tilt_values = [
                map_action_to_tilt(action_vec[1], tilt_range),
                map_action_to_tilt(action_vec[2], tilt_range),
                map_action_to_tilt(action_vec[3], tilt_range),
            ]
        level_params_vec[1:4] = tilt_values
        return

    if tilting_mode == "per_vehicle":
        if runtime_mode != "normal":
            level_params_vec[1:4] = [0, 0, 0]
            level_params_vec[4 : 4 + per_vehicle_tilting_length] = [
                0
            ] * per_vehicle_tilting_length
            return

        per_vehicle_values = [
            map_action_to_tilt(v, tilt_range)
            for v in action_vec[1 : 1 + per_vehicle_tilting_length]
        ]
        level_params_vec[4 : 4 + per_vehicle_tilting_length] = per_vehicle_values
        return

    if tilting_mode == "none":
        return

    raise ValueError(f"Unsupported tilting_mode: {tilting_mode}")


def sample_random_tilt_components(
    tilting_mode: str,
    per_vehicle_tilting_length: int,
    tilt_range: tuple[float, float],
    rng: Any,
    enable_goal_tilt: bool = True,
    enable_veh_veh_tilt: bool = True,
    enable_veh_edge_tilt: bool = True,
) -> tuple[int, int, int, tuple[int, ...]]:
    """Sample random tilt components for the current tilting mode."""
    zero_per_vehicle = (0,) * per_vehicle_tilting_length

    if tilting_mode == "global":
        return (
            (
                round_clipped_tilt(rng.uniform(*tilt_range), tilt_range)
                if enable_goal_tilt
                else 0
            ),
            (
                round_clipped_tilt(rng.uniform(*tilt_range), tilt_range)
                if enable_veh_veh_tilt
                else 0
            ),
            (
                round_clipped_tilt(rng.uniform(*tilt_range), tilt_range)
                if enable_veh_edge_tilt
                else 0
            ),
            zero_per_vehicle,
        )

    if tilting_mode == "none":
        return (0, 0, 0, zero_per_vehicle)

    enabled_dims = (
        enable_goal_tilt,
        enable_veh_veh_tilt,
        enable_veh_edge_tilt,
    )
    per_vehicle_values = []
    for index in range(per_vehicle_tilting_length):
        if enabled_dims[index % 3]:
            per_vehicle_values.append(
                round_clipped_tilt(rng.uniform(*tilt_range), tilt_range)
            )
        else:
            per_vehicle_values.append(0)

    return (
        0,
        0,
        0,
        tuple(per_vehicle_values),
    )


def encode_level_params(
    scenario_idx: int,
    level: ScenarioLevel,
    tilting_mode: str,
) -> list[int]:
    """Encode the current level into ``level_params_vec`` format."""
    if tilting_mode == "per_vehicle":
        return [
            scenario_idx,
            0,
            0,
            0,
            *level.per_vehicle_tilting,
        ]

    if tilting_mode == "none":
        return [
            scenario_idx,
        ]

    return [
        scenario_idx,
        level.goal_tilt,
        level.veh_veh_tilt,
        level.veh_edge_tilt,
    ]


def mutation_dims(mutation_mode: str, rng: Any) -> list[int]:
    """Return the tilt dimensions selected for mutation."""
    if mutation_mode == "one":
        return [rng.randint(0, 3)]
    return [0, 1, 2]


def mutate_global_level(
    level: ScenarioLevel,
    rng: Any,
    mutation_mode: str,
    mutation_range: float,
    tilt_range: tuple[float, float],
) -> ScenarioLevel:
    """Mutate global tilt fields for global and ego tilting modes."""
    params = ["goal_tilt", "veh_veh_tilt", "veh_edge_tilt"]
    mutations = {}
    dims = mutation_dims(mutation_mode, rng)
    if mutation_mode == "one":
        dim = dims[0]
        param = params[dim]
        delta = rng.uniform(-mutation_range, mutation_range)
        mutations[param] = round_clipped_tilt(getattr(level, param) + delta, tilt_range)
        return replace(level, **mutations)

    deltas = rng.uniform(-mutation_range, mutation_range, size=3)
    for param, delta in zip(params, deltas):
        mutations[param] = round_clipped_tilt(getattr(level, param) + delta, tilt_range)
    return replace(level, **mutations)


def mutate_per_vehicle_level(
    level: ScenarioLevel,
    rng: Any,
    mutation_mode: str,
    mutation_range: float,
    tilt_range: tuple[float, float],
    opponent_k: int,
    per_vehicle_tilting_length: int,
    normalize_per_vehicle_tilting: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> ScenarioLevel:
    """Mutate per-vehicle tilt fields for per_vehicle tilting mode."""
    per = list(normalize_per_vehicle_tilting(level.per_vehicle_tilting))
    if mutation_mode == "one":
        dim = mutation_dims(mutation_mode, rng)[0]
        deltas = rng.uniform(-mutation_range, mutation_range, size=opponent_k)
        for idx, delta in enumerate(deltas):
            per_idx = idx * 3 + dim
            per[per_idx] = round_clipped_tilt(per[per_idx] + delta, tilt_range)
        return replace(level, per_vehicle_tilting=tuple(per))

    deltas = rng.uniform(
        -mutation_range,
        mutation_range,
        size=per_vehicle_tilting_length,
    )
    for idx, delta in enumerate(deltas):
        per[idx] = round_clipped_tilt(per[idx] + delta, tilt_range)
    return replace(level, per_vehicle_tilting=tuple(per))
