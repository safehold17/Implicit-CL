"""Level lifecycle helpers for the Nocturne-CtrlSim adversarial env."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..level import (
    ScenarioLevel,
    build_zero_tilt_level,
    normalize_level_for_tilting_mode,
    normalize_per_vehicle_tilting,
)
from . import scenario_pool as scenario_pool_service
from ..utils.encoding_helpers import decode_level_from_string_array
from ..utils.tilt_helpers import (
    encode_level_params,
    mutate_global_level,
    mutate_per_vehicle_level,
    sample_random_tilt_components,
)
from ..utils.vehicle_map_helpers import (
    format_vehicle_map_exhausted_error,
    format_vehicle_map_skip_warning,
    is_retryable_vehicle_map_error,
)


def coerce_level(
    env: Any,
    level: ScenarioLevel | str | np.ndarray,
) -> ScenarioLevel:
    """Convert supported level input formats into a ``ScenarioLevel``."""
    if isinstance(level, ScenarioLevel):
        return level
    if isinstance(level, str):
        return ScenarioLevel.from_level_string(level)
    if isinstance(level, np.ndarray):
        if level.dtype.kind == "U":
            return decode_string_encoding(env, level)
        return ScenarioLevel.from_encoding(level, env.index_to_scenario_id)
    raise TypeError(f"Unsupported level type for reset/mutation: {type(level)}")


def decode_string_encoding(env: Any, encoding: np.ndarray) -> ScenarioLevel:
    """Decode the string-array level encoding using the current pool mapping."""
    if env._scenario_pool_dirty:
        scenario_pool_service.rebuild_index_mappings(env)
    return decode_level_from_string_array(
        encoding=encoding,
        resolve_scenario_id=lambda scenario_idx: scenario_pool_service.resolve_scenario_id(
            env,
            scenario_idx,
        ),
        normalize_per_vehicle_tilting=lambda values: normalize_per_vehicle_tilting(
            values,
            env.per_vehicle_tilting_length,
        ),
    )


def create_level_from_params(env: Any, scenario_id: str) -> ScenarioLevel:
    """Create a level from the environment's current parameter vector."""
    if env.tilting_mode == "per_vehicle":
        per_vehicle_tilting = tuple(
            int(round(float(value)))
            for value in env.level_params_vec[4 : 4 + env.per_vehicle_tilting_length]
        )
        per_vehicle_tilting = normalize_per_vehicle_tilting(
            per_vehicle_tilting,
            env.per_vehicle_tilting_length,
        )
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=env.level_seed,
            goal_tilt=0,
            veh_veh_tilt=0,
            veh_edge_tilt=0,
            per_vehicle_tilting=per_vehicle_tilting,
        )

    if env.tilting_mode == "none":
        return build_zero_tilt_level(
            scenario_id=scenario_id,
            seed=env.level_seed,
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
        )

    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=env.level_seed,
        goal_tilt=env.level_params_vec[1],
        veh_veh_tilt=env.level_params_vec[2],
        veh_edge_tilt=env.level_params_vec[3],
    )


def sync_level_state(env: Any, level: ScenarioLevel) -> ScenarioLevel:
    """Synchronize ``current_level`` and ``level_params_vec`` for a level."""
    normalized_level = normalize_level_for_tilting_mode(
        level=level,
        tilting_mode=env.tilting_mode,
        per_vehicle_tilting_length=env.per_vehicle_tilting_length,
    )
    scenario_idx = env.scenario_id_to_index.get(normalized_level.scenario_id, 0)
    env.level_params_vec = encode_level_params(
        scenario_idx=scenario_idx,
        level=normalized_level,
        tilting_mode=env.tilting_mode,
    )
    env._set_current_level(normalized_level)
    return normalized_level


def initialize_level_with_fallback(
    env: Any,
    start_idx: int,
    create_level: Callable[[str], ScenarioLevel],
    context: str,
) -> ScenarioLevel:
    """Initialize a level and skip retryable bad scenarios when needed."""
    if env._scenario_pool_dirty:
        scenario_pool_service.rebuild_index_mappings(env)

    num_scenarios = len(env.scenario_ids)
    last_error: Exception | None = None
    last_scenario_id: str | None = None

    for offset in range(num_scenarios):
        scenario_idx = (start_idx + offset) % num_scenarios
        scenario_id = scenario_pool_service.resolve_scenario_id(env, scenario_idx)
        level = sync_level_state(env, create_level(scenario_id))

        try:
            env._initialize_simulation()
            return level
        except Exception as error:
            if not is_retryable_vehicle_map_error(error):
                raise
            last_error = error
            last_scenario_id = scenario_id
            if offset < num_scenarios - 1:
                print(format_vehicle_map_skip_warning(scenario_id, context, error))

    raise RuntimeError(
        format_vehicle_map_exhausted_error(
            num_scenarios=num_scenarios,
            context=context,
            last_scenario_id=last_scenario_id,
            last_error=last_error,
        )
    ) from last_error


def build_level_from_params(env: Any) -> None:
    """Build and initialize a level from ``env.level_params_vec``."""
    initialize_level_with_fallback(
        env=env,
        start_idx=int(env.level_params_vec[0]),
        create_level=lambda scenario_id: create_level_from_params(env, scenario_id),
        context="during adversary build",
    )


def sample_random_level(env: Any) -> ScenarioLevel:
    """Sample a random level using the environment's RNG state."""
    scenario_id = env._level_seed_random_state.choice(env.scenario_ids)
    seed = env._sample_level_seed()
    runtime_mode = getattr(env, "opponent_runtime_mode", "normal")

    if runtime_mode != "normal":
        return build_zero_tilt_level(
            scenario_id=scenario_id,
            seed=seed,
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
        )

    goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilting = (
        sample_random_tilt_components(
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
            tilt_range=env.tilt_range,
        )
    )
    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=goal_tilt,
        veh_veh_tilt=veh_veh_tilt,
        veh_edge_tilt=veh_edge_tilt,
        per_vehicle_tilting=per_vehicle_tilting,
    )


def mutate_level_internal(env: Any, level: ScenarioLevel) -> ScenarioLevel:
    """Mutate a level according to the environment's tilting config."""
    if env.tilting_mode == "none":
        return level

    rng = env._mutation_random_state
    if env.tilting_mode in ("global", "ego"):
        return mutate_global_level(
            level=level,
            rng=rng,
            mutation_mode=env.mutation_mode,
            mutation_range=env.mutation_range,
            tilt_range=env.tilt_range,
        )

    return mutate_per_vehicle_level(
        level=level,
        rng=rng,
        mutation_mode=env.mutation_mode,
        mutation_range=env.mutation_range,
        tilt_range=env.tilt_range,
        opponent_k=env.opponent_k,
        per_vehicle_tilting_length=env.per_vehicle_tilting_length,
        normalize_per_vehicle_tilting=lambda values: normalize_per_vehicle_tilting(
            values,
            env.per_vehicle_tilting_length,
        ),
    )
