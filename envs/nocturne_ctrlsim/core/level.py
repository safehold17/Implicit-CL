"""Level domain objects and lifecycle operations for Nocturne CtrlSim."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from ..services import scenario_pool as scenario_pool_service
from ..services.vehicle_map import (
    format_vehicle_map_exhausted_error,
    format_vehicle_map_skip_warning,
    is_retryable_vehicle_map_error,
)


@dataclass
class ScenarioLevel:
    """Represents a Nocturne scenario configuration used by DCD."""

    scenario_id: str
    seed: int
    goal_tilt: int
    veh_veh_tilt: int
    veh_edge_tilt: int
    per_vehicle_tilting: tuple[int, ...] = ()

    @staticmethod
    def _coerce_integer_tilt(name: str, value: float) -> int:
        """Convert a tilt value to an in-range integer."""
        tilt = float(value)
        if not tilt.is_integer():
            raise ValueError(f"{name} must be an integer, got {value}")
        tilt_int = int(tilt)
        if not (-25 <= tilt_int <= 25):
            raise ValueError(f"{name} must be in [-25, 25], got {tilt_int}")
        return tilt_int

    def __post_init__(self) -> None:
        """Validate level fields after dataclass initialization."""
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        for name in ["goal_tilt", "veh_veh_tilt", "veh_edge_tilt"]:
            object.__setattr__(
                self,
                name,
                self._coerce_integer_tilt(name, getattr(self, name)),
            )

        normalized_per_vehicle_tilting = tuple(
            self._coerce_integer_tilt(f"per_vehicle_tilting[{idx}]", value)
            for idx, value in enumerate(self.per_vehicle_tilting)
        )
        object.__setattr__(self, "per_vehicle_tilting", normalized_per_vehicle_tilting)

    def to_tuple(self) -> tuple:
        """Return the canonical tuple representation for serialization."""
        return (
            self.scenario_id,
            self.seed,
            self.goal_tilt,
            self.veh_veh_tilt,
            self.veh_edge_tilt,
            self.per_vehicle_tilting,
        )

    def with_scenario_id(self, scenario_id: str) -> "ScenarioLevel":
        """Return a copy of the level with a different scenario ID."""
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=self.seed,
            goal_tilt=self.goal_tilt,
            veh_veh_tilt=self.veh_veh_tilt,
            veh_edge_tilt=self.veh_edge_tilt,
            per_vehicle_tilting=self.per_vehicle_tilting,
        )

    def to_level_string(self) -> str:
        """Serialize the level to the string format used by LevelStore."""
        return str(self.to_tuple())

    @classmethod
    def from_level_string(cls, level_str: str) -> "ScenarioLevel":
        """Parse a level from the stored string representation."""
        values = ast.literal_eval(level_str)
        per_vehicle_tilting = tuple(values[5]) if len(values) > 5 else ()
        return cls(
            scenario_id=values[0],
            seed=values[1],
            goal_tilt=values[2],
            veh_veh_tilt=values[3],
            veh_edge_tilt=values[4],
            per_vehicle_tilting=per_vehicle_tilting,
        )

    def to_encoding(self, scenario_id_to_index: dict[str, int]) -> np.ndarray:
        """Encode the level into the numeric PLR representation."""
        scenario_index = scenario_id_to_index.get(self.scenario_id)
        if scenario_index is None:
            raise KeyError(f"Unknown scenario_id: {self.scenario_id}")

        return np.array(
            [
                scenario_index,
                self.goal_tilt,
                self.veh_veh_tilt,
                self.veh_edge_tilt,
                *self.per_vehicle_tilting,
                self.seed,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_encoding(
        cls,
        encoding: np.ndarray,
        index_to_scenario_id: dict[int, str],
    ) -> "ScenarioLevel":
        """Decode a numeric PLR representation into a level."""
        (
            scenario_index,
            goal_tilt,
            veh_veh_tilt,
            veh_edge_tilt,
            per_vehicle_tilting,
            seed,
        ) = cls.decode_encoding_fields(encoding)

        scenario_id = index_to_scenario_id.get(scenario_index)
        if scenario_id is None:
            raise KeyError(f"Unknown scenario_index: {scenario_index}")

        return cls(
            scenario_id=scenario_id,
            seed=seed,
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
            per_vehicle_tilting=per_vehicle_tilting,
        )

    @classmethod
    def decode_encoding_fields(
        cls,
        encoding: np.ndarray,
    ) -> tuple[int, float, float, float, tuple[int, ...], int]:
        """Decode primitive level fields from the PLR representation."""
        if len(encoding) <= 5:
            per_vehicle_tilting = ()
            seed_idx = 4
        else:
            seed_idx = len(encoding) - 1
            per_vehicle_tilting = tuple(
                cls._coerce_integer_tilt(
                    f"per_vehicle_tilting[{idx - 4}]",
                    encoding[idx],
                )
                for idx in range(4, seed_idx)
            )

        return (
            int(float(encoding[0])),
            cls._coerce_integer_tilt("goal_tilt", encoding[1]),
            cls._coerce_integer_tilt("veh_veh_tilt", encoding[2]),
            cls._coerce_integer_tilt("veh_edge_tilt", encoding[3]),
            per_vehicle_tilting,
            int(float(encoding[seed_idx])),
        )

    def __eq__(self, other: object) -> bool:
        """Compare levels using their serialized tuple representation."""
        if not isinstance(other, ScenarioLevel):
            return False
        return self.to_tuple() == other.to_tuple()

    def __hash__(self) -> int:
        """Hash the level via its serialized tuple representation."""
        return hash(self.to_tuple())


def normalize_per_vehicle_tilting(
    per_vehicle_tilting: tuple[int, ...],
    per_vehicle_tilting_length: int,
) -> tuple[int, ...]:
    """Normalize per-vehicle tilting to the configured flattened length."""
    normalized = [int(round(float(value))) for value in per_vehicle_tilting]
    if len(normalized) < per_vehicle_tilting_length:
        normalized.extend([0] * (per_vehicle_tilting_length - len(normalized)))
    elif len(normalized) > per_vehicle_tilting_length:
        normalized = normalized[:per_vehicle_tilting_length]
    return tuple(normalized)


def build_zero_tilt_level(
    scenario_id: str,
    seed: int,
    tilting_mode: str,
    per_vehicle_tilting_length: int,
) -> ScenarioLevel:
    """Build a zero-tilt level for the requested tilting mode."""
    per_vehicle_tilting = (
        (0,) * per_vehicle_tilting_length
        if tilting_mode == "per_vehicle"
        else ()
    )
    return ScenarioLevel(
        scenario_id=scenario_id,
        seed=seed,
        goal_tilt=0,
        veh_veh_tilt=0,
        veh_edge_tilt=0,
        per_vehicle_tilting=per_vehicle_tilting,
    )


def normalize_level_for_tilting_mode(
    level: ScenarioLevel,
    tilting_mode: str,
    per_vehicle_tilting_length: int,
) -> ScenarioLevel:
    """Normalize a level so its fields match the active tilting mode."""
    if tilting_mode == "per_vehicle":
        normalized_per_vehicle_tilting = normalize_per_vehicle_tilting(
            level.per_vehicle_tilting,
            per_vehicle_tilting_length,
        )
        return replace(level, per_vehicle_tilting=normalized_per_vehicle_tilting)

    if tilting_mode == "none":
        return build_zero_tilt_level(
            scenario_id=level.scenario_id,
            seed=level.seed,
            tilting_mode=tilting_mode,
            per_vehicle_tilting_length=per_vehicle_tilting_length,
        )

    return level


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

    from ..utils.encoding_helpers import decode_level_from_string_array

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

    from ..utils.tilt_helpers import encode_level_params

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
    scenario_id = env.np_random.choice(env.scenario_ids)
    seed = env._sample_level_seed()
    runtime_mode = getattr(env, "opponent_runtime_mode", "normal")

    if runtime_mode != "normal":
        return build_zero_tilt_level(
            scenario_id=scenario_id,
            seed=seed,
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
        )

    from ..utils.tilt_helpers import sample_random_tilt_components

    goal_tilt, veh_veh_tilt, veh_edge_tilt, per_vehicle_tilting = (
        sample_random_tilt_components(
            tilting_mode=env.tilting_mode,
            per_vehicle_tilting_length=env.per_vehicle_tilting_length,
            tilt_range=env.tilt_range,
            rng=env.np_random,
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

    from ..utils.tilt_helpers import mutate_global_level, mutate_per_vehicle_level

    rng = env.np_random
    if env.tilting_mode == "global":
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
