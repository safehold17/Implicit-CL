"""
Scenario Level data structure.
"""
import ast
from dataclasses import dataclass, replace
import numpy as np
from typing import Tuple, Dict


@dataclass
class ScenarioLevel:
    """
    Represents a Nocturne driving scenario configuration.

    In DCD, a level is stored in two ways:
    1. LevelStore: use to_level_string() / from_level_string()
    2. PLR buffer: use to_encoding() / from_encoding()

    A level can be:
    - randomly generated (reset_random)
    - loaded by id (reset_to_level)
    - mutated (mutate_level)
    - encoded for storage (encoding property)
    """

    # Core fields (minimal set)
    scenario_id: str
    seed: int

    # Domain tilting parameters in [-25, 25]
    goal_tilt: int
    veh_veh_tilt: int
    veh_edge_tilt: int
    
    # Per-vehicle tilting (flattened, variable length)
    per_vehicle_tilting: Tuple[int, ...] = ()

    @staticmethod
    def _coerce_integer_tilt(name: str, value: float) -> int:
        tilt = float(value)
        if not tilt.is_integer():
            raise ValueError(f"{name} must be an integer, got {value}")
        tilt_int = int(tilt)
        if not (-25 <= tilt_int <= 25):
            raise ValueError(f"{name} must be in [-25, 25], got {tilt_int}")
        return tilt_int

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        for name in ["goal_tilt", "veh_veh_tilt", "veh_edge_tilt"]:
            object.__setattr__(
                self,
                name,
                self._coerce_integer_tilt(name, getattr(self, name)),
            )

        normalized_per_vehicle_tilting = tuple(
            self._coerce_integer_tilt(f"per_vehicle_tilting[{i}]", val)
            for i, val in enumerate(self.per_vehicle_tilting)
        )
        object.__setattr__(self, 'per_vehicle_tilting', normalized_per_vehicle_tilting)

    def to_tuple(self) -> Tuple:
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
        return str(self.to_tuple())

    @classmethod
    def from_level_string(cls, level_str: str) -> "ScenarioLevel":
        # Use ast.literal_eval instead of eval to avoid code execution risks
        t = ast.literal_eval(level_str)
        # Handle backward compatibility: old format without per_vehicle_tilting
        per_vehicle_tilting = tuple(t[5]) if len(t) > 5 else ()
        return cls(
            scenario_id=t[0],
            seed=t[1],
            goal_tilt=t[2],
            veh_veh_tilt=t[3],
            veh_edge_tilt=t[4],
            per_vehicle_tilting=per_vehicle_tilting,
        )

    def to_encoding(self, scenario_id_to_index: Dict[str, int]) -> np.ndarray:
        scenario_index = scenario_id_to_index.get(self.scenario_id)
        if scenario_index is None:
            raise KeyError(f"Unknown scenario_id: {self.scenario_id}")

        encoding = np.array(
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
        return encoding

    @classmethod
    def from_encoding(
        cls, encoding: np.ndarray, index_to_scenario_id: Dict[int, str]
    ) -> "ScenarioLevel":
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
    ) -> Tuple[int, float, float, float, Tuple[int, ...], int]:
        """
        Decode encoding array into primitive level fields.

        This method only parses raw values and keeps no scenario mapping logic.
        """
        # Old format: [scenario_idx, goal, veh_veh, veh_edge, seed]
        if len(encoding) <= 5:
            per_vehicle_tilting = ()
            seed_idx = 4
        else:
            # New variable-length format:
            # [scenario_idx, goal, veh_veh, veh_edge, per_vehicle..., seed]
            seed_idx = len(encoding) - 1
            per_vehicle_tilting = tuple(
                cls._coerce_integer_tilt(
                    f"per_vehicle_tilting[{i - 4}]",
                    encoding[i],
                )
                for i in range(4, seed_idx)
            )

        return (
            int(float(encoding[0])),
            cls._coerce_integer_tilt("goal_tilt", encoding[1]),
            cls._coerce_integer_tilt("veh_veh_tilt", encoding[2]),
            cls._coerce_integer_tilt("veh_edge_tilt", encoding[3]),
            per_vehicle_tilting,
            int(float(encoding[seed_idx])),
        )

    def __eq__(self, other):
        if not isinstance(other, ScenarioLevel):
            return False
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())


def normalize_per_vehicle_tilting(
    per_vehicle_tilting: Tuple[int, ...],
    per_vehicle_tilting_length: int,
) -> Tuple[int, ...]:
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
