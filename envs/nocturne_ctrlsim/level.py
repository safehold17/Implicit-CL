"""
Scenario Level data structure.
"""
import ast
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict

# Constants for per-vehicle tilting
OPPONENT_K = 7
PER_VEHICLE_TILTING_LENGTH = 3 * OPPONENT_K  # 21


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
    
    # Per-vehicle tilting (flattened, length = PER_VEHICLE_TILTING_LENGTH)
    per_vehicle_tilting: Tuple[int, ...] = ()

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        for name in ["goal_tilt", "veh_veh_tilt", "veh_edge_tilt"]:
            val = int(round(float(getattr(self, name))))
            setattr(self, name, val)
            if not (-25 <= val <= 25):
                raise ValueError(f"{name} must be in [-25, 25], got {val}")
        
        # Normalize per_vehicle_tilting: pad with 0 if too short, truncate if too long
        if len(self.per_vehicle_tilting) < PER_VEHICLE_TILTING_LENGTH:
            padded = list(self.per_vehicle_tilting) + [0] * (PER_VEHICLE_TILTING_LENGTH - len(self.per_vehicle_tilting))
            object.__setattr__(self, 'per_vehicle_tilting', tuple(padded))
        elif len(self.per_vehicle_tilting) > PER_VEHICLE_TILTING_LENGTH:
            object.__setattr__(self, 'per_vehicle_tilting', self.per_vehicle_tilting[:PER_VEHICLE_TILTING_LENGTH])
        
        # Validate per_vehicle_tilting values
        for i, val in enumerate(self.per_vehicle_tilting):
            val = int(round(float(val)))
            if not (-25 <= val <= 25):
                raise ValueError(f"per_vehicle_tilting[{i}] must be in [-25, 25], got {val}")

    def to_tuple(self) -> Tuple:
        return (
            self.scenario_id,
            self.seed,
            round(self.goal_tilt),
            round(self.veh_veh_tilt),
            round(self.veh_edge_tilt),
            self.per_vehicle_tilting,
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
                round(self.goal_tilt),
                round(self.veh_veh_tilt),
                round(self.veh_edge_tilt),
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
        scenario_index = int(encoding[0])
        scenario_id = index_to_scenario_id.get(scenario_index)
        if scenario_id is None:
            raise KeyError(f"Unknown scenario_index: {scenario_index}")
        
        # Handle backward compatibility: old encoding has length 5, new has length 26
        if len(encoding) >= 5 + PER_VEHICLE_TILTING_LENGTH:
            # New format: [scenario_idx, goal, veh_veh, veh_edge, per_vehicle(21), seed]
            per_vehicle_tilting = tuple(int(round(float(encoding[i]))) for i in range(4, 4 + PER_VEHICLE_TILTING_LENGTH))
            seed_idx = 4 + PER_VEHICLE_TILTING_LENGTH
        else:
            # Old format: [scenario_idx, goal, veh_veh, veh_edge, seed]
            per_vehicle_tilting = ()
            seed_idx = 4

        return cls(
            scenario_id=scenario_id,
            seed=int(encoding[seed_idx]),
            goal_tilt=round(float(encoding[1])),
            veh_veh_tilt=round(float(encoding[2])),
            veh_edge_tilt=round(float(encoding[3])),
            per_vehicle_tilting=per_vehicle_tilting,
        )

    def __eq__(self, other):
        if not isinstance(other, ScenarioLevel):
            return False
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())
