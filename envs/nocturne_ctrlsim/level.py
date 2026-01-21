"""
Scenario Level data structure.
"""
import ast
from dataclasses import dataclass
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

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        for name in ["goal_tilt", "veh_veh_tilt", "veh_edge_tilt"]:
            val = int(round(float(getattr(self, name))))
            setattr(self, name, val)
            if not (-25 <= val <= 25):
                raise ValueError(f"{name} must be in [-25, 25], got {val}")

    def to_tuple(self) -> Tuple:
        return (
            self.scenario_id,
            self.seed,
            round(self.goal_tilt),
            round(self.veh_veh_tilt),
            round(self.veh_edge_tilt),
        )

    def to_level_string(self) -> str:
        return str(self.to_tuple())

    @classmethod
    def from_level_string(cls, level_str: str) -> "ScenarioLevel":
        # Use ast.literal_eval instead of eval to avoid code execution risks
        t = ast.literal_eval(level_str)
        return cls(
            scenario_id=t[0],
            seed=t[1],
            goal_tilt=t[2],
            veh_veh_tilt=t[3],
            veh_edge_tilt=t[4],
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

        return cls(
            scenario_id=scenario_id,
            seed=int(encoding[4]),
            goal_tilt=round(float(encoding[1])),
            veh_veh_tilt=round(float(encoding[2])),
            veh_edge_tilt=round(float(encoding[3])),
        )

    def __eq__(self, other):
        if not isinstance(other, ScenarioLevel):
            return False
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())
