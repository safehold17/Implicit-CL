from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class SparseInferenceConfig:
    enabled: bool = False
    interval: int = 2

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")


class SparseInferenceController:
    def __init__(self, cfg: SparseInferenceConfig) -> None:
        self.cfg = cfg
        self._cached_actions: Dict[int, Tuple[float, float]] = {}

    def clear_on_reset(self) -> None:
        self._cached_actions.clear()

    def is_sparse_step(self, t: int, history_steps: int) -> bool:
        if not self.cfg.enabled:
            return False
        anchor_t = history_steps - 1
        if t < anchor_t:
            return False
        phase = (t - anchor_t) % self.cfg.interval
        return phase == self.cfg.interval - 1

    def should_infer(self, t: int, history_steps: int) -> bool:
        # Backward-compatible API: infer on non-sparse model steps.
        anchor_t = history_steps - 1
        if t < anchor_t:
            return False
        return not self.is_sparse_step(t=t, history_steps=history_steps)

    def cache_actions(self, actions: Dict[int, Tuple[float, float]]) -> None:
        self._cached_actions.update(actions)

    def get_cached_action(self, veh_id: int) -> Optional[Tuple[float, float]]:
        return self._cached_actions.get(veh_id)
