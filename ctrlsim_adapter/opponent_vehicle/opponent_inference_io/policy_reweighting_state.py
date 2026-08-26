"""Worker-owned pending state for the new policy-reweighting path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PolicyReweightingState:
    """Track one decoded tilted RTG and its realized reward propagation."""

    sampled_tilted_rtg: Optional[np.ndarray] = None
    accumulated_realized_component_rewards: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    query_gap: int = 0

    @property
    def has_pending(self) -> bool:
        """Return whether a decoded tilted RTG is pending comparison."""
        return self.sampled_tilted_rtg is not None

    @property
    def target_rtg(self) -> Optional[np.ndarray]:
        """Return the pending RTG after subtracting realized rewards."""
        if self.sampled_tilted_rtg is None:
            return None
        return self.sampled_tilted_rtg - self.accumulated_realized_component_rewards

    def start_query(self, sampled_tilted_rtg: np.ndarray) -> None:
        """Start a pending interval from the decoded tilted ego RTG."""
        self.sampled_tilted_rtg = np.asarray(
            sampled_tilted_rtg,
            dtype=np.float32,
        ).reshape(3).copy()
        self.accumulated_realized_component_rewards.fill(0.0)
        self.query_gap = 0

    def accumulate_realized_reward(self, component_reward: np.ndarray) -> None:
        """Accumulate one realized component reward for the pending interval."""
        if not self.has_pending:
            return
        self.accumulated_realized_component_rewards += np.asarray(
            component_reward,
            dtype=np.float32,
        ).reshape(3)
        self.query_gap += 1

    def discard(self) -> None:
        """Discard the pending interval without producing statistics."""
        self.sampled_tilted_rtg = None
        self.accumulated_realized_component_rewards.fill(0.0)
        self.query_gap = 0

    def reset(self) -> None:
        """Reset all episode-scoped pending state."""
        self.discard()
