"""Pure helpers for adversarial RTG-based policy reweighting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from ctrlsim_adapter.opponent_vehicle.discretization import (
    undiscretize_rtg_indices,
)


@dataclass(frozen=True, slots=True)
class AdversarialRTGConfig:
    """Configuration for adversarial RTG reweighting."""

    enabled: bool
    reward_scale: float
    epsilon: float
    error_mean: float
    error_sigma: float

    def __post_init__(self) -> None:
        """Validate the normalization scale."""
        if self.error_sigma <= 0:
            raise ValueError(
                f"error_sigma must be > 0, got {self.error_sigma}"
            )


def should_trigger_policy_reweighting_step(
    *,
    t: int,
    history_steps: int,
    reweighting_frequency: int,
) -> bool:
    """Return whether the current step should emit a new delayed reweighting signal."""
    if reweighting_frequency < 1:
        raise ValueError(
            "reweighting_frequency must be >= 1, "
            f"got {reweighting_frequency}"
        )

    anchor_t = int(history_steps) - 1
    if int(t) < anchor_t:
        return False
    phase = (int(t) - anchor_t) % int(reweighting_frequency)
    return phase == 0


def _to_numpy_vector(
    rtg_value: Sequence[float] | np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(rtg_value, torch.Tensor):
        return rtg_value.detach().cpu().numpy()
    return np.asarray(rtg_value)


def _is_discrete_rtg(
    rtg_value: Sequence[float] | np.ndarray | torch.Tensor,
) -> bool:
    if isinstance(rtg_value, torch.Tensor):
        return not torch.is_floating_point(rtg_value)
    return np.issubdtype(np.asarray(rtg_value).dtype, np.integer)


def recover_current_ego_rtg(
    rtg_value: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    rtg_discretization: int,
    min_rtg_pos: float,
    max_rtg_pos: float,
    min_rtg_veh: float,
    max_rtg_veh: float,
    min_rtg_road: float,
    max_rtg_road: float,
) -> np.ndarray:
    """Recover the current ego RTG as a continuous 3D vector."""
    raw = _to_numpy_vector(rtg_value).reshape(3)
    if _is_discrete_rtg(rtg_value):
        continuous = undiscretize_rtg_indices(
            int(raw[0]),
            int(raw[1]),
            int(raw[2]),
            rtg_discretization,
            min_rtg_pos,
            max_rtg_pos,
            min_rtg_veh,
            max_rtg_veh,
            min_rtg_road,
            max_rtg_road,
        )
        return np.asarray(continuous, dtype=np.float32)
    return raw.astype(np.float32, copy=False)


def compute_reward_delta(
    current_rtg: np.ndarray,
    next_rtg: np.ndarray,
) -> np.ndarray:
    """Estimate the instantaneous reward from two RTG vectors."""
    current = np.asarray(current_rtg, dtype=np.float32)
    next_value = np.asarray(next_rtg, dtype=np.float32)
    return current - next_value


def compute_target_next_rtg(
    tilted_current_rtg: np.ndarray | torch.Tensor,
    reward_delta: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Build a stopgrad target next RTG vector."""
    if isinstance(tilted_current_rtg, torch.Tensor) or isinstance(
        reward_delta, torch.Tensor
    ):
        if isinstance(tilted_current_rtg, torch.Tensor):
            base = tilted_current_rtg
        else:
            base = torch.as_tensor(tilted_current_rtg, dtype=torch.float32)
        if isinstance(reward_delta, torch.Tensor):
            delta = reward_delta.to(dtype=base.dtype, device=base.device)
        else:
            delta = torch.as_tensor(
                reward_delta,
                dtype=base.dtype,
                device=base.device,
            )
        with torch.no_grad():
            return base.detach() - delta.detach()

    tilted = np.asarray(tilted_current_rtg, dtype=np.float32)
    delta = np.asarray(reward_delta, dtype=np.float32)
    return tilted - delta


def compute_scale_from_error(
    error_value: float,
    config: AdversarialRTGConfig,
) -> float:
    """Convert a normalized RTG error into a scalar reweighting factor."""
    error_norm = (float(error_value) - config.error_mean) / config.error_sigma
    sigmoid = 1.0 / (1.0 + math.exp(-error_norm))
    return float(
        config.reward_scale * math.sqrt(sigmoid + config.epsilon)
    )


def compute_ego_action_scale(
    *,
    config: AdversarialRTGConfig,
    current_rtg: np.ndarray,
    next_rtg: np.ndarray,
    tilted_current_rtg: np.ndarray,
    ego_reweight_tilt: tuple[float, float, float],
    error_weights: np.ndarray | None = None,
) -> float:
    """Compute the delayed reweighting scale from RTG mismatch."""
    if not config.enabled:
        return 1.0
    if all(float(value) == 0.0 for value in ego_reweight_tilt):
        return 1.0

    current = np.asarray(current_rtg, dtype=np.float32)
    next_value = np.asarray(next_rtg, dtype=np.float32)
    tilted = np.asarray(tilted_current_rtg, dtype=np.float32)
    reward_delta = compute_reward_delta(current, next_value)
    target_next_rtg = np.asarray(
        compute_target_next_rtg(
            tilted_current_rtg=tilted,
            reward_delta=reward_delta,
        ),
        dtype=np.float32,
    )
    if error_weights is None:
        weights = np.ones_like(next_value, dtype=np.float32)
    else:
        weights = np.asarray(error_weights, dtype=np.float32)

    error_value = float(
        np.sum(np.square(weights * (next_value - target_next_rtg)))
    )
    return compute_scale_from_error(error_value, config)
