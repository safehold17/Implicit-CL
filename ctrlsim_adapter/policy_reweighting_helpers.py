"""Pure helpers for adversarial RTG-based policy reweighting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ctrlsim_adapter.opponent_vehicle.discretization import (
    undiscretize_rtgs,
)


@dataclass(frozen=True, slots=True)
class AdversarialRTGConfig:
    """Configuration for adversarial RTG reweighting."""

    enabled: bool
    reward_scale: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class AdversarialRTGRunningStats:
    """Running statistics used to normalize RTG mismatch online."""

    error_mean: float
    error_sigma: float
    error_count: float


def should_trigger_policy_reweighting_step(
    *,
    t: int,
    history_steps: int,
    reweighting_frequency: int,
) -> bool:
    """Return whether the current step should emit a new delayed reweighting signal."""
    if reweighting_frequency < 1:
        raise ValueError(
            f"reweighting_frequency must be >= 1, got {reweighting_frequency}"
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
    return recover_current_ego_rtgs(
        _to_numpy_vector(rtg_value).reshape(1, 3),
        rtg_discretization=rtg_discretization,
        min_rtg_pos=min_rtg_pos,
        max_rtg_pos=max_rtg_pos,
        min_rtg_veh=min_rtg_veh,
        max_rtg_veh=max_rtg_veh,
        min_rtg_road=min_rtg_road,
        max_rtg_road=max_rtg_road,
    )[0]


def recover_current_ego_rtgs(
    rtg_values: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    *,
    rtg_discretization: int,
    min_rtg_pos: float,
    max_rtg_pos: float,
    min_rtg_veh: float,
    max_rtg_veh: float,
    min_rtg_road: float,
    max_rtg_road: float,
) -> np.ndarray:
    """Recover a batch of ego RTGs as continuous 3D vectors."""
    raw = _to_numpy_vector(rtg_values).reshape(-1, 3)
    if _is_discrete_rtg(rtg_values):
        continuous = undiscretize_rtgs(
            raw,
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

def compute_scale_from_error(
    error_value: float,
    config: AdversarialRTGConfig,
    running_stats: AdversarialRTGRunningStats,
) -> float:
    """Convert RTG error into a scalar using online running statistics."""
    if float(running_stats.error_count) < 2.0:
        return 1.0
    if float(running_stats.error_sigma) <= 0.0:
        return 1.0

    error_norm = (float(error_value) - float(running_stats.error_mean)) / float(
        running_stats.error_sigma
    )
    sigmoid = 1.0 / (1.0 + math.exp(-error_norm))
    return float(config.reward_scale * math.sqrt(sigmoid + config.epsilon))
