"""Batch inference 模式下 adapter 侧的数据准备与结果应用。"""

from __future__ import annotations

from .apply import apply_predictions
from .preparation.focal_batch import build_focal_batch, build_focal_batches
from .preparation.rng import (
    capture_sampling_rng_state,
    restore_sampling_rng_state,
)
from .preparation.step import prepare_step

_build_focal_batch = build_focal_batch

__all__ = [
    "apply_predictions",
    "build_focal_batches",
    "capture_sampling_rng_state",
    "prepare_step",
    "restore_sampling_rng_state",
    "_build_focal_batch",
]
