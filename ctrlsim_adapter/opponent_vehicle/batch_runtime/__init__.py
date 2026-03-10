"""Adapter-side batch runtime helpers for Nocturne + CtRL-Sim."""

from .apply import apply_predictions
from .focal_batch import build_focal_batch, build_focal_batches
from .prepare import prepare_step
from .rng import capture_sampling_rng_state, restore_sampling_rng_state

_build_focal_batch = build_focal_batch

__all__ = [
    "apply_predictions",
    "build_focal_batches",
    "capture_sampling_rng_state",
    "prepare_step",
    "restore_sampling_rng_state",
    "_build_focal_batch",
]
