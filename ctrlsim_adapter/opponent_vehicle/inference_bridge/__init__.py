"""Adapter-side inference bridge helpers for Nocturne + CtRL-Sim."""

from .apply_outputs import apply_predictions
from .focal_input import build_focal_batch, build_focal_batches
from .prepare_inference_payload import prepare_step
from .sampling_rng import capture_sampling_rng_state, restore_sampling_rng_state

_build_focal_batch = build_focal_batch

__all__ = [
    "apply_predictions",
    "build_focal_batches",
    "capture_sampling_rng_state",
    "prepare_step",
    "restore_sampling_rng_state",
    "_build_focal_batch",
]
