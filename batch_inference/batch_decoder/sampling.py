"""
负责无状态随机票据生成与 batched categorical / top-p 采样。
该模块为 action / RTG fully vectorized decode 提供共享张量算子。
Builds stateless random tickets plus batched categorical / top-p samplers.
Provides shared tensor helpers for fully vectorized action and RTG decoding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
_DRAW_OFFSET = np.uint64(0xD6E8FEB86659FD93)
_STEP_MIX = np.uint64(0xA0761D6478BD642F)
_VEHICLE_MIX = np.uint64(0xE7037ED1A0B428DB)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    x = (values.astype(np.uint64, copy=False) + _SPLITMIX64_GAMMA) & _UINT64_MASK
    x ^= x >> np.uint64(30)
    x = (x * _SPLITMIX64_MUL1) & _UINT64_MASK
    x ^= x >> np.uint64(27)
    x = (x * _SPLITMIX64_MUL2) & _UINT64_MASK
    x ^= x >> np.uint64(31)
    return x


def build_stateless_uniforms(
    base_seed: int | np.ndarray,
    row_keys: np.ndarray,
    draws_per_row: int,
) -> np.ndarray:
    row_keys_u64 = np.asarray(row_keys, dtype=np.uint64).reshape((-1, 1))
    base_seed_u64 = np.asarray(base_seed, dtype=np.uint64).reshape((-1, 1))
    if base_seed_u64.shape[0] == 1:
        base_seed_u64 = np.broadcast_to(base_seed_u64, row_keys_u64.shape)
    elif base_seed_u64.shape[0] != row_keys_u64.shape[0]:
        raise ValueError("base_seed must be scalar or have one entry per row_key.")
    draw_offsets = _DRAW_OFFSET * np.arange(draws_per_row, dtype=np.uint64).reshape((1, -1))
    mixed = _splitmix64(row_keys_u64 ^ base_seed_u64 ^ draw_offsets)
    return ((mixed >> np.uint64(11)).astype(np.float64)) / float(1 << 53)


def build_row_keys(
    step_t: np.ndarray,
    veh_ids: np.ndarray,
    stage_tag: int,
) -> np.ndarray:
    step_u64 = np.asarray(step_t, dtype=np.uint64)
    veh_u64 = np.asarray(veh_ids, dtype=np.uint64)
    stage_u64 = np.uint64(int(stage_tag))
    return _splitmix64((step_u64 * _STEP_MIX) ^ (veh_u64 * _VEHICLE_MIX) ^ stage_u64)


def sample_categorical_from_uniform(
    probs: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    cdf = probs.cumsum(dim=-1)
    return torch.sum(uniforms.unsqueeze(-1) > cdf, dim=-1, dtype=torch.long)


def build_action_probabilities(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    nucleus_sampling: torch.Tensor,
    nucleus_thresholds: torch.Tensor,
) -> torch.Tensor:
    scaled_logits = logits / temperatures.unsqueeze(-1)
    action_probs = F.softmax(scaled_logits, dim=-1)

    top_p_row_idx = torch.nonzero(nucleus_sampling, as_tuple=False).flatten()
    if top_p_row_idx.numel() == 0:
        return action_probs

    top_p_probs = action_probs[top_p_row_idx]
    top_p_thresholds = nucleus_thresholds[top_p_row_idx]
    sorted_probs, sorted_indices = torch.sort(top_p_probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    selected = cum_probs < top_p_thresholds.unsqueeze(-1)
    selected = torch.cat(
        [
            torch.ones(
                (sorted_probs.shape[0], 1),
                dtype=torch.bool,
                device=sorted_probs.device,
            ),
            selected[:, :-1],
        ],
        dim=-1,
    )
    masked_sorted_probs = torch.where(selected, sorted_probs, torch.zeros_like(sorted_probs))
    masked_sorted_probs = masked_sorted_probs / masked_sorted_probs.sum(dim=-1, keepdim=True)
    top_p_action_probs = torch.zeros_like(top_p_probs).scatter(
        dim=-1,
        index=sorted_indices,
        src=masked_sorted_probs,
    )
    action_probs = action_probs.clone()
    action_probs[top_p_row_idx] = top_p_action_probs
    return action_probs
