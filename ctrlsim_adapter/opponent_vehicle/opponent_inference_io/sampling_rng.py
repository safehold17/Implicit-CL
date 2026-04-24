"""
负责 adapter 侧推理桥的无状态采样 seed 抓取、episode 初始化与默认解析。
Captures, initializes, and resolves stateless sampling seeds for the adapter-side inference bridge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
_SEED_MIX_CONST = np.uint64(0xD6E8FEB86659FD93)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    x = (values.astype(np.uint64, copy=False) + _SPLITMIX64_GAMMA) & _UINT64_MASK
    x ^= x >> np.uint64(30)
    x = (x * _SPLITMIX64_MUL1) & _UINT64_MASK
    x ^= x >> np.uint64(27)
    x = (x * _SPLITMIX64_MUL2) & _UINT64_MASK
    x ^= x >> np.uint64(31)
    return x


def capture_sampling_seed(device: Any) -> int:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(torch_device)
    else:
        rng_state = torch.get_rng_state()
    rng_bytes = rng_state.detach().cpu().numpy().astype(np.uint8, copy=False)
    seed_words = np.frombuffer(rng_bytes.tobytes(), dtype=np.uint64)
    if seed_words.size == 0:
        seed_words = np.asarray([np.uint64(0)], dtype=np.uint64)
    mixed = _splitmix64(seed_words ^ (_SEED_MIX_CONST * np.arange(seed_words.size, dtype=np.uint64)))
    return int(np.bitwise_xor.reduce(mixed, dtype=np.uint64))


def initialize_episode_sampling_seed(adapter: Any) -> int:
    reset_count = int(getattr(adapter, "_sampling_episode_index", 0))
    base_seed = np.uint64(capture_sampling_seed(adapter.device))
    episode_seed = int(_splitmix64(np.asarray([base_seed ^ np.uint64(reset_count)], dtype=np.uint64))[0])
    adapter._sampling_episode_index = reset_count + 1
    adapter._sampling_seed = episode_seed
    return episode_seed


def resolve_sampling_seed(adapter: Any) -> int:
    sampling_seed = getattr(adapter, "_sampling_seed", None)
    if sampling_seed is None:
        return initialize_episode_sampling_seed(adapter)
    return int(sampling_seed)
