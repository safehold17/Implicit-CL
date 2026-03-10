from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


def capture_sampling_rng_state(device: Any) -> np.ndarray:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(torch_device)
    else:
        rng_state = torch.get_rng_state()
    return rng_state.detach().cpu().numpy().astype(np.uint8, copy=False)


def resolve_sampling_rng_state(
    adapter: Any,
    worker_rng_state: Optional[np.ndarray],
) -> np.ndarray:
    if worker_rng_state is not None:
        return np.asarray(worker_rng_state, dtype=np.uint8)
    return capture_sampling_rng_state(adapter.device)


def restore_sampling_rng_state(device: Any, rng_state: Any) -> None:
    state_tensor = torch.as_tensor(
        np.asarray(rng_state, dtype=np.uint8),
        dtype=torch.uint8,
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_rng_state(state_tensor, device=torch_device)
        return
    torch.set_rng_state(state_tensor)
