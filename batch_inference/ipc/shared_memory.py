from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import Any, Dict, Tuple

import numpy as np

from .schema import SHM_MOTION_STORAGE


def shared_memory_enabled() -> bool:
    value = str(os.getenv("CTRLSIM_IPC_USE_SHM", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def shared_memory_threshold_bytes() -> int:
    return max(0, int(os.getenv("CTRLSIM_IPC_SHM_THRESHOLD_BYTES", "1048576")))


def should_use_shared_memory(total_bytes: int) -> bool:
    return shared_memory_enabled() and total_bytes >= shared_memory_threshold_bytes()


def pack_motion_array_to_shared_memory(array: np.ndarray) -> Dict[str, Any]:
    shm = shared_memory.SharedMemory(create=True, size=int(array.nbytes))
    try:
        shm_view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        np.copyto(shm_view, array)
    finally:
        shm.close()
    return {
        "storage": SHM_MOTION_STORAGE,
        "name": shm.name,
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": array.dtype.str,
    }


def unpack_motion_array_from_shared_memory(
    payload: Dict[str, Any],
) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=str(payload["name"]))
    array = np.ndarray(
        tuple(int(dim) for dim in payload["shape"]),
        dtype=np.dtype(str(payload["dtype"])),
        buffer=shm.buf,
    )
    return array, shm


def close_and_unlink_shared_memory(shm_handle: shared_memory.SharedMemory) -> None:
    try:
        shm_handle.close()
    finally:
        try:
            shm_handle.unlink()
        except FileNotFoundError:
            pass

