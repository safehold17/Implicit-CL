"""
负责在大体积 motion 数组场景下通过共享内存传递 IPC 负载。
该模块根据环境变量与字节阈值决定是否启用 SHM，并提供数组写入、恢复和清理逻辑。
Provides shared-memory transport helpers for large motion arrays in IPC payloads.
Decides when to use SHM from env/size thresholds and implements write, restore, and cleanup paths.
"""

from __future__ import annotations

import os
from multiprocessing import shared_memory
from typing import Any, Dict, Tuple

import numpy as np

from .schema import SHM_MOTION_STORAGE

DEFAULT_IPC_USE_SHM = True
DEFAULT_IPC_SHM_THRESHOLD_BYTES = 131072


def shared_memory_enabled() -> bool:
    """返回当前是否启用 SHM IPC。 / Return whether SHM-based IPC is enabled.

    默认启用 SHM，以匹配当前项目在 Nocturne-CtrlSim 批量推理上的优化基线。
    若环境变量显式设置，则始终以环境变量为准，便于实验时快速覆盖默认行为。

    SHM IPC is enabled by default to match the current optimization baseline used by
    Nocturne-CtrlSim batch inference. Explicit environment variables always override
    the default so experiments can still flip the behavior quickly.
    """
    default_value = "1" if DEFAULT_IPC_USE_SHM else "0"
    value = str(os.getenv("CTRLSIM_IPC_USE_SHM", default_value)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def shared_memory_threshold_bytes() -> int:
    """返回触发 SHM 传输的字节阈值。 / Return the byte threshold that switches IPC to SHM.

    默认阈值下调到 256 KiB，使 prepared payload 更早走 SHM 通道，减少大数组直接
    经由 pipe 传输时的复制与序列化成本。环境变量仍可覆盖该默认值。

    The default threshold is lowered to 256 KiB so prepared payloads switch to SHM
    earlier, reducing copy and serialization overhead when large arrays would otherwise
    travel directly through pipes. The environment variable still overrides the default.
    """
    return max(
        0,
        int(
            os.getenv(
                "CTRLSIM_IPC_SHM_THRESHOLD_BYTES",
                str(DEFAULT_IPC_SHM_THRESHOLD_BYTES),
            )
        ),
    )


def should_use_shared_memory(total_bytes: int) -> bool:
    return shared_memory_enabled() and total_bytes >= shared_memory_threshold_bytes()


def pack_motion_array_to_shared_memory(
    array: np.ndarray,
) -> Dict[str, Any]:
    """将数组写入一次性 SHM，并返回 IPC payload。 / Write one array into one-shot SHM and return its IPC payload.

    这里始终创建一次性 SHM 段，让 prepared payload 的生命周期保持简单：生产端 close，
    消费端 close+unlink，不再维护跨 step 复用状态。

    This path always creates a one-shot SHM segment so prepared payload lifecycle stays
    simple: the producer closes it, and the consumer closes plus unlinks it without
    keeping any cross-step reuse state.
    """
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
    """从 SHM payload 恢复数组视图。 / Restore one array view from a SHM payload.

    恢复结果只返回数组视图和一次性 SHM 句柄。消费方在使用完后统一执行 close+unlink。

    The restored result only returns the array view and the one-shot SHM handle.
    Consumers always release it with close plus unlink after use.
    """
    shm = shared_memory.SharedMemory(name=str(payload["name"]))
    array = np.ndarray(
        tuple(int(dim) for dim in payload["shape"]),
        dtype=np.dtype(str(payload["dtype"])),
        buffer=shm.buf,
    )
    return array, shm


def release_shared_memory_handle(
    shm_handle: shared_memory.SharedMemory,
) -> None:
    """释放一次性 SHM 句柄。 / Release one one-shot SHM handle.

    prepared payload 恢复出的 SHM 句柄在消费后统一 close+unlink，保持 IPC 生命周期
    简单明确。

    SHM handles restored from prepared payloads are always closed and unlinked after
    consumption so IPC lifecycle remains simple and explicit.
    """
    close_and_unlink_shared_memory(shm_handle)


def close_and_unlink_shared_memory(shm_handle: shared_memory.SharedMemory) -> None:
    try:
        shm_handle.close()
    finally:
        try:
            shm_handle.unlink()
        except FileNotFoundError:
            pass
