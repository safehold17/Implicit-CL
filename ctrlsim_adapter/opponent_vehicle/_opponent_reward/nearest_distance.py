from __future__ import annotations

import numpy as np


def compute_nearest_dist_to_all(
    target_positions: np.ndarray,
    all_positions: np.ndarray,
    all_existence: np.ndarray,
    target_existence: np.ndarray,
    target_all_indices: np.ndarray,
) -> np.ndarray:
    if len(target_positions) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    with np.errstate(invalid="ignore"):
        diff = target_positions[:, np.newaxis, :] - all_positions[np.newaxis, :, :]
        squared_dist = np.sum(diff**2, axis=-1)

    valid_all = all_existence.astype(bool)
    squared_dist[:, ~valid_all] = np.inf
    row_idx = np.arange(len(target_positions), dtype=np.int64)
    squared_dist[row_idx, target_all_indices] = np.inf

    nearest = np.sqrt(np.min(squared_dist, axis=1))
    nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)
    nearest = nearest * target_existence
    return nearest[:, np.newaxis].astype(np.float32)

