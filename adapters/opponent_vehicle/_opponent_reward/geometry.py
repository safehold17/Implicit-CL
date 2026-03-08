from __future__ import annotations

import numpy as np

CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0


def dot_product_2d_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs[..., 0] * rhs[..., 0] + lhs[..., 1] * rhs[..., 1]


def cross_product_2d_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def compute_signed_distance_to_polyline_np(
    xys: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    is_cyclic = (
        np.square(polyline[0] - polyline[-1]).sum()
        < CYCLIC_MAP_FEATURE_TOLERANCE_M2
    )
    xy_starts = polyline[None, :-1, :2]
    xy_ends = polyline[None, 1:, :2]
    start_to_point = xys[:, None, :2] - xy_starts
    start_to_end = xy_ends - xy_starts

    rel_t = np.nan_to_num(
        dot_product_2d_np(start_to_point, start_to_end)
        / dot_product_2d_np(start_to_end, start_to_end)
    )
    n = np.sign(cross_product_2d_np(start_to_point, start_to_end))
    distance_to_segment = np.linalg.norm(
        start_to_point - (start_to_end * np.clip(rel_t, 0.0, 1.0)[..., None]),
        axis=-1,
    )

    start_to_end_padded = np.concatenate(
        [start_to_end[:, -1:], start_to_end, start_to_end[:, :1]],
        axis=1,
    )
    is_locally_convex = (
        cross_product_2d_np(
            start_to_end_padded[:, :-1],
            start_to_end_padded[:, 1:],
        )
        > 0.0
    )
    n_prior = np.concatenate(
        [np.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]],
        axis=-1,
    )
    n_next = np.concatenate(
        [n[:, 1:], np.where(is_cyclic, n[:, :1], n[:, -1:])],
        axis=-1,
    )
    sign_if_before = np.where(
        is_locally_convex[:, :-1],
        np.maximum(n, n_prior),
        np.minimum(n, n_prior),
    )
    sign_if_after = np.where(
        is_locally_convex[:, 1:],
        np.maximum(n, n_next),
        np.minimum(n, n_next),
    )
    sign_to_segment = np.where(
        rel_t < 0.0,
        sign_if_before,
        np.where(rel_t < 1.0, n, sign_if_after),
    )
    min_segment_idx = np.argmin(distance_to_segment, axis=-1)
    distance_sign = np.take_along_axis(
        sign_to_segment,
        min_segment_idx[:, None],
        axis=1,
    )[:, 0]
    return distance_sign * np.min(distance_to_segment, axis=-1)


def compute_signed_distance_to_polylines_np(
    xys: np.ndarray,
    polylines: tuple[np.ndarray, ...],
) -> np.ndarray:
    distances = [
        compute_signed_distance_to_polyline_np(xys, polyline)
        for polyline in polylines
        if len(polyline) >= 2
    ]
    if not distances:
        return np.zeros((len(xys),), dtype=np.float32)
    stacked = np.stack(distances, axis=-1)
    nearest_idx = np.argmin(np.abs(stacked), axis=-1)
    return np.take_along_axis(stacked, nearest_idx[:, None], axis=1)[:, 0]
