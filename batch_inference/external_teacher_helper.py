"""
聚合 external teacher 与 collator 共享的纯 helper。
该模块只放不依赖实例状态的函数，供批量推理主流程和 collate 组装路径共同复用。
Collect pure helpers shared by the external teacher and collator.
This module stores only stateless functions that can be reused by both the
batched inference main flow and the collate/metadata assembly path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .batch_ipc.prepared import (
    get_prepared_focal_count,
    get_prepared_focal_predict_rtgs,
)


def _assert_required_keys(
    payload: Dict[str, Any],
    required: Tuple[str, ...],
    payload_name: str,
) -> None:
    """检查 payload 是否包含所有必需键。

    这个 helper 为 prepared/model-output 一类的字典契约提供统一的必填字段校验，
    这样调用方在进入主逻辑前就能尽早失败，而不是在后续某个更深的位置因为缺键抛出
    不清晰的异常。

    Validate that a payload contains all required keys. It gives prepared/model
    output contracts one central fail-fast check so callers raise a clear error
    before entering deeper logic instead of failing later with an opaque
    missing-key exception.
    """
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def _config_get(source: Any, key: str, default: Any) -> Any:
    """从对象或映射中读取配置值。

    external teacher 既会接收 argparse/namespace 风格对象，也会接收 dict 风格配置。
    这个 helper 把两种读取路径收口成统一入口，避免在构造参数时重复写分支判断。

    Read one config value from either an object or a mapping. The external
    teacher consumes both argparse-style objects and dict-like configs, so this
    helper centralizes the lookup logic and avoids repeated branching at each
    call site.
    """
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _collect_focal_jobs(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result,
    job_type: str = "opponent",
    return_action_logits: bool | Sequence[bool] = False,
) -> List[Dict[str, Any]]:
    """按 env prepared 收集 flat focal jobs。

    该 helper 把 per-env prepared 扫描成 flat job 列表，并在 `status == "skip"`
    时直接回填空 env 结果。这样主流程只处理“真正需要前向的 jobs”，而 skip/env
    边界逻辑被统一收敛在同一个地方。

    Collect flat focal jobs from per-env prepared payloads. It also writes back
    empty env results for `status == "skip"` payloads so the main inference flow
    only needs to reason about jobs that actually require a forward pass.
    """
    focal_jobs: List[Dict[str, Any]] = []
    per_env_return_action_logits = None
    if not isinstance(return_action_logits, bool):
        if len(return_action_logits) != len(per_env_prepared):
            raise ValueError(
                "return_action_logits length must match per_env_prepared length."
            )
        per_env_return_action_logits = return_action_logits

    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None:
            continue
        env_return_action_logits = (
            bool(return_action_logits)
            if per_env_return_action_logits is None
            else bool(per_env_return_action_logits[env_idx])
        )

        _assert_required_keys(
            prepared,
            ("status", "step_t", "token_index", "dead_ids"),
            f"prepared env_idx={env_idx}",
        )
        status = prepared["status"]
        if status == "skip":
            results[env_idx] = build_empty_env_result(
                prepared,
                env_idx=env_idx,
                status="skip",
            )
            continue
        if status != "ok":
            raise ValueError(f"prepared env_idx={env_idx} has invalid status={status!r}")

        for focal_idx in range(get_prepared_focal_count(prepared)):
            focal_jobs.append(
                {
                    "env_idx": env_idx,
                    "prepared": prepared,
                    "focal_idx": focal_idx,
                    "predict_rtgs": get_prepared_focal_predict_rtgs(prepared, focal_idx),
                    "job_type": job_type,
                    "return_action_logits": env_return_action_logits,
                }
            )

    return focal_jobs


def _fill_empty_ok_env_results(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result,
) -> None:
    """为 status=ok 但没有任何输出的 env 补空结果。

    这个 helper 统一承载“没有任何 job 命中，但 env 状态仍然是 ok”时的补齐逻辑，
    让主流程不用到处写 `if results[env_idx] is None` 的收尾代码。

    Fill empty outputs for envs whose status is still `ok` but that produced no
    job results. It centralizes the post-processing step so callers do not need
    to repeat the same `results[env_idx] is None` cleanup logic.
    """
    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None or prepared["status"] != "ok":
            continue
        if results[env_idx] is None:
            results[env_idx] = build_empty_env_result(
                prepared,
                env_idx=env_idx,
                status="ok",
            )


def _concat_or_empty_ids(parts: List[np.ndarray]) -> np.ndarray:
    """拼接多段 flat ID 数组；若为空则返回空数组。

    该 helper 专门用于 env 聚合收尾阶段，把按 parts 收集的一维 id 片段统一拼成
    输出契约需要的 flat 数组，同时避免调用方反复手写空列表分支。

    Concatenate flat id-array parts or return an empty array. It is used during
    env-level aggregation finalization so callers can join collected id spans
    without re-implementing empty-list handling each time.
    """
    return _concat_or_empty(parts, dtype=np.int64)


def _concat_or_empty_values(parts: List[np.ndarray], width: int) -> np.ndarray:
    """拼接多段 flat 值矩阵；若为空则返回空矩阵。

    这个 helper 与 `_concat_or_empty_ids` 对应，只是面向二维值矩阵。它统一保证
    输出 shape 是 `(N, width)`，并在没有任何分段时返回形状稳定的空矩阵。

    Concatenate flat value-matrix parts or return an empty matrix. It mirrors
    `_concat_or_empty_ids`, but keeps the output shape stable as `(N, width)`
    and returns a correctly shaped empty matrix when there are no parts.
    """
    if not parts:
        return np.zeros((0, width), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _find_flat_rtg_value(
    flat_rtg_results: Dict[str, Any],
    job_idx: int,
    veh_id: int,
) -> np.ndarray | None:
    """在 flat RTG 结果中查找指定 job/veh 的连续 RTG 值。

    `flat_rtg_results` 以 job offset 编码多个 job 的 RTG 输出。这个 helper 负责
    用 `(job_idx, veh_id)` 从中定位某辆车的最后一行 RTG 结果，供 delayed
    reweighting 的 scale 计算使用。

    Find the continuous RTG value of one vehicle inside the flat RTG result
    object. The helper resolves the `(job_idx, veh_id)` pair against the job
    offsets and returns the last matching RTG row for delayed reweighting.
    """
    job_offsets = np.asarray(flat_rtg_results["job_offsets"], dtype=np.int64)
    start = int(job_offsets[job_idx])
    end = int(job_offsets[job_idx + 1])
    veh_ids = np.asarray(flat_rtg_results["veh_id"], dtype=np.int64)[start:end]
    values = np.asarray(flat_rtg_results["values"], dtype=np.float32)[start:end]
    row_indices = np.nonzero(veh_ids == int(veh_id))[0]
    if row_indices.size == 0:
        return None
    return values[int(row_indices[-1])]


def _job_span(job_offsets: Any, job_idx: int) -> Tuple[int, int]:
    """返回某个 job 在 flat 数组中的起止区间。

    该 helper 统一解析 `job_offsets[job_idx:job_idx+2]`，让上层聚合逻辑只关心
    “这个 job 对应哪一段 flat rows”，不用反复手写 offset 下标和类型转换。

    Return the `[start, end)` span of one job inside a flat result array. It
    centralizes `job_offsets[job_idx:job_idx+2]` decoding so callers can focus
    on aggregation logic instead of repeating offset indexing boilerplate.
    """
    offsets = np.asarray(job_offsets, dtype=np.int64).reshape((-1,))
    return int(offsets[job_idx]), int(offsets[job_idx + 1])


def _build_flat_batch_views(batch_output: Dict[str, Any]) -> Dict[str, Any]:
    """把 flat batch 输出整理成便于聚合的只读视图集合。

    `_forward_job_batch` 返回的是跨多个数组的 flat 结果对象。这个 helper 负责把常用
    字段统一转成稳定的 numpy views，避免每个聚合入口都重复写同一套 `np.asarray`
    和 reshape 逻辑，同时保证后续 span slicing 都落在一致的数据表示上。

    Normalize a flat batch output into the read-only array views used during
    aggregation. It centralizes the repeated `np.asarray` and reshape steps so
    every aggregation path slices job spans from the same stable representation.
    """
    flat_action_results = batch_output["flat_action_results"]
    flat_rtg_results = batch_output["flat_rtg_results"]
    return {
        "jobs": batch_output["jobs"],
        "env_idx_by_job": np.asarray(batch_output["env_idx_by_job"], dtype=np.int64),
        "job_types": batch_output["job_types"],
        "prepared_by_job": batch_output["prepared_by_job"],
        "ego_action_scales_by_job": batch_output["ego_action_scales_by_job"],
        "action_logits_by_job_vehicle": batch_output["action_logits_by_job_vehicle"],
        "action_veh_ids": np.asarray(flat_action_results["veh_id"], dtype=np.int64),
        "action_values": np.asarray(flat_action_results["values"], dtype=np.float32).reshape((-1, 2)),
        "action_job_offsets": flat_action_results["job_offsets"],
        "rtg_veh_ids": np.asarray(flat_rtg_results["veh_id"], dtype=np.int64),
        "rtg_values": np.asarray(flat_rtg_results["values"], dtype=np.float32).reshape((-1, 3)),
        "rtg_job_offsets": flat_rtg_results["job_offsets"],
        "processed_rtg_veh_ids": np.asarray(flat_rtg_results["processed_veh_ids"], dtype=np.int64),
        "processed_job_offsets": flat_rtg_results["processed_offsets"],
    }


def _append_flat_job_views_to_env_parts(
    env_parts: Dict[str, Any],
    *,
    job_idx: int,
    views: Dict[str, Any],
    ego_action_scale: float,
    excluded_action_veh_ids: Optional[set[int]] = None,
    excluded_rtg_veh_ids: Optional[set[int]] = None,
) -> None:
    """把一个 job 在 flat batch 中对应的视图片段追加到 env parts。

    这个 helper 只处理“同一个 job 在 action/rtg/processed 三个 flat 数组里的 span
    如何追加到 env 聚合容器”这件事。它让调用方专注于 job 类型和 env 路由判断，
    避免在多个热点路径里重复写 offset 解析、切片和列表追加。

    Append one job's span views from the flat batch arrays into an env-level
    aggregation container. Callers only decide routing by env/job type, while
    this helper owns the repeated offset parsing, slicing, and part-list
    appends for action, RTG, and processed-RTG arrays. Action and RTG filters are
    kept separate because joint ego side-channel outputs must suppress executable
    ego actions while preserving the owner ego RTG history row.
    """
    excluded_action_ids = excluded_action_veh_ids or set()
    excluded_action_ids_array = (
        np.asarray(list(excluded_action_ids), dtype=np.int64)
        if excluded_action_ids
        else None
    )
    excluded_rtg_ids = excluded_rtg_veh_ids or set()
    excluded_rtg_ids_array = (
        np.asarray(list(excluded_rtg_ids), dtype=np.int64)
        if excluded_rtg_ids
        else None
    )
    action_start, action_end = _job_span(views["action_job_offsets"], job_idx)
    if action_end > action_start:
        action_veh_ids = views["action_veh_ids"][action_start:action_end]
        action_values = views["action_values"][action_start:action_end]
        if excluded_action_ids_array is not None:
            keep_mask = ~np.isin(action_veh_ids, excluded_action_ids_array)
            action_veh_ids = action_veh_ids[keep_mask]
            action_values = action_values[keep_mask]
        if action_veh_ids.shape[0] > 0:
            env_parts["action_veh_ids_parts"].append(action_veh_ids)
            env_parts["action_values_parts"].append(action_values)

    rtg_start, rtg_end = _job_span(views["rtg_job_offsets"], job_idx)
    if rtg_end > rtg_start:
        rtg_veh_ids = views["rtg_veh_ids"][rtg_start:rtg_end]
        rtg_values = views["rtg_values"][rtg_start:rtg_end]
        if excluded_rtg_ids_array is not None:
            keep_mask = ~np.isin(rtg_veh_ids, excluded_rtg_ids_array)
            rtg_veh_ids = rtg_veh_ids[keep_mask]
            rtg_values = rtg_values[keep_mask]
        if rtg_veh_ids.shape[0] > 0:
            env_parts["rtg_veh_ids_parts"].append(rtg_veh_ids)
            env_parts["rtg_values_parts"].append(rtg_values)

    processed_start, processed_end = _job_span(views["processed_job_offsets"], job_idx)
    if processed_end > processed_start:
        processed_veh_ids = views["processed_rtg_veh_ids"][
            processed_start:processed_end
        ]
        if excluded_rtg_ids_array is not None:
            processed_veh_ids = processed_veh_ids[
                ~np.isin(processed_veh_ids, excluded_rtg_ids_array)
            ]
        if processed_veh_ids.shape[0] > 0:
            env_parts["processed_rtg_veh_ids_parts"].append(processed_veh_ids)

    if float(ego_action_scale) != 1.0:
        env_parts["ego_action_scale"] = float(ego_action_scale)


def _build_last_decode_row_by_job_and_vehicle(
    decode_meta: Dict[str, np.ndarray],
) -> Dict[Tuple[int, int], int]:
    """构造 `(job_idx, veh_id) -> 最后一行 decode row` 的查找表。

    delayed reweighting 会反复按 `(job_idx, ego_id)` 查询 decode row。这个 helper
    预先把查找过程摊平成一个字典，从而避免在热路径里重复做布尔筛选或扫描。

    Build a lookup from `(job_idx, veh_id)` to the last decode-row index. The
    delayed reweighting path repeatedly resolves `(job_idx, ego_id)`, so this
    helper precomputes the mapping and avoids repeated scans or boolean masks in
    the hot path.
    """
    job_idx = np.asarray(decode_meta["job_idx"], dtype=np.int64)
    veh_id = np.asarray(decode_meta["veh_id"], dtype=np.int64)
    return {
        (int(job_idx[row_idx]), int(veh_id[row_idx])): int(row_idx)
        for row_idx in range(job_idx.shape[0])
    }


def _numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    """将 numpy dtype 映射到 torch dtype。

    collator 会同时维护 numpy 视图与 host/device tensor，因此需要一个统一的 dtype
    转换入口，保证缓冲区复用时两侧类型严格一致。

    Map a numpy dtype to the matching torch dtype. The collator keeps numpy
    views alongside host/device tensors, so it needs one central conversion path
    to keep reused buffers strictly aligned across both representations.
    """
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return torch.bool
    if np.issubdtype(dtype, np.integer):
        if dtype.itemsize <= np.dtype(np.int32).itemsize:
            return torch.int32
        return torch.int64
    if dtype == np.dtype(np.float64):
        return torch.float64
    return torch.float32


def _allocate_host_collate_buffer(
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Tuple[torch.Tensor, np.ndarray]:
    """分配一对可复用的 host tensor / numpy 视图。

    collator 的填充路径仍然以 numpy 写入为主，而真正喂给模型的是 tensor。这个
    helper 统一创建共享底层存储的一对对象，让缓冲区复用逻辑不必分别处理两套分配。

    Allocate a reusable host tensor and its numpy view. The collator still
    writes primarily through numpy, while the model consumes tensors, so this
    helper creates both objects over shared storage in one place.
    """
    host_tensor = torch.empty(shape, dtype=_numpy_dtype_to_torch(dtype))
    return host_tensor, host_tensor.numpy()


def _concat_or_empty(parts: List[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
    """拼接一组同 dtype 数组；若为空则返回对应 dtype 的空数组。

    这个 helper 统一处理 decode metadata 里多段一维数组的收口逻辑，让调用方只关心
    收集 `parts`，而不用重复写空列表分支和类型转换。

    Concatenate a list of same-dtype arrays or return an empty array of the
    requested dtype. It centralizes the metadata finalization pattern so callers
    only need to collect `parts` without repeating empty-list handling.
    """
    if not parts:
        return np.zeros((0,), dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype, copy=False)


def _resolve_token_index(raw_token_index: int, seq_len: int) -> int:
    """将 prepared 中的 token_index 归一化到当前序列范围内。

    prepared payload 里允许用负索引表达“倒数第几个 token”。这个 helper 负责把它
    规范化到 `[0, seq_len - 1]` 范围，供 collate 时直接写入 batch 的 token 索引。

    Normalize the prepared token index into the valid sequence range. Prepared
    payloads may use negative indexing to mean "count from the end", and this
    helper converts that into `[0, seq_len - 1]` for batch collation.
    """
    if raw_token_index < 0:
        raw_token_index = seq_len + raw_token_index
    return max(0, min(raw_token_index, seq_len - 1))


def _build_rtg_row_metadata(
    *,
    valid_context_veh_ids: np.ndarray,
    data_veh_ids: np.ndarray,
    default_tilt: Tuple[int, int, int],
    tilt_by_veh_id: Dict[int, Tuple[int, int, int]],
    delayed_scale: float,
    is_opponent_job: bool,
    ego_id: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """为单个 job 的 RTG rows 构造 tilt 与 effective-scale 元数据。

    这个 helper 只关心一件事：给已经过滤掉 `idx_in_model < 0` 的 context rows
    写入 RTG decode 所需的 `goal_tilt / veh_tilt / road_tilt / effective_scale`。
    它通过 `veh_id -> row_indices` 的映射一次性定位匹配行，替代原先
    `np.isin + np.flatnonzero + Python loop` 的重复扫描，同时保持原语义：
    只有出现在 `data_veh_ids` 里的 context rows 会被写值，未匹配行维持
    `tilt=0`、`effective_scale=1.0`。

    Build tilt and effective-scale metadata for the RTG rows of one job. The
    helper takes the context rows that already survived the `idx_in_model >= 0`
    filtering step and writes the `goal_tilt / veh_tilt / road_tilt /
    effective_scale` arrays needed by RTG decode. It uses a
    `veh_id -> row_indices` mapping to locate matching rows in one pass,
    replacing the previous `np.isin + np.flatnonzero + Python loop` pattern
    while preserving semantics: only context rows whose vehicle id appears in
    `data_veh_ids` receive values, and unmatched rows stay at `tilt=0` with
    `effective_scale=1.0`.
    """
    row_count = int(valid_context_veh_ids.shape[0])
    goal_tilt = np.zeros((row_count,), dtype=np.int64)
    veh_tilt = np.zeros((row_count,), dtype=np.int64)
    road_tilt = np.zeros((row_count,), dtype=np.int64)
    effective_scale = np.ones((row_count,), dtype=np.float32)
    if row_count <= 0 or int(np.asarray(data_veh_ids).shape[0]) <= 0:
        return goal_tilt, veh_tilt, road_tilt, effective_scale

    row_indices_by_veh_id: Dict[int, List[int]] = {}
    for row_idx, veh_id in enumerate(np.asarray(valid_context_veh_ids, dtype=np.int64).tolist()):
        row_indices_by_veh_id.setdefault(int(veh_id), []).append(row_idx)

    for veh_id in np.asarray(data_veh_ids, dtype=np.int64).tolist():
        row_indices = row_indices_by_veh_id.get(int(veh_id))
        if not row_indices:
            continue
        goal_val, veh_val, road_val = tilt_by_veh_id.get(int(veh_id), default_tilt)
        goal_tilt[row_indices] = int(goal_val)
        veh_tilt[row_indices] = int(veh_val)
        road_tilt[row_indices] = int(road_val)
        if is_opponent_job and (ego_id is None or int(veh_id) != int(ego_id)):
            effective_scale[row_indices] = float(delayed_scale)

    return goal_tilt, veh_tilt, road_tilt, effective_scale
