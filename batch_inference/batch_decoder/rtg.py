"""
负责 RTG logits 的批量解码、车辆索引解析与向量化 writeback。
该模块将模型输出转换成按车辆组织的 RTG 结果，并在批量路径上消除逐 row 采样。
Handles batched RTG-logit decoding, vehicle-index resolution, and vectorized writeback.
Converts model outputs into per-vehicle RTG results while removing per-row sampling from the hot path.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from utils.data import MotionData

from .sampling import build_row_keys, build_stateless_uniforms, sample_categorical_from_uniform

_RTG_STAGE_TAG = 0
# Keep the side-channel RTG sample in its own stateless-sampling namespace so
# it never collides with baseline RTG decode or action decode tickets.
_SIDE_CHANNEL_RTG_STAGE_TAG = 2


def _build_flat_tilt_logits(
    teacher: Any,
    goal_tilt: np.ndarray,
    veh_tilt: np.ndarray,
    road_tilt: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cached_tilt_scale = getattr(teacher, "_cached_tilt_scale", None)
    if (
        cached_tilt_scale is None
        or cached_tilt_scale.dtype != dtype
        or cached_tilt_scale.device != device
        or cached_tilt_scale.shape != (1, int(teacher.rtg_discretization))
    ):
        cached_tilt_scale = torch.linspace(
            0.0,
            1.0,
            int(teacher.rtg_discretization),
            dtype=dtype,
            device=device,
        ).unsqueeze(0)
        teacher._cached_tilt_scale = cached_tilt_scale
    tilt_scale = cached_tilt_scale
    goal = torch.as_tensor(goal_tilt, dtype=dtype, device=device).unsqueeze(-1) * tilt_scale
    veh = torch.as_tensor(veh_tilt, dtype=dtype, device=device).unsqueeze(-1) * tilt_scale
    road = torch.as_tensor(road_tilt, dtype=dtype, device=device).unsqueeze(-1) * tilt_scale
    return torch.stack((goal, veh, road), dim=-1)


def _undiscretize_rtg_indices_batched(
    teacher: Any,
    discrete_idx: torch.Tensor,
) -> torch.Tensor:
    continuous = discrete_idx.to(dtype=torch.float32) / float(teacher.rtg_discretization - 1)
    continuous[:, 0] = continuous[:, 0] * float(teacher.max_rtg_pos - teacher.min_rtg_pos) + float(teacher.min_rtg_pos)
    continuous[:, 1] = continuous[:, 1] * float(teacher.max_rtg_veh - teacher.min_rtg_veh) + float(teacher.min_rtg_veh)
    continuous[:, 2] = continuous[:, 2] * float(teacher.max_rtg_road - teacher.min_rtg_road) + float(teacher.min_rtg_road)
    return continuous


def _sample_rtg_indices(
    teacher: Any,
    flat_rtg_logits: torch.Tensor,
    flat_tilt_logits: torch.Tensor,
    sampling_seed: np.ndarray,
    step_t: np.ndarray,
    veh_id: np.ndarray,
    stage_tag: int = _RTG_STAGE_TAG,
) -> torch.Tensor:
    probs = F.softmax((flat_rtg_logits + flat_tilt_logits).permute(0, 2, 1), dim=-1)
    row_keys = build_row_keys(step_t=step_t, veh_ids=veh_id, stage_tag=stage_tag)
    uniforms = build_stateless_uniforms(
        base_seed=sampling_seed,
        row_keys=row_keys,
        draws_per_row=3,
        as_tensor=True,
        device=flat_rtg_logits.device,
        dtype=flat_rtg_logits.dtype,
    )
    return sample_categorical_from_uniform(
        probs.reshape(-1, probs.shape[-1]),
        uniforms.reshape(-1),
    ).reshape(-1, teacher.num_reward_components)


def sample_tilted_rtg_side_channel_impl(
    teacher: Any,
    rtg_logits_row: torch.Tensor,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
    sampling_seed: int,
    step_t: int,
    veh_id: int,
    stage_tag: int = _SIDE_CHANNEL_RTG_STAGE_TAG,
) -> np.ndarray:
    """Sample a tilted RTG side-channel without mutating baseline RTG writeback."""
    logits_3 = rtg_logits_row.reshape(
        teacher.rtg_discretization,
        teacher.num_reward_components,
    )
    tilt_logits = _build_flat_tilt_logits(
        teacher=teacher,
        goal_tilt=np.asarray([goal_tilt], dtype=np.int64),
        veh_tilt=np.asarray([veh_tilt], dtype=np.int64),
        road_tilt=np.asarray([road_tilt], dtype=np.int64),
        dtype=logits_3.dtype,
        device=logits_3.device,
    )
    discrete = _sample_rtg_indices(
        teacher=teacher,
        flat_rtg_logits=logits_3.unsqueeze(0),
        flat_tilt_logits=tilt_logits,
        sampling_seed=np.asarray([sampling_seed], dtype=np.uint64),
        step_t=np.asarray([step_t], dtype=np.int64),
        veh_id=np.asarray([veh_id], dtype=np.int64),
        stage_tag=stage_tag,
    )
    continuous = _undiscretize_rtg_indices_batched(
        teacher=teacher,
        discrete_idx=discrete,
    )
    return continuous[0].detach().cpu().numpy().astype(np.float32, copy=False)


def decode_rtg_jobs_batched_impl(
    teacher: Any,
    batched_data: MotionData,
    rtg_logits: torch.Tensor,
    decode_meta: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """批量解码 RTG logits，并返回 flat 结果数组。

    Decode RTG logits in batch and return the flat result arrays.
    """
    job_offsets = decode_meta["job_offsets"]
    job_count = int(decode_meta["job_count"][0])
    if decode_meta["job_idx"].size == 0:
        empty_offsets = np.zeros((job_count + 1,), dtype=np.int64)
        return {
            "veh_id": np.zeros((0,), dtype=np.int64),
            "values": np.zeros((0, 3), dtype=np.float32),
            "job_offsets": empty_offsets,
            "processed_veh_ids": np.zeros((0,), dtype=np.int64),
            "processed_offsets": empty_offsets,
            "job_count": job_count,
        }

    device = rtg_logits.device
    env_idx = np.asarray(decode_meta["env_idx"], dtype=np.int64)
    veh_id = np.asarray(decode_meta["veh_id"], dtype=np.int64)
    key_scale = max(1, int(veh_id.max(initial=0)) + 1)
    cache_keys_t = torch.as_tensor(
        env_idx * key_scale + veh_id,
        dtype=torch.long,
        device=device,
    )
    unique_cache_keys, inverse_t = torch.unique(
        cache_keys_t,
        sorted=True,
        return_inverse=True,
    )
    unique_row_indices_t = torch.empty(
        (unique_cache_keys.shape[0],),
        dtype=torch.long,
        device=device,
    )
    for row_idx in range(int(cache_keys_t.shape[0]) - 1, -1, -1):
        unique_row_indices_t[inverse_t[row_idx]] = row_idx
    unique_row_indices = unique_row_indices_t.detach().cpu().numpy()
    inverse = inverse_t.detach().cpu().numpy()

    unique_job_idx = decode_meta.get("job_idx_t")
    unique_idx_in_model = decode_meta.get("idx_in_model_t")
    unique_token_index = decode_meta.get("token_index_t")
    if unique_job_idx is None or unique_idx_in_model is None or unique_token_index is None:
        unique_job_idx = torch.as_tensor(decode_meta["job_idx"], dtype=torch.long, device=device)
        unique_idx_in_model = torch.as_tensor(decode_meta["idx_in_model"], dtype=torch.long, device=device)
        unique_token_index = torch.as_tensor(decode_meta["token_index"], dtype=torch.long, device=device)
    unique_job_idx = unique_job_idx[unique_row_indices_t]
    unique_idx_in_model = unique_idx_in_model[unique_row_indices_t]
    unique_token_index = unique_token_index[unique_row_indices_t]
    flat_rtg_logits = rtg_logits[unique_job_idx, unique_idx_in_model, unique_token_index].reshape(
        -1,
        teacher.rtg_discretization,
        teacher.num_reward_components,
    )
    flat_tilt_logits = _build_flat_tilt_logits(
        teacher=teacher,
        goal_tilt=decode_meta["goal_tilt"][unique_row_indices],
        veh_tilt=decode_meta["veh_tilt"][unique_row_indices],
        road_tilt=decode_meta["road_tilt"][unique_row_indices],
        dtype=flat_rtg_logits.dtype,
        device=device,
    )
    if getattr(teacher, "policy_reweighting_target", "rtg") == "rtg":
        delayed_scale_all = np.asarray(
            decode_meta.get(
                "delayed_scale",
                np.ones((decode_meta["job_idx"].shape[0],), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        delayed_active_all = np.asarray(
            decode_meta.get(
                "delayed_active",
                np.zeros((decode_meta["job_idx"].shape[0],), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        unique_delayed_scale = np.ones(
            (unique_row_indices.shape[0],),
            dtype=np.float32,
        )
        unique_delayed_active = np.zeros(
            (unique_row_indices.shape[0],),
            dtype=np.bool_,
        )
        for row_idx, unique_idx in enumerate(inverse):
            if not delayed_active_all[row_idx]:
                continue
            unique_delayed_scale[int(unique_idx)] = float(delayed_scale_all[row_idx])
            unique_delayed_active[int(unique_idx)] = True
        unique_scale_t = torch.as_tensor(
            np.where(unique_delayed_active, unique_delayed_scale, 1.0),
            dtype=flat_rtg_logits.dtype,
            device=device,
        )
        flat_rtg_logits = flat_rtg_logits * unique_scale_t.view(-1, 1, 1)
    discrete_unique = _sample_rtg_indices(
        teacher=teacher,
        flat_rtg_logits=flat_rtg_logits,
        flat_tilt_logits=flat_tilt_logits,
        sampling_seed=decode_meta["sampling_seed"][unique_row_indices],
        step_t=decode_meta["step_t"][unique_row_indices],
        veh_id=decode_meta["veh_id"][unique_row_indices],
    )
    continuous_unique = _undiscretize_rtg_indices_batched(
        teacher=teacher,
        discrete_idx=discrete_unique,
    )

    discrete_all = discrete_unique[inverse_t]
    write_batch_idx = decode_meta.get("job_idx_t")
    if write_batch_idx is None:
        write_batch_idx = torch.as_tensor(decode_meta["job_idx"], dtype=torch.long, device=device)
    write_idx_in_model = decode_meta.get("idx_in_model_t")
    if write_idx_in_model is None:
        write_idx_in_model = torch.as_tensor(decode_meta["idx_in_model"], dtype=torch.long, device=device)
    write_token_index = decode_meta.get("token_index_t")
    if write_token_index is None:
        write_token_index = torch.as_tensor(decode_meta["token_index"], dtype=torch.long, device=device)
    write_discrete = discrete_all.to(dtype=batched_data["agent"].rtgs.dtype)
    batched_data["agent"].rtgs[write_batch_idx, write_idx_in_model, write_token_index, 0] = write_discrete[:, 0]
    batched_data["agent"].rtgs[write_batch_idx, write_idx_in_model, write_token_index, 1] = write_discrete[:, 1]
    batched_data["agent"].rtgs[write_batch_idx, write_idx_in_model, write_token_index, 2] = write_discrete[:, 2]

    ordered_unique_indices = np.argsort(unique_row_indices)
    sorted_row_indices = unique_row_indices[ordered_unique_indices]
    sorted_job_idx = np.asarray(decode_meta["job_idx"], dtype=np.int64)[sorted_row_indices]
    sorted_veh_id = np.asarray(decode_meta["veh_id"], dtype=np.int64)[sorted_row_indices]
    sorted_values = (
        continuous_unique[torch.as_tensor(ordered_unique_indices, dtype=torch.long, device=continuous_unique.device)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    per_job_counts = np.bincount(sorted_job_idx, minlength=job_count).astype(np.int64, copy=False)
    result_offsets = np.zeros((job_count + 1,), dtype=np.int64)
    result_offsets[1:] = np.cumsum(per_job_counts)
    return {
        "veh_id": sorted_veh_id,
        "values": sorted_values,
        "job_offsets": result_offsets,
        "processed_veh_ids": sorted_veh_id.copy(),
        "processed_offsets": result_offsets.copy(),
        "job_count": job_count,
    }
