"""
负责在主进程中聚合多个环境的 prepared 输入，并执行跨 env 的批量推理。
该模块处理作业收集、单批次模型前向、结果汇总以及 IPC 负载的进出边界。
Implements the main-process engine for aggregating prepared inputs and running cross-environment batched inference.
Handles job collection, single-batch model forward passes, result aggregation, and IPC payload boundaries.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

from ctrlsim_adapter.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from models.ctrl_sim import CtRLSim

from ctrlsim_adapter.policy_reweighting_helpers import (
    AdversarialRTGConfig,
    AdversarialRTGRunningStats,
    compute_scale_from_error,
    recover_current_ego_rtgs,
)

from .batch_decoder import rtg as rtg_impl
from .batch_decoder.forward_batch import (
    batch_predict_rtgs_mode as _batch_predict_rtgs_mode,
)
from .batch_decoder.forward_batch import forward_job_batch_impl
from .batch_decoder.rtg import (
    _SIDE_CHANNEL_RTG_STAGE_TAG,
    sample_tilted_rtg_side_channel_batched_impl,
)
from .batch_ipc import pack_model_outputs, release_prepared_payload, unpack_prepared
from .batch_ipc.prepared import (
    get_prepared_focal_id,
    get_prepared_focal_motion_data,
)
from .external_teacher_batch_collator import collate_jobs_with_padding
from .external_teacher_helper import (
    _append_flat_job_views_to_env_parts,
    _assert_required_keys,
    _build_flat_batch_views,
    _build_last_decode_row_by_job_and_vehicle,
    _collect_focal_jobs,
    _concat_or_empty_ids,
    _concat_or_empty_values,
    _config_get,
    _fill_empty_ok_env_results,
    _find_flat_rtg_value,
)


def build_external_teacher_kwargs(
    *,
    checkpoint_path: str,
    device: str,
    inference_precision: str = "fp32",
    config_source: Any = None,
) -> Dict[str, Any]:
    """Build a normalized ExternalTeacher kwargs dict from args/env/config sources."""
    policy_reweighting_config = _config_get(
        config_source,
        "policy_reweighting_config",
        None,
    )
    return {
        "checkpoint_path": checkpoint_path,
        "device": device,
        "inference_precision": inference_precision,
        "use_policy_reweighting": bool(
            _config_get(config_source, "use_policy_reweighting", False)
        ),
        "policy_reweighting_target": str(
            _config_get(config_source, "policy_reweighting_target", "rtg")
        ),
        "policy_reweighting_reward_scale": float(
            _config_get(
                policy_reweighting_config,
                "reward_scale",
                _config_get(config_source, "policy_reweighting_reward_scale", 1.0),
            )
        ),
        "policy_reweighting_epsilon": float(
            _config_get(
                policy_reweighting_config,
                "epsilon",
                _config_get(config_source, "policy_reweighting_epsilon", 1e-6),
            )
        ),
    }


class ExternalTeacher:
    """
    主进程 GPU 批量推理引擎。
    Main-process GPU batched inference engine.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        base_seed: int = 1,
        inference_precision: str = "fp32",
        use_policy_reweighting: bool = False,
        policy_reweighting_target: str = "rtg",
        policy_reweighting_reward_scale: float = 1.0,
        policy_reweighting_epsilon: float = 1e-6,
    ) -> None:
        self.device = device
        self.base_seed = base_seed
        self.inference_precision = inference_precision
        self.policy_reweighting_target = str(policy_reweighting_target)
        self.policy_reweighting_config = AdversarialRTGConfig(
            enabled=bool(use_policy_reweighting),
            reward_scale=float(policy_reweighting_reward_scale),
            epsilon=float(policy_reweighting_epsilon),
        )
        self._policy_reweighting_error_rms = RunningMeanStd(shape=())

        (
            self._autocast_enabled,
            self._autocast_dtype,
        ) = self._resolve_inference_precision()

        print(f"[ExternalTeacher] Loading CtRL-Sim model from {checkpoint_path}...")
        self.model = CtRLSim.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        self._cast_model_for_inference_precision()
        self.model.to(device)
        self.model.eval()
        print("[ExternalTeacher] Model loaded successfully.")
        print(f"[ExternalTeacher] Inference precision: {self.inference_precision}")
        print(
            f"[ExternalTeacher] Runtime param dtype: {self._get_runtime_param_dtype()}"
        )

        ckpt_cfg = self.model.cfg
        ds = ckpt_cfg.dataset.waymo
        mdl = ckpt_cfg.model

        self.rtg_discretization = ds.rtg_discretization
        self.accel_discretization = ds.accel_discretization
        self.steer_discretization = ds.steer_discretization
        self.min_accel = ds.min_accel
        self.max_accel = ds.max_accel
        self.min_steer = ds.min_steer
        self.max_steer = ds.max_steer
        self.min_rtg_pos = ds.min_rtg_pos
        self.max_rtg_pos = ds.max_rtg_pos
        self.min_rtg_veh = ds.min_rtg_veh
        self.max_rtg_veh = ds.max_rtg_veh
        self.min_rtg_road = ds.min_rtg_road
        self.max_rtg_road = ds.max_rtg_road
        self.num_reward_components = mdl.num_reward_components

        self._collate_numpy_buffers: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}

        # Compile after all config attributes are extracted so OptimizedModule
        self.model = torch.compile(self.model, dynamic=True)
        print("[ExternalTeacher] Model compiled with torch.compile (dynamic=True).")

    def validate_student_action_space(
        self,
        student_accel_discretization: int,
        student_steer_discretization: int,
    ) -> None:
        if (
            self.accel_discretization != student_accel_discretization
            or self.steer_discretization != student_steer_discretization
        ):
            raise ValueError(
                "Student and teacher action discretization mismatch: "
                f"student=({student_accel_discretization}, {student_steer_discretization}), "
                f"teacher=({self.accel_discretization}, {self.steer_discretization})."
            )

    def run_batched_forward(
        self, per_env_prepared: List[Optional[Dict[str, Any]]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        跨 env 聚合 prepared 输入，并执行批量推理与结果回收。
        Aggregate prepared inputs across environments, run batched inference pipeline,
        and scatter outputs back per environment.
        """
        with torch.inference_mode():
            decoded_prepared = self._decode_prepared_batch(per_env_prepared)
            try:
                results: List[Optional[Dict[str, Any]]] = [None] * len(decoded_prepared)
                focal_jobs = _collect_focal_jobs(
                    decoded_prepared,
                    results,
                    self._build_empty_env_result,
                )

                if not focal_jobs:
                    _fill_empty_ok_env_results(
                        decoded_prepared,
                        results,
                        self._build_empty_env_result,
                    )
                    return self._pack_outputs(results)

                batch_output = forward_job_batch_impl(
                    teacher=self,
                    jobs=focal_jobs,
                    batch_predict_rtgs_mode_fn=_batch_predict_rtgs_mode,
                    decode_rtg_jobs_batched_fn=rtg_impl.decode_rtg_jobs_batched_impl,
                )
                env_outputs = self._aggregate_job_outputs_by_env(
                    batch_output, decoded_prepared, results
                )
                return self._pack_outputs(env_outputs)
            finally:
                for prepared in decoded_prepared:
                    release_prepared_payload(prepared)

    def _resolve_inference_precision(self) -> Tuple[bool, Optional[torch.dtype]]:
        allowed = {"fp32", "amp_fp16", "amp_bf16"}
        if self.inference_precision not in allowed:
            raise ValueError(
                f"Unsupported inference_precision={self.inference_precision}. "
                f"Expected one of {sorted(allowed)}."
            )

        if self.inference_precision == "fp32":
            return False, None

        if not str(self.device).startswith("cuda"):
            raise ValueError(
                f"inference_precision={self.inference_precision} requires CUDA device, got device={self.device}."
            )

        if self.inference_precision == "amp_fp16":
            return True, torch.float16

        if (
            self.inference_precision == "amp_bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("amp_bf16 is not supported on this CUDA device.")
        return True, torch.bfloat16

    def _cast_model_for_inference_precision(self) -> None:
        if self.inference_precision == "amp_fp16":
            self.model = self.model.half()
        elif self.inference_precision == "amp_bf16":
            self.model = self.model.to(dtype=torch.bfloat16)
        else:
            self.model = self.model.float()

    def _get_runtime_param_dtype(self) -> torch.dtype:
        for param in self.model.parameters():
            if param.is_floating_point():
                return param.dtype
        raise RuntimeError("ExternalTeacher model has no floating-point parameters.")

    def model_forward_context(self):
        if not getattr(self, "_autocast_enabled", False):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    def _decode_prepared_batch(
        self, per_env_prepared: List[Optional[Dict[str, Any]]]
    ) -> List[Optional[Dict[str, Any]]]:
        decoded_prepared: List[Optional[Dict[str, Any]]] = []
        for prepared in per_env_prepared:
            if prepared is None:
                decoded_prepared.append(None)
                continue
            decoded_prepared.append(unpack_prepared(prepared))
        return decoded_prepared

    def _pack_outputs(
        self, outputs: List[Optional[Dict[str, Any]]]
    ) -> List[Optional[Dict[str, Any]]]:
        return [
            pack_model_outputs(output) if output is not None else None
            for output in outputs
        ]

    def _build_empty_env_result(
        self, prepared: Dict[str, Any], env_idx: int, status: str
    ) -> Dict[str, Any]:
        """构造单个 env 的空 flat 输出结果。

        Build the empty flat output result for one environment.
        """
        _assert_required_keys(
            prepared,
            ("step_t", "token_index", "dead_ids"),
            f"prepared env_idx={env_idx}",
        )
        return {
            "status": status,
            "env_idx": env_idx,
            "step_t": prepared["step_t"],
            "token_index": prepared["token_index"],
            "ego_action_scale": 1.0,
            "action_veh_ids": np.zeros((0,), dtype=np.int64),
            "action_values": np.zeros((0, 2), dtype=np.float32),
            "rtg_veh_ids": np.zeros((0,), dtype=np.int64),
            "rtg_values": np.zeros((0, 3), dtype=np.float32),
            "processed_rtg_veh_ids": np.zeros((0,), dtype=np.int64),
            "dead_ids": np.asarray(prepared["dead_ids"], dtype=np.int64),
        }

    def _build_env_output_parts(
        self,
        *,
        prepared: Dict[str, Any],
        env_idx: int,
    ) -> Dict[str, Any]:
        """构造某个 env 的聚合中间态容器。

        这个容器只保存 array views 的分段列表和最终需要写回的标量信息。批量前向的
        flat 结果会先按 job span 挂到这些分段列表里，等同一 env 的所有 job 都处理完
        后，再统一做一次拼接，避免旧实现里每来一个 job 就做一次 `np.concatenate`。

        Build the intermediate aggregation container for one environment. It
        stores only per-env array-part lists plus the final scalar fields. Flat
        batch outputs append job-span views into these lists first, then the env
        result is concatenated once at the end instead of incrementally on every
        job.
        """
        return {
            "prepared": prepared,
            "env_idx": env_idx,
            "action_veh_ids_parts": [],
            "action_values_parts": [],
            "rtg_veh_ids_parts": [],
            "rtg_values_parts": [],
            "processed_rtg_veh_ids_parts": [],
            "ego_action_scale": 1.0,
        }

    def _finalize_env_output_parts(self, env_parts: Dict[str, Any]) -> Dict[str, Any]:
        """将某个 env 的聚合中间态收口成最终 flat 输出。

        输入是 `_build_env_output_parts` 产生的分段容器；输出是和 IPC 契约一致的
        单 env flat 结果。该函数把“按 env 收集 parts”和“最终一次性拼接”这两件事
        明确拆开，让热点聚合路径更稳定也更容易测试。

        Finalize one environment's aggregation container into the flat output
        contract used by IPC. It keeps "collect parts by env" separate from
        "concatenate once at the end", which makes the hot aggregation path both
        faster and easier to test.
        """
        env_output = self._build_empty_env_result(
            prepared=env_parts["prepared"],
            env_idx=int(env_parts["env_idx"]),
            status="ok",
        )
        env_output["action_veh_ids"] = _concat_or_empty_ids(
            env_parts["action_veh_ids_parts"]
        )
        env_output["action_values"] = _concat_or_empty_values(
            env_parts["action_values_parts"],
            width=2,
        )
        env_output["rtg_veh_ids"] = _concat_or_empty_ids(env_parts["rtg_veh_ids_parts"])
        env_output["rtg_values"] = _concat_or_empty_values(
            env_parts["rtg_values_parts"],
            width=3,
        )
        env_output["processed_rtg_veh_ids"] = _concat_or_empty_ids(
            env_parts["processed_rtg_veh_ids_parts"]
        )
        env_output["ego_action_scale"] = float(env_parts["ego_action_scale"])
        return env_output

    def _get_policy_reweighting_error_rms(self) -> RunningMeanStd:
        """Return the shared running-stat accumulator for RTG mismatch error."""
        rms = getattr(self, "_policy_reweighting_error_rms", None)
        if rms is None:
            rms = RunningMeanStd(shape=())
            self._policy_reweighting_error_rms = rms
        return rms

    def _get_policy_reweighting_running_stats(self) -> AdversarialRTGRunningStats:
        """Read the current online normalization statistics for RTG error."""
        rms = self._get_policy_reweighting_error_rms()
        return AdversarialRTGRunningStats(
            error_mean=float(np.asarray(rms.mean).reshape(())),
            error_sigma=float(np.sqrt(np.asarray(rms.var).reshape(()))),
            error_count=float(rms.count),
        )

    def _update_policy_reweighting_error_stats(self, error_value: float) -> None:
        """Update the shared RTG-error running statistics with one new sample."""
        if not self.policy_reweighting_config.enabled:
            return
        rms = self._get_policy_reweighting_error_rms()
        rms.update(np.asarray([error_value], dtype=np.float64))

    def _compute_ego_action_scales_by_job(
        self,
        *,
        jobs: List[Dict[str, Any]],
        rtg_logits: torch.Tensor,
        decode_meta: Dict[str, np.ndarray],
        flat_rtg_results: Dict[str, Any],
    ) -> List[float]:
        """Compute one ego_action_scale per job, owner focal only."""
        scales = [1.0] * len(jobs)
        if not jobs:
            return scales

        row_by_job_and_vehicle = _build_last_decode_row_by_job_and_vehicle(decode_meta)
        idx_in_model = np.asarray(decode_meta["idx_in_model"], dtype=np.int64)
        token_index = np.asarray(decode_meta["token_index"], dtype=np.int64)
        sampling_seed = np.asarray(decode_meta["sampling_seed"], dtype=np.uint64)
        step_t = np.asarray(decode_meta["step_t"], dtype=np.int64)

        valid_job_indices: List[int] = []
        valid_row_indices: List[int] = []
        valid_ego_ids: List[int] = []
        current_rtg_rows: List[np.ndarray] = []
        ego_reweight_tilts: List[Tuple[int, int, int]] = []

        for job_idx, job in enumerate(jobs):
            prepared = job["prepared"]
            focal_idx = int(job["focal_idx"])
            owner_focal_id = prepared.get("ego_context_owner_focal_id")
            if owner_focal_id is None or get_prepared_focal_id(
                prepared, focal_idx
            ) != int(owner_focal_id):
                continue

            ego_id = prepared.get("ego_id")
            if ego_id is None:
                continue

            if _find_flat_rtg_value(flat_rtg_results, job_idx, int(ego_id)) is None:
                continue

            row_idx = row_by_job_and_vehicle.get((int(job_idx), int(ego_id)))
            if row_idx is None:
                continue

            current_rtg_rows.append(
                get_prepared_focal_motion_data(prepared, focal_idx)["rtgs"][
                    int(idx_in_model[row_idx]),
                    int(token_index[row_idx]),
                ]
            )
            valid_job_indices.append(job_idx)
            valid_row_indices.append(row_idx)
            valid_ego_ids.append(int(ego_id))
            ego_reweight_tilts.append(
                tuple(int(v) for v in prepared.get("ego_reweight_tilt", (0, 0, 0)))
            )

        if not valid_row_indices:
            return scales

        valid_row_index_arr = np.asarray(valid_row_indices, dtype=np.int64)
        valid_job_index_arr = np.asarray(valid_job_indices, dtype=np.int64)
        valid_ego_id_arr = np.asarray(valid_ego_ids, dtype=np.int64)
        goal_tilt = np.asarray([tilt[0] for tilt in ego_reweight_tilts], dtype=np.int64)
        veh_tilt = np.asarray([tilt[1] for tilt in ego_reweight_tilts], dtype=np.int64)
        road_tilt = np.asarray([tilt[2] for tilt in ego_reweight_tilts], dtype=np.int64)

        valid_job_index_t = torch.as_tensor(
            valid_job_index_arr,
            dtype=torch.long,
            device=rtg_logits.device,
        )
        valid_idx_in_model_t = torch.as_tensor(
            idx_in_model[valid_row_index_arr],
            dtype=torch.long,
            device=rtg_logits.device,
        )
        valid_token_index_t = torch.as_tensor(
            token_index[valid_row_index_arr],
            dtype=torch.long,
            device=rtg_logits.device,
        )
        flat_rtg_logits = rtg_logits[
            valid_job_index_t,
            valid_idx_in_model_t,
            valid_token_index_t,
        ]
        tilted_current_rtgs = sample_tilted_rtg_side_channel_batched_impl(
            teacher=self,
            flat_rtg_logits=flat_rtg_logits,
            goal_tilt=goal_tilt,
            veh_tilt=veh_tilt,
            road_tilt=road_tilt,
            sampling_seed=sampling_seed[valid_row_index_arr],
            step_t=step_t[valid_row_index_arr],
            veh_id=valid_ego_id_arr,
            stage_tag=_SIDE_CHANNEL_RTG_STAGE_TAG,
        )
        current_rtgs_arr = recover_current_ego_rtgs(
            np.asarray(current_rtg_rows),
            rtg_discretization=self.rtg_discretization,
            min_rtg_pos=self.min_rtg_pos,
            max_rtg_pos=self.max_rtg_pos,
            min_rtg_veh=self.min_rtg_veh,
            max_rtg_veh=self.max_rtg_veh,
            min_rtg_road=self.min_rtg_road,
            max_rtg_road=self.max_rtg_road,
        )
        error_values = np.sum(
            np.square(current_rtgs_arr - tilted_current_rtgs),
            axis=1,
            dtype=np.float32,
        )

        for idx, job_idx in enumerate(valid_job_indices):
            running_stats = self._get_policy_reweighting_running_stats()
            error_value = float(error_values[idx])
            if self.policy_reweighting_config.enabled and any(
                float(value) != 0.0 for value in ego_reweight_tilts[idx]
            ):
                scales[job_idx] = compute_scale_from_error(
                    error_value=error_value,
                    config=self.policy_reweighting_config,
                    running_stats=running_stats,
                )
            self._update_policy_reweighting_error_stats(error_value)

        return scales

    def _aggregate_job_outputs_by_env(
        self,
        batch_output: Dict[str, Any],
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> List[Optional[Dict[str, Any]]]:
        """按 env 一次性聚合 flat batch 输出，并回填 per-env 结果。

        `_forward_job_batch` 现在返回统一的 flat batch 结果对象，而不是按 job 切开的
        小字典列表。这里按 job offset 取 array views，先挂到 env 级 parts 容器里，
        最后对每个 env 做一次性拼接，避免旧实现中重复切片和增量 concat 的开销。

        Aggregate the flat batch output back into per-env results. The new
        `_forward_job_batch` returns one flat batch object, so this helper reads
        job spans via offsets, appends those array views into env-level part
        containers, and performs one final concatenate per env instead of
        repeated slicing and incremental concatenation.
        """
        views = _build_flat_batch_views(batch_output)
        env_accum: Dict[int, Dict[str, Any]] = {}

        for job_idx, _job in enumerate(views["jobs"]):
            if str(views["job_types"][job_idx]) != "opponent":
                continue
            env_idx = int(views["env_idx_by_job"][job_idx])
            env_parts = env_accum.setdefault(
                env_idx,
                self._build_env_output_parts(
                    prepared=views["prepared_by_job"][job_idx],
                    env_idx=env_idx,
                ),
            )
            _append_flat_job_views_to_env_parts(
                env_parts,
                job_idx=job_idx,
                views=views,
                ego_action_scale=views["ego_action_scales_by_job"][job_idx],
            )

        for env_idx, env_parts in env_accum.items():
            results[env_idx] = self._finalize_env_output_parts(env_parts)

        _fill_empty_ok_env_results(
            per_env_prepared,
            results,
            self._build_empty_env_result,
        )
        return results

    def _collate_jobs_with_padding(self, jobs: List[Dict[str, Any]]):
        return collate_jobs_with_padding(
            jobs=jobs,
            device=self.device,
            collate_numpy_buffers=self._collate_numpy_buffers,
        )

    def run_batched_forward_with_ego_logits(
        self,
        opponent_prepared_batch: List[Optional[Dict[str, Any]]],
        ego_prepared_batch: List[Optional[Dict[str, Any]]],
    ) -> Tuple[
        List[Optional[Dict[str, Any]]],
        List[Optional[np.ndarray]],
        List[Optional[np.ndarray]],
        List[Optional[Dict[str, int]]],
    ]:
        with torch.inference_mode():
            decoded_opponent = self._decode_prepared_batch(opponent_prepared_batch)
            decoded_ego = self._decode_prepared_batch(ego_prepared_batch)
            try:
                opponent_results: List[Optional[Dict[str, Any]]] = [None] * len(
                    decoded_opponent
                )
                ego_logits_by_env: List[Optional[np.ndarray]] = [None] * len(
                    decoded_ego
                )
                ego_rtgs_by_env: List[Optional[np.ndarray]] = [None] * len(decoded_ego)
                ego_rtg_metadata_by_env: List[Optional[Dict[str, int]]] = [
                    None
                ] * len(decoded_ego)

                opponent_jobs = _collect_focal_jobs(
                    decoded_opponent,
                    opponent_results,
                    self._build_empty_env_result,
                    "opponent",
                    False,
                )
                ego_jobs = _collect_focal_jobs(
                    decoded_ego,
                    [None] * len(decoded_ego),
                    self._build_empty_env_result,
                    "ego_ctrlsim",
                    True,
                )
                combined_jobs = opponent_jobs + ego_jobs
                opponent_env_parts: Dict[int, Dict[str, Any]] = {}
                if combined_jobs:
                    for predict_rtgs_mode in (False, True):
                        bucket_jobs = [
                            job
                            for job in combined_jobs
                            if bool(job.get("predict_rtgs", True)) == predict_rtgs_mode
                        ]
                        if not bucket_jobs:
                            continue
                        batch_output = forward_job_batch_impl(
                            teacher=self,
                            jobs=bucket_jobs,
                            batch_predict_rtgs_mode_fn=_batch_predict_rtgs_mode,
                            decode_rtg_jobs_batched_fn=rtg_impl.decode_rtg_jobs_batched_impl,
                        )
                        views = _build_flat_batch_views(batch_output)

                        for job_idx, _job in enumerate(views["jobs"]):
                            env_idx = int(views["env_idx_by_job"][job_idx])
                            job_type = str(views["job_types"][job_idx])
                            scale = float(views["ego_action_scales_by_job"][job_idx])
                            if job_type == "ego_ctrlsim":
                                action_logits = views["action_logits_by_job"][job_idx]
                                ego_logits_by_env[env_idx] = (
                                    action_logits.cpu().numpy()
                                    if action_logits is not None
                                    else None
                                )
                                ego_id = views["prepared_by_job"][job_idx].get("ego_id")
                                if ego_id is None:
                                    focal_ids = np.asarray(
                                        views["prepared_by_job"][job_idx]["focal_ids"],
                                        dtype=np.int64,
                                    )
                                    ego_id = int(focal_ids[0])
                                if ego_id is not None:
                                    rtg_value = _find_flat_rtg_value(
                                        batch_output["flat_rtg_results"],
                                        job_idx=job_idx,
                                        veh_id=int(ego_id),
                                    )
                                    if rtg_value is not None:
                                        prepared = views["prepared_by_job"][job_idx]
                                        ego_rtgs_by_env[env_idx] = np.asarray(
                                            rtg_value,
                                            dtype=np.float32,
                                        ).reshape((3,))  # flatten to 1D
                                        ego_rtg_metadata_by_env[env_idx] = {
                                            "env_idx": int(env_idx),
                                            "step_t": int(prepared["step_t"]),
                                            "token_index": int(prepared["token_index"]),
                                            "ego_id": int(ego_id),
                                        }
                                if decoded_opponent[env_idx] is not None:
                                    env_parts = opponent_env_parts.setdefault(
                                        env_idx,
                                        self._build_env_output_parts(
                                            prepared=decoded_opponent[env_idx],
                                            env_idx=env_idx,
                                        ),
                                    )
                                    if scale != 1.0:
                                        env_parts["ego_action_scale"] = scale
                                continue

                            env_parts = opponent_env_parts.setdefault(
                                env_idx,
                                self._build_env_output_parts(
                                    prepared=views["prepared_by_job"][job_idx],
                                    env_idx=env_idx,
                                ),
                            )
                            _append_flat_job_views_to_env_parts(
                                env_parts,
                                job_idx=job_idx,
                                views=views,
                                ego_action_scale=scale,
                            )

                for env_idx, env_parts in opponent_env_parts.items():
                    opponent_results[env_idx] = self._finalize_env_output_parts(
                        env_parts
                    )

                _fill_empty_ok_env_results(
                    decoded_opponent,
                    opponent_results,
                    self._build_empty_env_result,
                )
                return (
                    self._pack_outputs(opponent_results),
                    ego_logits_by_env,
                    ego_rtgs_by_env,
                    ego_rtg_metadata_by_env,
                )
            finally:
                for prepared in decoded_opponent:
                    release_prepared_payload(prepared)
                for prepared in decoded_ego:
                    release_prepared_payload(prepared)
