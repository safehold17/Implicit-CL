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

from ctrlsim_adapter.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from models.ctrl_sim import CtRLSim

from .external_teacher_batch_collator import collate_jobs_with_padding
from .batch_decoder.action import (
    decode_action_stage_batched_impl,
)
from .batch_decoder.forward_batch import batch_predict_rtgs_mode as _batch_predict_rtgs_mode
from .batch_decoder.forward_batch import forward_job_batch_impl
from .batch_decoder import rtg as rtg_impl
from .batch_decoder.rtg import sample_tilted_rtg_side_channel_impl
from .batch_ipc import pack_model_outputs, release_prepared_payload, unpack_prepared
from ctrlsim_adapter.policy_reweighting_helpers import (
    AdversarialRTGConfig,
    compute_ego_action_scale,
    recover_current_ego_rtg,
)

def _assert_required_keys(payload: Dict[str, Any], required: Tuple[str, ...], payload_name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def _config_get(source: Any, key: str, default: Any) -> Any:
    """Read a configuration value from either a mapping or an object."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


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
        "opponent_policy_reweighting_enabled": bool(
            _config_get(config_source, "opponent_policy_reweighting_enabled", False)
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
        "policy_reweighting_error_mean": float(
            _config_get(
                policy_reweighting_config,
                "error_mean",
                _config_get(config_source, "policy_reweighting_error_mean", 0.0),
            )
        ),
        "policy_reweighting_error_sigma": float(
            _config_get(
                policy_reweighting_config,
                "error_sigma",
                _config_get(config_source, "policy_reweighting_error_sigma", 1.0),
            )
        ),
    }


def _collect_focal_jobs(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result,
    job_type: str = "opponent",
    return_action_logits: bool = False,
) -> List[Dict[str, Any]]:
    focal_jobs: List[Dict[str, Any]] = []
    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None:
            continue

        _assert_required_keys(
            prepared,
            ("status", "step_t", "token_index", "dead_ids"),
            f"prepared env_idx={env_idx}",
        )
        status = prepared["status"]
        if status == "skip":
            results[env_idx] = build_empty_env_result(prepared, env_idx=env_idx, status="skip")
            continue
        if status != "ok":
            raise ValueError(f"prepared env_idx={env_idx} has invalid status={status!r}")

        _assert_required_keys(prepared, ("focal_batches",), f"prepared env_idx={env_idx}")
        for focal_batch in prepared["focal_batches"]:
            _assert_required_keys(focal_batch, ("focal_id", "motion_data_np"), "prepared focal_batch")
            focal_jobs.append(
                {
                    "env_idx": env_idx,
                    "prepared": prepared,
                    "focal_batch": focal_batch,
                    "job_type": job_type,
                    "return_action_logits": return_action_logits,
                }
            )

    return focal_jobs


def _fill_empty_ok_env_results(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result,
) -> None:
    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None or prepared["status"] != "ok":
            continue
        if results[env_idx] is None:
            results[env_idx] = build_empty_env_result(prepared, env_idx=env_idx, status="ok")


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
        opponent_policy_reweighting_enabled: bool = False,
        policy_reweighting_reward_scale: float = 1.0,
        policy_reweighting_epsilon: float = 1e-6,
        policy_reweighting_error_mean: float = 0.0,
        policy_reweighting_error_sigma: float = 1.0,
    ) -> None:
        self.device = device
        self.base_seed = base_seed
        self.inference_precision = inference_precision
        self.policy_reweighting_config = AdversarialRTGConfig(
            enabled=bool(opponent_policy_reweighting_enabled),
            reward_scale=float(policy_reweighting_reward_scale),
            epsilon=float(policy_reweighting_epsilon),
            error_mean=float(policy_reweighting_error_mean),
            error_sigma=float(policy_reweighting_error_sigma),
        )

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
            "[ExternalTeacher] Runtime param dtype: "
            f"{self._get_runtime_param_dtype()}"
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

    def run_batched_forward(self, per_env_prepared: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """
        跨 env 聚合 prepared 输入，并执行批量推理与结果回收。
        Aggregate prepared inputs across environments, run batched inference pipeline,
        and scatter outputs back per environment.
        """
        decoded_prepared = self._decode_prepared_batch(per_env_prepared)
        try:
            results: List[Optional[Dict[str, Any]]] = [None] * len(decoded_prepared)
            focal_jobs = self._collect_focal_jobs(decoded_prepared, results)

            if not focal_jobs:
                self._fill_empty_ok_env_results(decoded_prepared, results)
                return self._pack_outputs(results)

            job_outputs = self._forward_job_batch(focal_jobs)
            env_outputs = self._aggregate_job_outputs_by_env(job_outputs, decoded_prepared, results)
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

        if self.inference_precision == "amp_bf16" and not torch.cuda.is_bf16_supported():
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
        if not self._autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    def _decode_prepared_batch(self, per_env_prepared: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        decoded_prepared: List[Optional[Dict[str, Any]]] = []
        for prepared in per_env_prepared:
            if prepared is None:
                decoded_prepared.append(None)
                continue
            decoded_prepared.append(unpack_prepared(prepared))
        return decoded_prepared

    def _pack_outputs(self, outputs: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        return [pack_model_outputs(output) if output is not None else None for output in outputs]

    def _build_empty_env_result(self, prepared: Dict[str, Any], env_idx: int, status: str) -> Dict[str, Any]:
        _assert_required_keys(prepared, ("step_t", "token_index", "dead_ids"), f"prepared env_idx={env_idx}")
        return {
            "status": status,
            "env_idx": env_idx,
            "step_t": prepared["step_t"],
            "token_index": prepared["token_index"],
            "ego_action_scale": 1.0,
            "action_results": {},
            "rtg_results": {},
            "processed_rtg_veh_ids": [],
            "dead_ids": prepared["dead_ids"],
        }

    def _compute_ego_action_scale_for_job(
        self,
        *,
        job: Dict[str, Any],
        job_idx: int,
        rtg_logits: torch.Tensor,
        decode_meta: Dict[str, np.ndarray],
        rtg_results: Dict[int, Tuple[float, float, float]],
    ) -> float:
        """Compute ego_action_scale for the owner focal of one job."""
        prepared = job["prepared"]
        focal_batch = job["focal_batch"]
        owner_focal_id = prepared.get("ego_context_owner_focal_id")
        if owner_focal_id is None or int(focal_batch["focal_id"]) != int(owner_focal_id):
            return 1.0

        ego_id = prepared.get("ego_id")
        if ego_id is None:
            return 1.0
        next_rtg = rtg_results.get(int(ego_id))
        if next_rtg is None:
            return 1.0

        row_mask = (
            (np.asarray(decode_meta["job_idx"], dtype=np.int64) == int(job_idx))
            & (np.asarray(decode_meta["veh_id"], dtype=np.int64) == int(ego_id))
        )
        row_indices = np.nonzero(row_mask)[0]
        if row_indices.size == 0:
            return 1.0
        row_idx = int(row_indices[-1])
        idx_in_model = int(np.asarray(decode_meta["idx_in_model"], dtype=np.int64)[row_idx])
        token_index = int(np.asarray(decode_meta["token_index"], dtype=np.int64)[row_idx])

        current_rtg_raw = np.asarray(
            focal_batch["motion_data_np"]["rtgs"][idx_in_model, token_index]
        )
        current_rtg = recover_current_ego_rtg(
            current_rtg_raw,
            rtg_discretization=self.rtg_discretization,
            min_rtg_pos=self.min_rtg_pos,
            max_rtg_pos=self.max_rtg_pos,
            min_rtg_veh=self.min_rtg_veh,
            max_rtg_veh=self.max_rtg_veh,
            min_rtg_road=self.min_rtg_road,
            max_rtg_road=self.max_rtg_road,
        )
        ego_reweight_tilt = tuple(
            int(v) for v in prepared.get("ego_reweight_tilt", (0, 0, 0))
        )
        tilted_current_rtg = sample_tilted_rtg_side_channel_impl(
            teacher=self,
            rtg_logits_row=rtg_logits[job_idx, idx_in_model, token_index],
            goal_tilt=ego_reweight_tilt[0],
            veh_tilt=ego_reweight_tilt[1],
            road_tilt=ego_reweight_tilt[2],
            sampling_seed=int(np.asarray(decode_meta["sampling_seed"], dtype=np.uint64)[row_idx]),
            step_t=int(np.asarray(decode_meta["step_t"], dtype=np.int64)[row_idx]),
            veh_id=int(ego_id),
            stage_tag=1,
        )
        return compute_ego_action_scale(
            config=self.policy_reweighting_config,
            current_rtg=current_rtg,
            next_rtg=np.asarray(next_rtg, dtype=np.float32),
            tilted_current_rtg=np.asarray(tilted_current_rtg, dtype=np.float32),
            ego_reweight_tilt=ego_reweight_tilt,
        )

    def _compute_ego_action_scales_by_job(
        self,
        *,
        jobs: List[Dict[str, Any]],
        rtg_logits: torch.Tensor,
        decode_meta: Dict[str, np.ndarray],
        rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]],
    ) -> List[float]:
        """Compute one ego_action_scale per job, owner focal only."""
        return [
            self._compute_ego_action_scale_for_job(
                job=job,
                job_idx=job_idx,
                rtg_logits=rtg_logits,
                decode_meta=decode_meta,
                rtg_results=rtg_results_by_job[job_idx],
            )
            for job_idx, job in enumerate(jobs)
        ]

    def _collect_focal_jobs(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
        job_type: str = "opponent",
        return_action_logits: bool = False,
    ) -> List[Dict[str, Any]]:
        return _collect_focal_jobs(
            per_env_prepared=per_env_prepared,
            results=results,
            build_empty_env_result=self._build_empty_env_result,
            job_type=job_type,
            return_action_logits=return_action_logits,
        )

    def _fill_empty_ok_env_results(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> None:
        _fill_empty_ok_env_results(
            per_env_prepared=per_env_prepared,
            results=results,
            build_empty_env_result=self._build_empty_env_result,
        )

    def _aggregate_job_outputs_by_env(
        self,
        job_outputs: List[Dict[str, Any]],
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        将按 job 粒度的推理结果按 env 聚合，并回填到 per-env 结果列表中。
        该过程会合并 action / RTG 输出，同时为没有 job 输出但状态为 ok 的 env 补齐空结果。

        Aggregate job-level inference outputs by environment and write them back into the
        per-env result list.
        This merges action / RTG outputs and also fills empty results for environments whose
        status is `ok` but produced no job-level outputs.
        """
        env_accum: Dict[int, Dict[str, Any]] = {}
        for job_output in job_outputs:
            env_idx = int(job_output["env_idx"])
            if env_idx not in env_accum:
                env_accum[env_idx] = self._build_empty_env_result(
                    prepared=job_output["prepared"],
                    env_idx=env_idx,
                    status="ok",
                )

            env_accum_item = env_accum[env_idx]
            env_accum_item["action_results"].update(job_output["action_results"])
            env_accum_item["rtg_results"].update(job_output["rtg_results"])
            env_accum_item["processed_rtg_veh_ids"].extend(job_output["processed_rtg_veh_ids"])
            scale = float(job_output.get("ego_action_scale", 1.0))
            if scale != 1.0:
                env_accum_item["ego_action_scale"] = scale

        for env_idx, env_output in env_accum.items():
            results[env_idx] = env_output

        self._fill_empty_ok_env_results(per_env_prepared, results)
        return results

    def _collate_jobs_with_padding(self, jobs: List[Dict[str, Any]]):
        return collate_jobs_with_padding(
            jobs=jobs,
            device=self.device,
            collate_numpy_buffers=self._collate_numpy_buffers,
        )

    @torch.no_grad()
    def _decode_rtg_stage_batched(self, batched_data, batch_meta, decode_rtg_jobs_batched_fn):
        with self.model_forward_context():
            preds = self.model(batched_data, eval=True)
        rtg_logits = preds["rtg_preds"].float()
        rtg_results_by_job, processed_rtg_veh_ids_by_job = decode_rtg_jobs_batched_fn(
            teacher=self,
            batched_data=batched_data,
            rtg_logits=rtg_logits,
            decode_meta=batch_meta["decode_meta"]["rtg"],
        )
        return batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job

    @torch.no_grad()
    def _decode_action_stage_batched(
        self,
        batched_data,
        batch_meta,
        return_logits: bool = False,
        logits_job_indices=(),
    ):
        return decode_action_stage_batched_impl(
            teacher=self,
            batched_data=batched_data,
            batch_meta=batch_meta,
            return_logits=return_logits,
            logits_job_indices=logits_job_indices,
        )

    def _forward_job_batch(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return forward_job_batch_impl(
            teacher=self,
            jobs=jobs,
            batch_predict_rtgs_mode_fn=_batch_predict_rtgs_mode,
            decode_rtg_jobs_batched_fn=rtg_impl.decode_rtg_jobs_batched_impl,
        )

    def run_batched_forward_with_ego_logits(
        self,
        opponent_prepared_batch: List[Optional[Dict[str, Any]]],
        ego_prepared_batch: List[Optional[Dict[str, Any]]],
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[Optional[np.ndarray]]]:
        decoded_opponent = self._decode_prepared_batch(opponent_prepared_batch)
        decoded_ego = self._decode_prepared_batch(ego_prepared_batch)
        try:
            opponent_results: List[Optional[Dict[str, Any]]] = [None] * len(decoded_opponent)
            ego_logits_by_env: List[Optional[np.ndarray]] = [None] * len(decoded_ego)

            opponent_jobs = self._collect_focal_jobs(
                decoded_opponent,
                opponent_results,
                job_type="opponent",
                return_action_logits=False,
            )
            ego_jobs = self._collect_focal_jobs(
                decoded_ego,
                [None] * len(decoded_ego),
                job_type="ego_ctrlsim",
                return_action_logits=True,
            )
            combined_jobs = opponent_jobs + ego_jobs
            if combined_jobs:
                for predict_rtgs_mode in (False, True):
                    bucket_jobs = [
                        job
                        for job in combined_jobs
                        if bool(job["focal_batch"].get("predict_rtgs", True))
                        == predict_rtgs_mode
                    ]
                    if not bucket_jobs:
                        continue
                    for job_output in self._forward_job_batch(bucket_jobs):
                        job_type = str(job_output.get("job_type", "opponent"))
                        env_idx = int(job_output["env_idx"])
                        if job_type == "ego_ctrlsim":
                            action_logits = job_output.get("action_logits")
                            ego_logits_by_env[env_idx] = (
                                action_logits.cpu().numpy()
                                if action_logits is not None
                                else None
                            )
                            if opponent_results[env_idx] is None:
                                opponent_results[env_idx] = self._build_empty_env_result(
                                    prepared=job_output["prepared"],
                                    env_idx=env_idx,
                                    status="ok",
                                )
                            opponent_results[env_idx]["ego_action_scale"] = float(
                                job_output.get("ego_action_scale", 1.0)
                            )
                            continue

                        if opponent_results[env_idx] is None:
                            opponent_results[env_idx] = self._build_empty_env_result(
                                prepared=job_output["prepared"],
                                env_idx=env_idx,
                                status="ok",
                            )
                        opponent_results[env_idx]["action_results"].update(
                            job_output["action_results"]
                        )
                        opponent_results[env_idx]["rtg_results"].update(
                            job_output["rtg_results"]
                        )
                        opponent_results[env_idx]["processed_rtg_veh_ids"].extend(
                            job_output["processed_rtg_veh_ids"]
                        )
                        scale = float(job_output.get("ego_action_scale", 1.0))
                        if scale != 1.0:
                            opponent_results[env_idx]["ego_action_scale"] = scale

            self._fill_empty_ok_env_results(decoded_opponent, opponent_results)
            return self._pack_outputs(opponent_results), ego_logits_by_env
        finally:
            for prepared in decoded_opponent:
                release_prepared_payload(prepared)
            for prepared in decoded_ego:
                release_prepared_payload(prepared)
