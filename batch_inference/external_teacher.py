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
    export_action_logits_by_job_impl,
)
from .batch_decoder.forward_batch import batch_predict_rtgs_mode as _batch_predict_rtgs_mode
from .batch_decoder.forward_batch import forward_job_batch_impl
from .batch_decoder import rtg as rtg_impl
from .batch_ipc import pack_model_outputs, release_prepared_payload, unpack_prepared

def _assert_required_keys(payload: Dict[str, Any], required: Tuple[str, ...], payload_name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def _collect_focal_jobs(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result,
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
    ) -> None:
        self.device = device
        self.base_seed = base_seed
        self.inference_precision = inference_precision

        (
            self._autocast_enabled,
            self._autocast_dtype,
        ) = self._resolve_inference_precision()

        print(f"[ExternalTeacher] Loading CtRL-Sim model from {checkpoint_path}...")
        self.model = CtRLSim.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        print("[ExternalTeacher] Model loaded successfully.")
        print(f"[ExternalTeacher] Inference precision: {self.inference_precision}")

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

    def run_batched_forward_action_logits(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
    ) -> List[Optional[np.ndarray]]:
        """按 env 导出 raw action logits / Export raw action logits aligned by environment."""
        decoded_prepared = self._decode_prepared_batch(per_env_prepared)
        try:
            logits_by_env: List[Optional[np.ndarray]] = [None] * len(decoded_prepared)
            focal_jobs: List[Dict[str, Any]] = []
            for env_idx, prepared in enumerate(decoded_prepared):
                if prepared is None:
                    continue

                _assert_required_keys(
                    prepared,
                    ("status", "focal_batches"),
                    f"prepared env_idx={env_idx}",
                )
                status = prepared["status"]
                if status == "skip":
                    continue
                if status != "ok":
                    raise ValueError(
                        f"prepared env_idx={env_idx} has invalid status={status!r}"
                    )

                focal_batches = prepared["focal_batches"]
                if len(focal_batches) > 1:
                    raise ValueError(
                        "ego_ctrlsim action-logit export expects at most one focal batch per env."
                    )
                for focal_batch in focal_batches:
                    _assert_required_keys(
                        focal_batch,
                        ("focal_id", "motion_data_np"),
                        "prepared focal_batch",
                    )
                    focal_jobs.append(
                        {
                            "env_idx": env_idx,
                            "prepared": prepared,
                            "focal_batch": focal_batch,
                        }
                    )

            if not focal_jobs:
                return logits_by_env

            logits_by_job = self._forward_action_logits_job_batch(focal_jobs)
            if len(logits_by_job) != len(focal_jobs):
                raise ValueError("Action-logit export job/result count mismatch.")

            for job, logits in zip(focal_jobs, logits_by_job):
                env_idx = int(job["env_idx"])
                if logits_by_env[env_idx] is not None:
                    raise ValueError(
                        "ego_ctrlsim action-logit export expects at most one job per env."
                    )
                logits_by_env[env_idx] = logits

            return logits_by_env
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
            "action_results": {},
            "rtg_results": {},
            "processed_rtg_veh_ids": [],
            "dead_ids": prepared["dead_ids"],
        }

    def _collect_focal_jobs(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        return _collect_focal_jobs(
            per_env_prepared=per_env_prepared,
            results=results,
            build_empty_env_result=self._build_empty_env_result,
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
    ):
        return decode_action_stage_batched_impl(
            teacher=self,
            batched_data=batched_data,
            batch_meta=batch_meta,
        )

    @torch.no_grad()
    def _forward_action_logits_job_batch(
        self,
        jobs: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """导出每个 job 的 raw action logits。 / Export raw action logits for each job."""
        if not jobs:
            return []

        batched_data, batch_meta = self._collate_jobs_with_padding(jobs)
        if _batch_predict_rtgs_mode(jobs):
            batched_data, _, _ = self._decode_rtg_stage_batched(
                batched_data=batched_data,
                batch_meta=batch_meta,
                decode_rtg_jobs_batched_fn=rtg_impl.decode_rtg_jobs_batched_impl,
            )

        with self.model_forward_context():
            preds = self.model(batched_data, eval=True)
        action_logits = preds["action_preds"].float()

        logits_by_job = export_action_logits_by_job_impl(
            action_logits=action_logits,
            decode_meta=batch_meta["decode_meta"]["action"],
        )
        return [logits.cpu().numpy() for logits in logits_by_job]

    def _forward_job_batch(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return forward_job_batch_impl(
            teacher=self,
            jobs=jobs,
            batch_predict_rtgs_mode_fn=_batch_predict_rtgs_mode,
            decode_rtg_jobs_batched_fn=rtg_impl.decode_rtg_jobs_batched_impl,
        )
