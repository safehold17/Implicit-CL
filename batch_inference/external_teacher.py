"""ExternalTeacher: 主进程跨 env 批量推理引擎。"""

from __future__ import annotations

from contextlib import nullcontext
import os as _os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from adapters.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from models.ctrl_sim import CtRLSim

from .external_teacher_batch_collator import collate_chunk_with_padding
from .external_teacher_batch_decoder import (
    decode_action_stage_batched,
    decode_rtg_stage_batched,
    forward_chunk_batched,
    get_next_worker_rng_state,
)
from .external_teacher_job_scheduler import (
    build_chunks,
    collect_flat_jobs,
    estimate_job_tokens,
    fill_empty_ok_env_results,
)
from .ipc_codec import pack_model_outputs, release_prepared_payload, unpack_prepared

def _assert_required_keys(payload: Dict[str, Any], required: Tuple[str, ...], payload_name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def _merge_profile_sums(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            _merge_profile_sums(child, value)
            continue
        target[key] = float(target.get(key, 0.0)) + float(value)


def _aggregate_forward_chunk_profiles(chunk_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {
        "chunk_count": len(chunk_profiles),
        "stage_ms": {},
        "detail_ms": {},
    }
    for profile in chunk_profiles:
        _merge_profile_sums(aggregate["stage_ms"], profile.get("stage_ms", {}))
        _merge_profile_sums(aggregate["detail_ms"], profile.get("detail_ms", {}))
    return aggregate


class ExternalTeacher:
    """主进程 GPU 批量推理引擎。"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        micro_batch: Optional[int] = None,
        base_seed: int = 1,
        inference_precision: str = "fp32",
    ) -> None:
        self.device = device
        self.micro_batch = micro_batch
        self.base_seed = base_seed
        self.inference_precision = inference_precision
        self._profile_enabled = self._read_env_flag(
            "CTRLSIM_EXTERNAL_TEACHER_PROFILE",
            default="0",
        )
        self._profile_every = max(
            1,
            int(_os.getenv("CTRLSIM_EXTERNAL_TEACHER_PROFILE_EVERY", "50")),
        )
        self._profile_counter = 0

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
        self._last_forward_chunk_profile: Optional[Dict[str, Any]] = None
        self._last_forward_chunk_profiles: List[Dict[str, Any]] = []
        self._last_batched_forward_profile: Optional[Dict[str, Any]] = None

    def batched_forward(self, per_env_prepared: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        """跨 env 批量推理。"""
        profile_enabled = self._profile_enabled
        total_start = time.perf_counter() if profile_enabled else 0.0

        unpack_start = time.perf_counter() if profile_enabled else 0.0
        decoded_prepared = self._decode_prepared_batch(per_env_prepared)
        unpack_ms = (time.perf_counter() - unpack_start) * 1000.0 if profile_enabled else 0.0
        try:
            num_envs = len(decoded_prepared)
            results: List[Optional[Dict[str, Any]]] = [None] * num_envs

            collect_start = time.perf_counter() if profile_enabled else 0.0
            flat_jobs = self._collect_flat_jobs(decoded_prepared, results)
            collect_ms = (time.perf_counter() - collect_start) * 1000.0 if profile_enabled else 0.0

            if not flat_jobs:
                self._fill_empty_ok_env_results(decoded_prepared, results)
                pack_start = time.perf_counter() if profile_enabled else 0.0
                packed_outputs = self._pack_outputs(results)
                pack_ms = (time.perf_counter() - pack_start) * 1000.0 if profile_enabled else 0.0
                forward_detail_ms = _aggregate_forward_chunk_profiles([])
                self._last_batched_forward_profile = {
                    "num_envs": num_envs,
                    "flat_job_count": 0,
                    "chunk_count": 0,
                    "stage_ms": {
                        "unpack": unpack_ms,
                        "collect": collect_ms,
                        "build_chunks": 0.0,
                        "forward": 0.0,
                        "scatter": 0.0,
                        "pack": pack_ms,
                        "total": (time.perf_counter() - total_start) * 1000.0 if profile_enabled else 0.0,
                    },
                    "forward_detail_ms": forward_detail_ms,
                }
                self._maybe_log_profile(
                    num_envs=num_envs,
                    flat_jobs=flat_jobs,
                    chunks=[],
                    stage_ms=self._last_batched_forward_profile["stage_ms"],
                    forward_detail_ms=forward_detail_ms,
                )
                return packed_outputs

            build_chunks_start = time.perf_counter() if profile_enabled else 0.0
            chunks = self._build_chunks(flat_jobs)
            build_chunks_ms = (time.perf_counter() - build_chunks_start) * 1000.0 if profile_enabled else 0.0

            forward_start = time.perf_counter() if profile_enabled else 0.0
            all_per_focal = self._run_forward_chunks(chunks)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0 if profile_enabled else 0.0

            scatter_start = time.perf_counter() if profile_enabled else 0.0
            per_env_outputs = self._scatter_chunk_results(all_per_focal, decoded_prepared, results)
            scatter_ms = (time.perf_counter() - scatter_start) * 1000.0 if profile_enabled else 0.0

            pack_start = time.perf_counter() if profile_enabled else 0.0
            packed_outputs = self._pack_outputs(per_env_outputs)
            pack_ms = (time.perf_counter() - pack_start) * 1000.0 if profile_enabled else 0.0

            forward_detail_ms = _aggregate_forward_chunk_profiles(
                getattr(self, "_last_forward_chunk_profiles", []),
            )
            self._last_batched_forward_profile = {
                "num_envs": num_envs,
                "flat_job_count": len(flat_jobs),
                "chunk_count": len(chunks),
                "stage_ms": {
                    "unpack": unpack_ms,
                    "collect": collect_ms,
                    "build_chunks": build_chunks_ms,
                    "forward": forward_ms,
                    "scatter": scatter_ms,
                    "pack": pack_ms,
                    "total": (time.perf_counter() - total_start) * 1000.0 if profile_enabled else 0.0,
                },
                "forward_detail_ms": forward_detail_ms,
            }
            self._maybe_log_profile(
                num_envs=num_envs,
                flat_jobs=flat_jobs,
                chunks=chunks,
                stage_ms=self._last_batched_forward_profile["stage_ms"],
                forward_detail_ms=forward_detail_ms,
            )
            return packed_outputs
        finally:
            for prepared in decoded_prepared:
                release_prepared_payload(prepared)

    @staticmethod
    def _read_env_flag(name: str, default: str = "0") -> bool:
        value = str(_os.getenv(name, default)).strip().lower()
        return value in {"1", "true", "yes", "on"}

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

    def _run_forward_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        all_per_focal: List[Dict[str, Any]] = []
        chunk_profiles: List[Dict[str, Any]] = []
        for chunk in chunks:
            all_per_focal.extend(self._forward_chunk_batched(chunk))
            chunk_profile = getattr(self, "_last_forward_chunk_profile", None)
            if self._profile_enabled and chunk_profile is not None:
                chunk_profiles.append(
                    {
                        "chunk_jobs": int(chunk_profile.get("chunk_jobs", len(chunk))),
                        "stage_ms": dict(chunk_profile.get("stage_ms", {})),
                        "detail_ms": dict(chunk_profile.get("detail_ms", {})),
                    }
                )
        self._last_forward_chunk_profiles = chunk_profiles
        return all_per_focal

    def _maybe_log_profile(
        self,
        num_envs: int,
        flat_jobs: List[Dict[str, Any]],
        chunks: List[List[Dict[str, Any]]],
        stage_ms: Dict[str, float],
        forward_detail_ms: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._profile_enabled:
            return

        self._profile_counter += 1
        if self._profile_counter % self._profile_every != 0:
            return

        chunk_sizes = [len(chunk) for chunk in chunks]
        chunk_tokens = [sum(self._estimate_job_tokens(job) for job in chunk) for chunk in chunks]

        def _summary(values: List[int]) -> Tuple[int, float, int]:
            if not values:
                return 0, 0.0, 0
            return min(values), float(sum(values)) / len(values), max(values)

        cs_min, cs_avg, cs_max = _summary(chunk_sizes)
        ct_min, ct_avg, ct_max = _summary(chunk_tokens)
        collate_detail = (forward_detail_ms or {}).get("detail_ms", {}).get("collate", {})
        print(
            (
                "[ExternalTeacher][Profile] call=%d envs=%d flat_jobs=%d chunks=%d "
                "chunk_jobs[min/avg/max]=%d/%.2f/%d chunk_tokens[min/avg/max]=%d/%.1f/%d "
                "ms(unpack=%.2f collect=%.2f build=%.2f forward=%.2f scatter=%.2f pack=%.2f total=%.2f) "
                "forward_detail(ms collate=%.2f model_rtg=%.2f rtg_decode=%.2f rng=%.2f model_action=%.2f action_decode=%.2f) "
                "collate_detail(ms fill=%.2f from_numpy=%.2f to_device=%.2f token_index=%.2f)"
            )
            % (
                self._profile_counter,
                num_envs,
                len(flat_jobs),
                len(chunks),
                cs_min,
                cs_avg,
                cs_max,
                ct_min,
                ct_avg,
                ct_max,
                stage_ms.get("unpack", 0.0),
                stage_ms.get("collect", 0.0),
                stage_ms.get("build_chunks", 0.0),
                stage_ms.get("forward", 0.0),
                stage_ms.get("scatter", 0.0),
                stage_ms.get("pack", 0.0),
                stage_ms.get("total", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("collate", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("model_rtg", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("rtg_decode", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("rng_reserve", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("model_action", 0.0),
                (forward_detail_ms or {}).get("stage_ms", {}).get("action_decode", 0.0),
                collate_detail.get("fill_buffers", 0.0),
                collate_detail.get("from_numpy", 0.0),
                collate_detail.get("to_device", 0.0),
                collate_detail.get("token_index_to_device", 0.0),
            )
        )

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
            "next_worker_rng_state": get_next_worker_rng_state(
                teacher=self,
                env_idx=env_idx,
                step_t=prepared["step_t"],
                fallback_rng_state=prepared.get("worker_rng_state"),
            ),
        }

    def _collect_flat_jobs(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        return collect_flat_jobs(
            per_env_prepared=per_env_prepared,
            results=results,
            build_empty_env_result=self._build_empty_env_result,
        )

    def _fill_empty_ok_env_results(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> None:
        fill_empty_ok_env_results(
            per_env_prepared=per_env_prepared,
            results=results,
            build_empty_env_result=self._build_empty_env_result,
        )

    def _scatter_chunk_results(
        self,
        all_per_focal: List[Dict[str, Any]],
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> List[Optional[Dict[str, Any]]]:
        env_accum: Dict[int, Dict[str, Any]] = {}
        for per_focal in all_per_focal:
            env_idx = int(per_focal["env_idx"])
            if env_idx not in env_accum:
                env_accum[env_idx] = self._build_empty_env_result(
                    prepared=per_focal["prepared"],
                    env_idx=env_idx,
                    status="ok",
                )

            env_accum_item = env_accum[env_idx]
            env_accum_item["action_results"].update(per_focal["action_results"])
            env_accum_item["rtg_results"].update(per_focal["rtg_results"])
            env_accum_item["processed_rtg_veh_ids"].extend(per_focal["processed_rtg_veh_ids"])

        for env_idx, env_output in env_accum.items():
            results[env_idx] = env_output

        self._fill_empty_ok_env_results(per_env_prepared, results)
        self._attach_next_worker_rng_states(per_env_prepared, results)
        return results

    def _attach_next_worker_rng_states(
        self,
        per_env_prepared: List[Optional[Dict[str, Any]]],
        results: List[Optional[Dict[str, Any]]],
    ) -> None:
        for env_idx, prepared in enumerate(per_env_prepared):
            if prepared is None:
                continue
            result = results[env_idx]
            if result is None:
                continue
            result["next_worker_rng_state"] = get_next_worker_rng_state(
                teacher=self,
                env_idx=env_idx,
                step_t=prepared["step_t"],
                fallback_rng_state=prepared.get("worker_rng_state"),
            )

    def _estimate_job_tokens(self, job: Dict[str, Any]) -> int:
        return estimate_job_tokens(job)

    def _build_chunks(self, flat_jobs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return build_chunks(
            flat_jobs=flat_jobs,
            micro_batch=self.micro_batch,
        )

    def _collate_chunk_with_padding(self, chunk: List[Dict[str, Any]]):
        return collate_chunk_with_padding(
            chunk=chunk,
            device=self.device,
            collate_numpy_buffers=self._collate_numpy_buffers,
            profile_enabled=self._profile_enabled,
        )

    @torch.no_grad()
    def _decode_rtg_stage_batched(self, batched_data, batch_meta, rtg_cache):
        return decode_rtg_stage_batched(
            teacher=self,
            batched_data=batched_data,
            batch_meta=batch_meta,
            rtg_cache=rtg_cache,
        )

    @torch.no_grad()
    def _decode_action_stage_batched(
        self,
        batched_data,
        batch_meta,
        reserved_rng_states_by_job=None,
    ):
        return decode_action_stage_batched(
            teacher=self,
            batched_data=batched_data,
            batch_meta=batch_meta,
            reserved_rng_states_by_job=reserved_rng_states_by_job,
        )

    def _forward_chunk_batched(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return forward_chunk_batched(self, chunk)
