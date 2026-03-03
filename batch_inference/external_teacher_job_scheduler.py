"""ExternalTeacher 的 job 收集、分桶与切块逻辑。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def _assert_required_keys(payload: Dict[str, Any], required: Tuple[str, ...], payload_name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{payload_name} missing required keys: {missing}")


def collect_flat_jobs(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result: Callable[[Dict[str, Any], int, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flat_jobs: List[Dict[str, Any]] = []
    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None:
            continue

        _assert_required_keys(prepared, ("status", "step_t", "token_index", "dead_ids"), f"prepared env_idx={env_idx}")
        status = prepared["status"]
        if status == "skip":
            results[env_idx] = build_empty_env_result(prepared, env_idx=env_idx, status="skip")
            continue
        if status != "ok":
            raise ValueError(f"prepared env_idx={env_idx} has invalid status={status!r}")

        _assert_required_keys(prepared, ("focal_batches",), f"prepared env_idx={env_idx}")
        for focal_batch in prepared["focal_batches"]:
            _assert_required_keys(focal_batch, ("focal_id", "motion_data_np"), "prepared focal_batch")
            flat_jobs.append(
                {
                    "env_idx": env_idx,
                    "prepared": prepared,
                    "focal_batch": focal_batch,
                }
            )

    flat_jobs.sort(key=lambda job: (int(job["env_idx"]), int(job["focal_batch"]["focal_id"])))
    return flat_jobs


def fill_empty_ok_env_results(
    per_env_prepared: List[Optional[Dict[str, Any]]],
    results: List[Optional[Dict[str, Any]]],
    build_empty_env_result: Callable[[Dict[str, Any], int, str], Dict[str, Any]],
) -> None:
    for env_idx, prepared in enumerate(per_env_prepared):
        if prepared is None or prepared["status"] != "ok":
            continue
        if results[env_idx] is None:
            results[env_idx] = build_empty_env_result(prepared, env_idx=env_idx, status="ok")


def job_shape_key(job: Dict[str, Any]) -> Tuple[int, int]:
    agent_states = job["focal_batch"]["motion_data_np"]["agent_states"]
    return int(agent_states.shape[1]), int(agent_states.shape[0])


def estimate_job_tokens(job: Dict[str, Any]) -> int:
    agent_states = job["focal_batch"]["motion_data_np"]["agent_states"]
    seq_len = int(agent_states.shape[1])
    max_num_agents = int(agent_states.shape[0])
    return seq_len * max_num_agents * 3


def build_chunks(
    flat_jobs: List[Dict[str, Any]],
    max_chunk_tokens: int,
    micro_batch: Optional[int],
) -> List[List[Dict[str, Any]]]:
    if not flat_jobs:
        return []

    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for job in flat_jobs:
        shape_key = job_shape_key(job)
        buckets.setdefault(shape_key, []).append(job)

    max_chunk_jobs = micro_batch if micro_batch and micro_batch > 0 else None
    chunks: List[List[Dict[str, Any]]] = []
    for shape_key in sorted(buckets.keys()):
        current_chunk: List[Dict[str, Any]] = []
        current_tokens = 0
        for job in buckets[shape_key]:
            job_tokens = estimate_job_tokens(job)
            exceed_token_budget = (
                bool(current_chunk)
                and max_chunk_tokens > 0
                and current_tokens + job_tokens > max_chunk_tokens
            )
            exceed_job_budget = max_chunk_jobs is not None and len(current_chunk) >= max_chunk_jobs
            if exceed_token_budget or exceed_job_budget:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(job)
            current_tokens += job_tokens

        if current_chunk:
            chunks.append(current_chunk)

    return chunks
