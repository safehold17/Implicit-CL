from __future__ import annotations

import time
from typing import Any, Dict, List


def elapsed_ms(start_time: float, profile_enabled: bool) -> float:
    if not profile_enabled:
        return 0.0
    return (time.perf_counter() - start_time) * 1000.0


def chunk_predict_rtgs_mode(chunk: List[Dict[str, Any]]) -> bool:
    predict_rtgs_flags = {
        bool(job["focal_batch"].get("predict_rtgs", True))
        for job in chunk
    }
    if not predict_rtgs_flags:
        return True
    if len(predict_rtgs_flags) != 1:
        raise ValueError("Mixed predict_rtgs modes in the same chunk are not supported.")
    return predict_rtgs_flags.pop()

