"""
负责批量解码阶段的轻量计时与单批次模式判定。
该模块为 forward/decode 流程提供统一的 profiling 小工具，不承载核心推理逻辑。
Provides lightweight timing helpers and single-batch mode checks for batch decoding.
Supports forward/decode profiling with small utilities rather than core inference logic.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


def elapsed_ms(start_time: float, profile_enabled: bool) -> float:
    if not profile_enabled:
        return 0.0
    return (time.perf_counter() - start_time) * 1000.0


def batch_predict_rtgs_mode(jobs: List[Dict[str, Any]]) -> bool:
    predict_rtgs_flags = {
        bool(job["focal_batch"].get("predict_rtgs", True))
        for job in jobs
    }
    if not predict_rtgs_flags:
        return True
    if len(predict_rtgs_flags) != 1:
        raise ValueError("Mixed predict_rtgs modes in the same batch are not supported.")
    return predict_rtgs_flags.pop()
