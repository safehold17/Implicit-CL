"""Batch inference: 主进程跨 env 的 ExternalTeacher 推理引擎。"""

from .external_teacher import ExternalTeacher

__all__ = ["ExternalTeacher"]
