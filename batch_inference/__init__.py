"""
Batch inference: 主进程跨 env 的 ExternalTeacher 推理引擎。
Batch inference: main-process cross-env ExternalTeacher inference engine.
"""

from .external_teacher import ExternalTeacher, build_external_teacher_kwargs

__all__ = ["ExternalTeacher", "build_external_teacher_kwargs"]
