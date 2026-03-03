"""
Batch inference: 主进程跨 env 批量推理引擎。

将 teacher（CtRL-Sim）GPU 模型从 env 子进程移到主进程，
实现跨 env 扁平化 focal 批的批量 forward。
"""
from .external_teacher import ExternalTeacher
from . import prepare_and_apply

__all__ = ['ExternalTeacher']
