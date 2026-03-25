"""
负责定位仓库中的 ctrl-sim 源码目录，并在运行时注入 `sys.path`。
该模块为 adapter 与 batch inference 模块导入原始 ctrl-sim 代码提供统一入口。
Locates the vendored ctrl-sim source tree and injects the path into `sys.path` at runtime.
Provides a single import-path bootstrap point for adapter and batch-inference modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_CTRLSIM_ROOT = _PROJECT_ROOT / "ctrlsim"
_DEFAULT_INSTALLED_CTRLSIM_ROOT = Path("/opt/ctrl-sim")


def _resolve_ctrlsim_root() -> Path:
    env_root = os.environ.get("CTRLSIM_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"CTRLSIM_ROOT is set but path does not exist: {candidate}."
        )

    if _LOCAL_CTRLSIM_ROOT.exists():
        return _LOCAL_CTRLSIM_ROOT

    if _DEFAULT_INSTALLED_CTRLSIM_ROOT.exists():
        return _DEFAULT_INSTALLED_CTRLSIM_ROOT

    raise FileNotFoundError(
        "ctrl-sim source tree not found. Checked "
        f"CTRLSIM_ROOT, {_DEFAULT_INSTALLED_CTRLSIM_ROOT}, and {_LOCAL_CTRLSIM_ROOT}."
    )


def get_ctrlsim_root() -> Path:
    return _resolve_ctrlsim_root()


def ctrlsim_path() -> Path:
    ctrlsim_root = get_ctrlsim_root()

    ctrlsim_root_str = str(ctrlsim_root)
    if ctrlsim_root_str not in sys.path:
        sys.path.insert(0, ctrlsim_root_str)

    return ctrlsim_root
