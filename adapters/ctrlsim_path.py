from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CTRLSIM_ROOT = _PROJECT_ROOT / "ctrlsim"


def get_ctrlsim_root() -> Path:
    if not _CTRLSIM_ROOT.exists():
        raise FileNotFoundError(
            f"ctrl-sim source tree not found at {_CTRLSIM_ROOT}."
        )

    return _CTRLSIM_ROOT


def ctrlsim_path() -> Path:
    ctrlsim_root = get_ctrlsim_root()

    ctrlsim_root_str = str(ctrlsim_root)
    if ctrlsim_root_str not in sys.path:
        sys.path.insert(0, ctrlsim_root_str)

    return ctrlsim_root
