"""Helpers for loading the Nocturne extension from known locations."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType


def _iter_nocturne_cpp_candidates(root: Path):
    seen: set[Path] = set()
    for search_root in (root, root / "nocturne"):
        for suffix in (*EXTENSION_SUFFIXES, ".py"):
            candidate = search_root / f"nocturne_cpp{suffix}"
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                yield candidate


def _load_module_from_path(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("nocturne_cpp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get("nocturne_cpp")
    sys.modules["nocturne_cpp"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop("nocturne_cpp", None)
        else:
            sys.modules["nocturne_cpp"] = previous_module
        raise
    return module


def load_nocturne_cpp_from_root(root: Path) -> ModuleType:
    """Load ``nocturne_cpp`` from an explicit installation root only."""
    for candidate in _iter_nocturne_cpp_candidates(root):
        return _load_module_from_path(candidate)
    raise ModuleNotFoundError(
        f"Could not locate nocturne_cpp under explicit root: {root}"
    )
