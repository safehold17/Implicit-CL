"""Shared warning suppression hooks for dcd-ctrlsim entrypoints."""

from __future__ import annotations

import os
import sys
import types
import warnings
from pathlib import Path


_GYM_MINIGRID_WARNING_RE = r"The package name gym_minigrid has been deprecated.*"
_PYGAME_PKGDATA_WARNING_RE = r"pkg_resources is deprecated as an API\..*"
_original_filterwarnings = warnings.filterwarnings
_INSTALLED = False


def _install_warning_filters() -> None:
    _original_filterwarnings(
        "ignore",
        message=_GYM_MINIGRID_WARNING_RE,
        category=DeprecationWarning,
        append=False,
    )
    _original_filterwarnings(
        "ignore",
        message=_PYGAME_PKGDATA_WARNING_RE,
        category=UserWarning,
        module=r"pygame\.pkgdata",
        append=False,
    )


def _patched_filterwarnings(*args, **kwargs):
    result = _original_filterwarnings(*args, **kwargs)
    _install_warning_filters()
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    warnings.filterwarnings = _patched_filterwarnings
    _install_warning_filters()

    # Gym prints the "Gym has been unmaintained..." banner during import by
    # loading ``gym_notices.notices`` and looking up a version-keyed message.
    # Replacing the notices table with an empty one suppresses that banner.
    gym_notices_pkg = sys.modules.get("gym_notices")
    if gym_notices_pkg is None:
        gym_notices_pkg = types.ModuleType("gym_notices")
        gym_notices_pkg.__path__ = []
        sys.modules["gym_notices"] = gym_notices_pkg

    gym_notices_module = types.ModuleType("gym_notices.notices")
    gym_notices_module.notices = {}
    sys.modules["gym_notices.notices"] = gym_notices_module
    _INSTALLED = True


def configure_subprocess_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    startup_dir = project_root / "util"

    current = os.environ.get("PYTHONPATH", "")
    path_parts = [p for p in current.split(os.pathsep) if p]

    prepend_parts = [str(startup_dir), str(project_root)]
    new_parts = []
    for part in prepend_parts:
        if part not in path_parts:
            new_parts.append(part)
    new_parts.extend(path_parts)

    os.environ["PYTHONPATH"] = os.pathsep.join(new_parts)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")


install()
