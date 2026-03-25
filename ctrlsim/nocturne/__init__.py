# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Import file for Nocturne objects."""
import sys
from pathlib import Path

try:
    from .nocturne_cpp import (Action, CollisionType, ObjectType, Object, RoadLine,
                                StopSign, RoadType, Scenario, Simulation, Vector2D, Vehicle,
                                Pedestrian, Cyclist)
except ModuleNotFoundError as exc:
    if exc.name not in {"nocturne.nocturne_cpp", f"{__name__}.nocturne_cpp"}:
        raise

    opt_ctrlsim_root = Path("/opt/ctrl-sim")
    opt_ctrlsim_root_str = str(opt_ctrlsim_root)
    if opt_ctrlsim_root.exists() and opt_ctrlsim_root_str not in sys.path:
        sys.path.append(opt_ctrlsim_root_str)

    from nocturne_cpp import (Action, CollisionType, ObjectType, Object, RoadLine,
                              StopSign, RoadType, Scenario, Simulation, Vector2D, Vehicle,
                              Pedestrian, Cyclist)

__all__ = [
    "Action",
    "CollisionType",
    "ObjectType",
    "Object",
    "RoadLine",
    "StopSign",
    "RoadType",
    "Scenario",
    "Simulation",
    "Vector2D",
    "Vehicle",
    "Pedestrian",
    "Cyclist",
    "envs",
]
import os
