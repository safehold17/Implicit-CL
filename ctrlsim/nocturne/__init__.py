# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Import file for Nocturne objects."""
from pathlib import Path

from ._import_utils import load_nocturne_cpp_from_root

try:
    from .nocturne_cpp import (Action, CollisionType, ObjectType, Object, RoadLine,
                                StopSign, RoadType, Scenario, Simulation, Vector2D, Vehicle,
                                Pedestrian, Cyclist)
except ModuleNotFoundError as exc:
    if exc.name not in {"nocturne.nocturne_cpp", f"{__name__}.nocturne_cpp"}:
        raise

    opt_ctrlsim_root = Path("/opt/ctrl-sim")
    nocturne_cpp = load_nocturne_cpp_from_root(opt_ctrlsim_root)
    Action = nocturne_cpp.Action
    CollisionType = nocturne_cpp.CollisionType
    ObjectType = nocturne_cpp.ObjectType
    Object = nocturne_cpp.Object
    RoadLine = nocturne_cpp.RoadLine
    StopSign = nocturne_cpp.StopSign
    RoadType = nocturne_cpp.RoadType
    Scenario = nocturne_cpp.Scenario
    Simulation = nocturne_cpp.Simulation
    Vector2D = nocturne_cpp.Vector2D
    Vehicle = nocturne_cpp.Vehicle
    Pedestrian = nocturne_cpp.Pedestrian
    Cyclist = nocturne_cpp.Cyclist

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
