# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .obs_wrappers import VecPreprocessImageWrapper, AdversarialObservationWrapper
from .parallel_wrappers import ParallelAdversarialVecEnv
from .time_limit import TimeLimit
from .vec_monitor import VecMonitor
from .vec_normalize import VecNormalize
from .vec_frame_stack import VecFrameStack

# multigrid_wrappers 依赖 gym_minigrid，设为可选导入
try:
    from .multigrid_wrappers import *
    HAS_MULTIGRID = True
except ImportError:
    HAS_MULTIGRID = False
    # 提供占位符，避免导入错误
    MultiGridFullyObsWrapper = None

from .car_racing_wrappers import CarRacingWrapper
