# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use try-except for conditional imports to avoid forced dependencies.
try:
    from .multigrid.adversarial import *
except ImportError:
    # if gym_minigrid not installed, skip
    pass

try:
    from .nocturne_ctrlsim import *
except ImportError:
    # if nocturne not installed, skip
    pass
