"""
adapters/ctrl_sim package initialization.
Sets up the environment for using ctrl-sim.
"""
import os
import sys

# Ensure ctrl-sim is in python path before importing submodules
CTRL_SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/ctrl-sim'))
if not os.path.exists(CTRL_SIM_ROOT):
    raise FileNotFoundError(
        f"ctrl-sim submodule not found at {CTRL_SIM_ROOT}. "
        "Please run `git submodule update --init --recursive`."
    )

if CTRL_SIM_ROOT not in sys.path:
    sys.path.insert(0, CTRL_SIM_ROOT)

from .opponent_adapter import CtrlSimOpponentAdapter, TiltConfig
from .data_bridge import DataBridge, ScenarioDataLoader
from .config_loader import (
    load_ctrl_sim_config,
    load_ctrl_sim_config_from_yaml,
    get_default_opponent_config,
    get_default_nocturne_config,
    get_default_dataset_config,
    create_minimal_config,
    ConfigManager,
)

__all__ = [
    # Core adapters
    'CtrlSimOpponentAdapter',
    'TiltConfig',
    
    # Data bridge
    'DataBridge',
    'ScenarioDataLoader',
    
    # Configuration loading
    'load_ctrl_sim_config',
    'load_ctrl_sim_config_from_yaml',
    'get_default_opponent_config',
    'get_default_nocturne_config',
    'get_default_dataset_config',
    'create_minimal_config',
    'ConfigManager',
]

# Version information
__version__ = '0.1.0'
