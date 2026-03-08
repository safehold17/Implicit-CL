"""
adapters/opponent_vehicle package initialization.
Sets up the environment for using ctrl-sim.
"""

from adapters.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from .opponent_adapter import CtrlSimOpponentAdapter, TiltConfig
from adapters.data_bridge import DataBridge, ScenarioDataLoader
from adapters.config_loader import (
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
