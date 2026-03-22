"""
ctrlsim_adapter/opponent_vehicle package initialization.
Sets up the environment for using ctrl-sim.
"""

from ctrlsim_adapter.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from . import opponent_inference_io
from .opponent_adapter import CtrlSimOpponentAdapter, TiltConfig
from ctrlsim_adapter.data_bridge import DataBridge, ScenarioDataLoader
from ctrlsim_adapter.config_loader import (
    load_ctrl_sim_config,
    load_ctrl_sim_config_from_yaml,
    get_default_opponent_config,
    get_default_nocturne_config,
    get_default_dataset_config,
    create_minimal_config,
    ConfigManager,
)

__all__ = [
    # Core adapter API
    'CtrlSimOpponentAdapter',
    'TiltConfig',
    'opponent_inference_io',
    
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
