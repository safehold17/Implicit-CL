"""
adapters/opponent_vehicle package initialization.
Sets up the environment for using ctrl-sim.
"""
import sys
from pathlib import Path

# Ensure ctrl-sim is in python path before importing submodules
CTRL_SIM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "ctrl-sim"
if not CTRL_SIM_ROOT.exists():
    raise FileNotFoundError(
        f"ctrl-sim submodule not found at {CTRL_SIM_ROOT}. "
        "Please run `git submodule update --init --recursive`."
    )

ctrl_sim_root_str = str(CTRL_SIM_ROOT)
if ctrl_sim_root_str not in sys.path:
    sys.path.insert(0, ctrl_sim_root_str)

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
