"""DCD project configuration file.

This file provides a configuration interface compatible with ctrl-sim.
"""
from omegaconf import OmegaConf

CONFIG_PATH = "/home/chen/workspace/dcd-ctrlsim/cfgs"


def get_scenario_dict(hydra_cfg):
    """
    Convert the `scenario` key in the Hydra config to a plain dictionary.

    This function provides compatibility with ctrl-sim, so ctrl-sim tools
    (e.g. utils.sim.get_sim) can work correctly.

    Args:
        hydra_cfg: Hydra/OmegaConf configuration object
    
    Returns:
        dict: scenario configuration dictionary
    """
    # Check whether the config contains the nocturne.scenario path
    if 'nocturne' in hydra_cfg and 'scenario' in hydra_cfg['nocturne']:
        scenario_cfg = hydra_cfg['nocturne']['scenario']
        
        # If it is already a dictionary, return it directly
        if isinstance(scenario_cfg, dict):
            return scenario_cfg
        
        # Otherwise convert the OmegaConf object to a dictionary
        return OmegaConf.to_container(scenario_cfg, resolve=True)
    
    # Return the default scenario configuration
    return get_default_scenario_dict()


def get_default_scenario_dict():
    """
    Construct the default scenario dictionary (without Hydra decorators).

    Returns:
        dict: default scenario configuration
    """
    return {
        'start_time': 0,
        'allow_non_vehicles': True,
        'max_visible_objects': 128,
        'max_visible_road_points': 500,
        'max_visible_stop_signs': 16,
        'max_visible_traffic_lights': 16,
    }
