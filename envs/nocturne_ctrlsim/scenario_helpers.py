"""
Scenario helper functions for Nocturne CtrlSim adversarial env.
"""

import os


def rebuild_vehicle_id_cache(env) -> dict:
    """Rebuild and store vehicle-id -> vehicle object cache."""
    cache = {veh.getID(): veh for veh in env.vehicles}
    env._vehicle_by_id_cache = cache
    return cache


def load_scenario(env, scenario_id: str) -> None:
    """
    Load Nocturne scenario.

    See: third_party/ctrl-sim/utils/sim.py get_sim() function
    """
    from nocturne import Simulation
    from omegaconf import OmegaConf

    scenario_path = os.path.join(env.scenario_data_dir, f"{scenario_id}.json")

    # Nocturne Simulation only needs scenario configuration dictionary
    # See cfgs/config.py get_scenario_dict() function
    if 'scenario' in env.cfg.nocturne:
        scenario_dict = OmegaConf.to_container(
            env.cfg.nocturne.scenario, resolve=True
        )
    else:
        # Fall back to basic configuration
        scenario_dict = {
            'start_time': 0,
            'allow_non_vehicles': False,
        }

    env.sim = Simulation(scenario_path, scenario_dict)
    env.scenario = env.sim.getScenario()
    env.vehicles = list(env.scenario.vehicles())
    rebuild_vehicle_id_cache(env)

    # Set vehicle control flags (see ctrl-sim evaluator.py line 37-39)
    for veh in env.vehicles:
        veh.expert_control = False
        veh.physics_simulated = True



def get_vehicle_by_id(env, veh_id: int):
    """Get vehicle object by ID."""
    cache = getattr(env, "_vehicle_by_id_cache", None)
    if cache is None:
        cache = rebuild_vehicle_id_cache(env)

    veh = cache.get(veh_id)
    if veh is not None:
        return veh

    # Fallback refresh in case env.vehicles changed without explicit cache update.
    cache = rebuild_vehicle_id_cache(env)
    return cache.get(veh_id)



def remove_background_moving_vehicles(env) -> None:
    """
    Remove moving vehicles other than ego/opponent.
    Uses preprocessed-filtered moving vehicles and skips removal if
    preprocessed data is missing (fallback B).
    """
    if not hasattr(env, '_preproc_data') or env._preproc_data is None:
        return
    
    from tools.vehicle_selection import get_moving_vehicle_ids, get_preproc_vehicle_ids

    preproc_ids = get_preproc_vehicle_ids(env)
    if not preproc_ids:
        return

    moving_ids = get_moving_vehicle_ids(env, filter_by_preproc=True)
    if not moving_ids:
        return

    ego_id = env.ego_vehicle.getID() if env.ego_vehicle else None
    protected_ids = set(env.opponent_vehicle_ids)
    if ego_id is not None:
        protected_ids.add(ego_id)

    remove_ids = [vid for vid in moving_ids if vid not in protected_ids]
    if not remove_ids:
        return

    removed_set = set(remove_ids)
    for veh in env.vehicles:
        if veh.getID() in removed_set:
            veh.setPosition(-1000000, -1000000)
            veh.physics_simulated = False
            env.opponent.apply_action(veh, (0.0, 0.0))
            veh.speed = 0.0

    env.vehicles = [veh for veh in env.vehicles if veh.getID() not in removed_set]
    rebuild_vehicle_id_cache(env)
    env.removed_vehicle_ids = remove_ids
