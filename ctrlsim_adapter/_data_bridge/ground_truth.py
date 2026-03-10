from __future__ import annotations

from typing import Any, Dict

from utils.sim import get_ground_truth_states


def get_ground_truth(
    bridge: Any,
    scenario_path: str,
    scenario_filename: str,
) -> Dict:
    files = [scenario_filename]
    file_id = 0
    return get_ground_truth_states(
        bridge.cfg,
        scenario_path,
        files,
        file_id,
        bridge.dt,
        bridge.steps,
    )


def get_ground_truth_from_sim(
    bridge: Any,
    sim: Any,
    scenario_filename: str,
) -> Dict:
    del scenario_filename
    from utils.data import get_agent_type_onehot

    def get_state(veh: Any):
        pos = veh.getPosition()
        heading = veh.getHeading()
        target = veh.getGoalPosition()
        speed = veh.getSpeed()
        agent_type = get_agent_type_onehot(veh.getType().value)
        existence = 1 if pos.x != -10000 else 0
        length = veh.getLength()
        veh_state = [pos.x, pos.y, heading, speed, existence, target.x, target.y, length]
        return veh_state, agent_type

    scenario = sim.getScenario()
    vehicles = scenario.vehicles()
    state_dict = {veh.getID(): {"traj": [], "type": None} for veh in vehicles}

    for veh in vehicles:
        veh.expert_control = True

    for _ in range(bridge.steps):
        for veh in vehicles:
            veh_state, veh_type = get_state(veh)
            state_dict[veh.getID()]["traj"].append(veh_state)
            state_dict[veh.getID()]["type"] = veh_type
        sim.step(bridge.dt)

    for veh in vehicles:
        veh_state, veh_type = get_state(veh)
        state_dict[veh.getID()]["traj"].append(veh_state)
        state_dict[veh.getID()]["type"] = veh_type

    sim.reset()
    return state_dict

