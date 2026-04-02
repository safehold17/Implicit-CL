"""Scenario-pool helpers for the Nocturne-CtrlSim environment."""

from __future__ import annotations

from typing import Any
import warnings


def add_scenario(env: Any, scenario_id: str) -> bool:
    """Add a scenario to the dynamic pool when capacity permits."""
    if not env.dynamic_scenario_pool:
        return False

    if scenario_id in env.scenario_id_to_index:
        return False

    if len(env.scenario_ids) >= env.max_scenario_pool_size:
        old_id = env.scenario_ids.pop(0)
        old_idx = env.scenario_id_to_index.pop(old_id)
        del env.index_to_scenario_id[old_idx]

    new_idx = len(env.scenario_ids)
    env.scenario_ids.append(scenario_id)
    env.scenario_id_to_index[scenario_id] = new_idx
    env.index_to_scenario_id[new_idx] = scenario_id
    env._scenario_pool_dirty = True
    return True


def get_scenario_pool_size(env: Any) -> int:
    """Return the current number of scenarios in the pool."""
    return len(env.scenario_ids)


def rebuild_index_mappings(env: Any) -> None:
    """Rebuild scenario index mappings from the current scenario list."""
    env.scenario_id_to_index = {
        scenario_id: index for index, scenario_id in enumerate(env.scenario_ids)
    }
    env.index_to_scenario_id = {
        index: scenario_id for index, scenario_id in enumerate(env.scenario_ids)
    }
    env._scenario_pool_dirty = False


def resolve_scenario_id(env: Any, scenario_idx: int) -> str:
    """Resolve a scenario index to an ID with a first-scenario fallback."""
    if scenario_idx not in env.index_to_scenario_id:
        warnings.warn(
            f"Scenario index {scenario_idx} not found in mapping. "
            f"Falling back to first scenario: {env.scenario_ids[0]}"
        )
    return env.index_to_scenario_id.get(scenario_idx, env.scenario_ids[0])
