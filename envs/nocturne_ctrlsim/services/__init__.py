"""Runtime services for the Nocturne CtrlSim environment."""

from .gt_helpers import (
    build_episode_gt_action_cache,
    get_goal_point_for_vehicle,
    get_gt_action,
    initialize_ego_goal_state,
    is_ego_position_reached,
)
from .level_manager import (
    build_level_from_params,
    coerce_level,
    create_level_from_params,
    decode_string_encoding,
    initialize_level_with_fallback,
    mutate_level_internal,
    sample_random_level,
    sync_level_state,
)
from .runtime import NocturneCtrlSimRuntime, split_prepared_pack_batch
from .scenario_pool import (
    add_scenario,
    get_scenario_pool_size,
    rebuild_index_mappings,
    resolve_scenario_id,
)
from .scenario_helpers import (
    get_vehicle_by_id,
    load_scenario,
    rebuild_vehicle_id_cache,
    remove_background_moving_vehicles,
)
from .simulation_info import (
    check_done,
    compute_current_progress,
    get_complexity_info,
    get_info,
    reset_metrics,
    update_episode_progress,
)

__all__ = [
    "NocturneCtrlSimRuntime",
    "add_scenario",
    "build_episode_gt_action_cache",
    "build_level_from_params",
    "check_done",
    "coerce_level",
    "compute_current_progress",
    "create_level_from_params",
    "decode_string_encoding",
    "get_complexity_info",
    "get_goal_point_for_vehicle",
    "get_gt_action",
    "get_info",
    "get_scenario_pool_size",
    "get_vehicle_by_id",
    "initialize_level_with_fallback",
    "initialize_ego_goal_state",
    "is_ego_position_reached",
    "load_scenario",
    "mutate_level_internal",
    "rebuild_vehicle_id_cache",
    "rebuild_index_mappings",
    "resolve_scenario_id",
    "remove_background_moving_vehicles",
    "reset_metrics",
    "sample_random_level",
    "split_prepared_pack_batch",
    "sync_level_state",
    "update_episode_progress",
]
