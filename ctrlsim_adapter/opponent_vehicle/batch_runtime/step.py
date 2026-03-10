from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from batch_inference.ipc_codec import pack_prepared

from .focal_batch import build_focal_batches
from .rng import resolve_sampling_rng_state
from .shared import (
    build_sparse_repeat_actions,
    build_warmup_gt_actions,
    clear_pending_sparse_actions,
    set_pending_sparse_actions,
)


def prepare_step(
    adapter: Any,
    t: int,
    vehicles: List[Any],
    worker_rng_state: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if adapter._policy is None or len(vehicles) == 0:
        clear_pending_sparse_actions(adapter)
        return None

    adapter._last_vehicles = vehicles
    adapter._last_vehicle_by_id = {veh.getID(): veh for veh in vehicles}

    adapter._vehicle_data_dict = adapter._update_vehicle_data_dict(
        t,
        vehicles,
        adapter._vehicle_data_dict,
    )
    adapter.update_policy_state(t)

    if t < adapter.history_steps - 1:
        warmup_actions = build_warmup_gt_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=warmup_actions)

    is_sparse_step = adapter.sparse_inference.is_sparse_step(
        t=t,
        history_steps=adapter.history_steps,
    )
    if is_sparse_step and adapter.sparse_inference_action_repeat:
        actions = build_sparse_repeat_actions(adapter, t)
        set_pending_sparse_actions(adapter, step_t=t, actions=actions)
        return None

    clear_pending_sparse_actions(adapter)
    focal_batches, dead_ids = build_focal_batches(adapter, t)
    token_index = t if t < adapter._policy.cfg_rl_waymo.train_context_length else -1
    sampling_rng_state = resolve_sampling_rng_state(adapter, worker_rng_state)
    if not focal_batches and not dead_ids:
        return pack_prepared(
            {
                "status": "skip",
                "step_t": t,
                "token_index": token_index,
                "dead_ids": [],
                "worker_rng_state": sampling_rng_state,
            }
        )

    tilt_by_veh_id: Dict[int, tuple[int, int, int]] = (
        dict(adapter.per_vehicle_tilting) if adapter.per_vehicle_tilting else {}
    )
    prepared_dict = {
        "status": "ok",
        "step_t": t,
        "token_index": token_index,
        "dead_ids": dead_ids,
        "worker_rng_state": sampling_rng_state,
        "sampling": {
            "action_temperature": adapter.action_temperature,
            "nucleus_sampling": adapter.nucleus_sampling,
            "nucleus_threshold": adapter.nucleus_threshold,
        },
        "default_tilt": (
            adapter.current_tilt.goal_tilt,
            adapter.current_tilt.veh_veh_tilt,
            adapter.current_tilt.veh_edge_tilt,
        ),
        "tilt_by_veh_id": tilt_by_veh_id,
        "veh_id_to_idx": dict(adapter._policy.veh_id_to_idx),
        "focal_batches": focal_batches,
    }
    return pack_prepared(prepared_dict)
