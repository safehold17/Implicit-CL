"""
Simulation info helper functions for Nocturne CtrlSim adversarial env.
"""

from typing import Any, Dict, Optional

import numpy as np

from ctrlsim_evaluation_metrics import (
    compute_ctrlsim_ego_metrics_from_adapter,
)
from ..student.student_reward import get_student_component_applied_return
from ..utils.common import clamp01, merge_episode_progress


EGO_TURN_HEADING_DELTA_THRESHOLD = 0.03


def compute_current_progress(env) -> float:
    """Calculate current normalized progress from ego position to goal."""
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return 0.0

    if env._ego_goal_dist_normalizer <= 0:
        return 0.0

    # Use the step-scoped dist-to-goal already computed by _update_step_ego_cache
    # to avoid an extra C++ getPosition() call.  Fall back to live computation
    # when the cache is not yet available (e.g. at reset time).
    dist_to_goal = getattr(env, "_step_dist_to_goal", None)
    if dist_to_goal is None:
        ego_pos = env.ego_vehicle.getPosition()
        dist_to_goal = np.linalg.norm(
            env._ego_goal_dict['pos'] - np.array([ego_pos.x, ego_pos.y])
        )

    progress = 1.0 - dist_to_goal / env._ego_goal_dist_normalizer
    return clamp01(progress)


def update_episode_progress(
    previous_progress: float,
    current_progress: float,
    position_reached: bool,
) -> float:
    """
    Merge current progress into episode progress.

    Once position threshold has been reached, lock episode progress to 1.0.
    Otherwise, keep the best progress reached in this episode.
    """
    return merge_episode_progress(
        previous_progress=previous_progress,
        current_progress=current_progress,
        position_reached=position_reached,
    )


def check_done(env) -> bool:
    """Check if episode is done."""
    # Max steps (timeout)
    if env.current_step >= env.max_episode_steps:
        return True

    # Position reached takes precedence over safety-triggered early termination.
    # This keeps progress and termination semantics aligned when the ego reaches
    # the goal on the same step that a collision/offroad flag is also raised.
    if bool(getattr(env, "_position_reached", False)):
        return True

    if bool(getattr(env, "early_termination", False)) and (
        env._collision_occurred or env._offroad_occurred
    ):
        return True

    done_on_position_reached_only = bool(
        getattr(env, 'done_on_position_reached_only', True)
    )
    # Success condition is configurable:
    # - True: only require position reached
    # - False: keep legacy goal_reached condition
    if done_on_position_reached_only:
        if env._position_reached:
            return True
    elif env._goal_reached:
        return True

    return False


def _build_policy_reweighting_info(env) -> Dict[str, float]:
    """Build step-level policy reweighting info for logging."""
    scale = 1.0
    ego_reweight_tilt = (0.0, 0.0, 0.0)

    if (
        bool(getattr(env, 'use_policy_reweighting', False))
        and str(getattr(env, 'opponent_runtime_mode', 'normal')) == 'normal'
        and len(getattr(env, 'opponent_vehicle_ids', ())) > 0
    ):
        opponent = getattr(env, 'opponent', None)
        if opponent is not None:
            raw_scale = float(getattr(opponent, '_ego_action_scale', 1.0))
            if np.isfinite(raw_scale):
                scale = raw_scale

        raw_tilt = tuple(
            float(v)
            for v in getattr(env, 'current_ego_reweight_tilt', (0.0, 0.0, 0.0))
        )
        if len(raw_tilt) == 3 and all(np.isfinite(v) for v in raw_tilt):
            ego_reweight_tilt = raw_tilt

    return {
        'ego_action_scale': scale,
        'ego_goal_tilt': ego_reweight_tilt[0],
        'ego_veh_veh_tilt': ego_reweight_tilt[1],
        'ego_veh_edge_tilt': ego_reweight_tilt[2],
    }


def _angle_diff(a: float, b: float) -> float:
    """Return the wrapped difference between two headings in radians."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _get_ego_gt_traj(env) -> Optional[np.ndarray]:
    """Return the ego GT trajectory array when it is available."""
    ego_vehicle = getattr(env, "ego_vehicle", None)
    if ego_vehicle is None:
        return None

    gt_traj_cache = getattr(env, "_gt_traj_cache", None)
    gt_data_dict = getattr(env, "_gt_data_dict", {})
    if gt_traj_cache is None and not gt_data_dict:
        return None
    if not hasattr(ego_vehicle, "getID"):
        return None

    ego_id = int(ego_vehicle.getID())
    if gt_traj_cache is not None and ego_id in gt_traj_cache:
        return np.asarray(gt_traj_cache[ego_id])

    data = gt_data_dict.get(ego_id)
    if not isinstance(data, dict) or "traj" not in data:
        return None

    gt_traj = np.asarray(data["traj"])
    if gt_traj_cache is not None:
        gt_traj_cache[ego_id] = gt_traj
    return gt_traj


def _build_ego_heading_error_info(env) -> Dict[str, float]:
    """Build ego heading-error diagnostics against the GT trajectory."""
    gt_traj = _get_ego_gt_traj(env)
    if gt_traj is None:
        return {}

    current_step = int(getattr(env, "current_step", -1))
    if current_step < 0 or current_step >= len(gt_traj):
        return {}

    ego_heading = getattr(env, "_step_ego_heading", None)
    if ego_heading is None:
        ego_vehicle = getattr(env, "ego_vehicle", None)
        if ego_vehicle is None:
            return {}
        ego_heading = float(ego_vehicle.getHeading())

    gt_heading = float(gt_traj[current_step][2])
    if not np.isfinite(ego_heading) or not np.isfinite(gt_heading):
        return {}

    heading_error = abs(_angle_diff(float(ego_heading), gt_heading))
    info = {"ego_heading_error_to_gt": heading_error}

    next_step = current_step + 1
    if next_step < len(gt_traj):
        next_gt_heading = float(gt_traj[next_step][2])
        if np.isfinite(next_gt_heading):
            gt_turn_delta = abs(_angle_diff(next_gt_heading, gt_heading))
            if gt_turn_delta > EGO_TURN_HEADING_DELTA_THRESHOLD:
                info["ego_turn_heading_error_to_gt"] = heading_error
            else:
                info["ego_non_turn_heading_error_to_gt"] = heading_error

    return info


def get_info(
    env,
    done: Optional[bool] = None,
    current_progress: Optional[float] = None,
) -> Dict[str, Any]:
    """Return additional information.

    ``done`` and ``current_progress`` may be passed in when the caller has
    already computed them, avoiding a second redundant evaluation.
    """
    if done is None:
        done = check_done(env)

    if current_progress is None:
        current_progress = compute_current_progress(env)
    progress = max(current_progress, float(getattr(env, '_episode_progress', 0.0)))

    if done:
        # Cache an atomic completed-episode snapshot for get_complexity_info()
        # so downstream logging never mixes new scenario_id with old stats.
        env._last_completed_complexity_info = build_complexity_snapshot(
            env,
            build_episode_stats(env),
        )

    info = {
        'step': env.current_step,
        'episode_reward': env.episode_reward,
        # Diagnostic information (参考 ctrl-sim metrics)
        # Diagnostic information (see ctrl-sim metrics).
        'collision': env._collision_occurred,
        'goal_reached': env._goal_reached,
        'position_reached': env._position_reached,
        'offroad': env._offroad_occurred,
        'progress': progress,
    }
    info.update(_build_ego_heading_error_info(env))

    # Always add complexity info (real-time data)
    info.update(get_complexity_info(env))
    info.update(_build_policy_reweighting_info(env))
    if bool(getattr(env, 'use_enhanced_regret', False)):
        student_component_applied_return = getattr(
            env,
            '_student_component_applied_return_before_inference_step',
            None,
        )
        if student_component_applied_return is None:
            student_component_applied_return = get_student_component_applied_return(env)
        info['student_component_applied_return'] = np.asarray(
            student_component_applied_return,
            dtype=np.float32,
        )

    # Add episode summary when episode ends
    if done:
        info['episode'] = {
            'r': env.episode_reward,
            'l': env.current_step,
        }
        ego_vehicle = getattr(env, "ego_vehicle", None)
        ego_id = ego_vehicle.getID() if ego_vehicle is not None else None
        info.update(
            compute_ctrlsim_ego_metrics_from_adapter(
                getattr(env, "opponent", None),
                ego_id,
            )
        )

    return info



def reset_metrics(env) -> None:
    """Reset metrics tracking."""
    env.episode_reward = 0.0
    env.collision_count = 0
    env.goal_reached = False


def get_complexity_info(env) -> Dict[str, Any]:
    """
    Return current level complexity information and episode statistics (for logging and analysis).

    Prioritizes returning cached data from the last completed episode to avoid
    returning zeros when called immediately after reset.
    """
    # Prioritize cached completed episode snapshot.
    # Return the cached dict directly — callers only read from it (e.g. via
    # info.update()), so no defensive copy is needed here.
    if env._last_completed_complexity_info is not None:
        return env._last_completed_complexity_info

    if env.current_level is None:
        return {}

    return build_complexity_snapshot(env, build_episode_stats(env))


def is_episode_success(
    max_progress: float,
    collision_occurred: float,
    offroad_occurred: float,
    threshold: float,
) -> bool:
    """Return whether an episode satisfies the default solvable criterion."""
    return (
        float(max_progress) > float(threshold)
        and float(collision_occurred) == 0.0
        and float(offroad_occurred) == 0.0
    )


def build_episode_stats(env) -> Dict[str, float]:
    """Build episode statistics payload shared by info and complexity APIs."""
    collision_occurred = 1.0 if env._episode_collision_occurred else 0.0
    offroad_occurred = 1.0 if env._episode_offroad_occurred else 0.0
    max_progress = float(env._episode_progress)
    success = 1.0 if is_episode_success(
        max_progress=max_progress,
        collision_occurred=collision_occurred,
        offroad_occurred=offroad_occurred,
        threshold=float(getattr(env, 'solvable_progress_threshold', 0.85)),
    ) else 0.0
    return {
        'collision_occurred': collision_occurred,
        'goal_reached_occurred': 1.0 if env._episode_goal_reached else 0.0,
        'position_reached_occurred': 1.0 if env._episode_position_reached else 0.0,
        'offroad_occurred': offroad_occurred,
        'max_progress': max_progress,
        'success': success,
        'episode_steps': env._episode_steps,
        'episode_reward': env.episode_reward,
    }


def build_level_context(env) -> Dict[str, Any]:
    """Build level metadata payload for complexity logging."""
    if env.current_level is None:
        return {}

    info = {
        'scenario_id': env.current_level.scenario_id,
        'seed': env.current_level.seed,
        'opponent_k': env.opponent_k,
        'opponent_vehicle_num': int(getattr(env, 'current_opponent_vehicle_num', 0)),
        'scenario_pool_size': len(env.scenario_ids),
    }

    if env.tilting_mode in ('global', 'none'):
        info.update(
            {
                'goal_tilt': 0 if env.tilting_mode == 'none' else env.current_level.goal_tilt,
                'veh_veh_tilt': 0 if env.tilting_mode == 'none' else env.current_level.veh_veh_tilt,
                'veh_edge_tilt': 0 if env.tilting_mode == 'none' else env.current_level.veh_edge_tilt,
            }
        )
        return info

    per = env.current_level.per_vehicle_tilting
    for i in range(env.opponent_k):
        base = 3 * i
        info[f'per_vehicle_goal_tilt_{i}'] = per[base]
        info[f'per_vehicle_veh_tilt_{i}'] = per[base + 1]
        info[f'per_vehicle_edge_tilt_{i}'] = per[base + 2]
    return info


def build_complexity_snapshot(env, episode_stats: Dict[str, float]) -> Dict[str, Any]:
    """Build atomic complexity snapshot: scenario context + episode stats."""
    snapshot = build_level_context(env)
    snapshot.update(episode_stats)
    return snapshot
