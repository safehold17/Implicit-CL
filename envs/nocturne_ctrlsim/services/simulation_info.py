"""
Simulation info helper functions for Nocturne CtrlSim adversarial env.
"""

from typing import Any, Dict

import numpy as np

from ..utils.common import clamp01, merge_episode_progress


def compute_current_progress(env) -> float:
    """Calculate current normalized progress from ego position to goal."""
    if env.ego_vehicle is None or env._ego_goal_dict is None:
        return 0.0

    ego_pos = env.ego_vehicle.getPosition()
    dist_to_goal = np.linalg.norm(
        env._ego_goal_dict['pos'] - np.array([ego_pos.x, ego_pos.y])
    )
    if env._ego_goal_dist_normalizer <= 0:
        return 0.0

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


def get_info(env) -> Dict[str, Any]:
    """Return additional information."""
    done = check_done(env)

    progress = compute_current_progress(env)
    progress = max(progress, float(getattr(env, '_episode_progress', 0.0)))

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

    # Always add complexity info (real-time data)
    info.update(get_complexity_info(env))
    scale = 1.0
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
    info['ego_action_scale'] = scale

    # Add episode summary when episode ends
    if done:
        info['episode'] = {
            'r': env.episode_reward,
            'l': env.current_step,
        }

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
    if env._last_completed_complexity_info is not None:
        return dict(env._last_completed_complexity_info)

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
