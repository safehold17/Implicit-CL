"""Configuration helpers for the Nocturne-CtrlSim adversarial environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ctrlsim_adapter.policy_reweighting_helpers import AdversarialRTGConfig


@dataclass(frozen=True)
class NocturneCtrlSimEnvConfig:
    """Normalized constructor config for ``NocturneCtrlSimAdversarial``."""

    scenario_index_path: str
    opponent_checkpoint: str
    scenario_data_dir: str
    preprocess_dir: str
    vehicle_map_path: str | None
    preproc_cache_size: int
    opponent_k: int
    max_episode_steps: int | None
    device: str
    cfg: Any
    seed: int
    fixed_environment: bool
    random_z_dim: int
    dynamic_scenario_pool: bool
    max_scenario_pool_size: int
    tilting_mode: str
    mutation_mode: str
    mutation_range: float
    show_tilting_params: bool
    show_vehicle_ids: bool
    show_ego_vehicle_selection: bool
    remove_background_vehicles: bool
    requested_obs_dim: Any
    student_accel_discretization: int
    student_steer_discretization: int
    tilt_range: tuple[float, float]
    action_repeat_frequency: int
    kl_loss_computation_frequency: int
    sparse_inference_action_repeat: bool
    use_ego_ctrlsim_kl_loss: bool
    enhanced_regret: bool
    use_policy_reweighting: bool
    policy_reweighting_target: str
    reweighting_frequency: int
    policy_reweighting_config: AdversarialRTGConfig
    inference_precision: str
    opponent_runtime_mode: str
    done_on_position_reached_only: bool
    goal_pos_tolerance: float
    solvable_progress_threshold: float
    student_num_neighbors: int
    student_top_k_road: int
    veh_veh_collision_rew_multiplier: float
    veh_edge_collision_rew_multiplier: float
    pos_target_achieved_rew_multiplier: float
    use_persistent_position_reward: bool
    use_pos_shaped: bool
    use_approaching_goal: bool
    use_speed_shaped: bool
    use_heading_shaped: bool
    use_speed_heading_target: bool
    shaped_goal_reward: bool
    shaped_goal_distance_scaling: float
    approaching_goal_scaling: float
    use_veh_veh_shaped: bool
    use_veh_edge_shaped: bool
    max_veh_veh_distance: float
    veh_edge_reward_distance_clip: float
    reset_random_max_retries: int


def _validate_min_int(name: str, value: Any, minimum: int) -> int:
    """Convert a config value to int and enforce a lower bound."""
    int_value = int(value)
    if int_value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {int_value}")
    return int_value


def _normalize_tilt_range(tilt_range: Any) -> tuple[float, float]:
    """Normalize the configured tilt range into a validated float pair."""
    if tilt_range is None:
        return (-25.0, 25.0)
    try:
        tilt_low, tilt_high = tilt_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tilt_range must be a length-2 iterable of (low, high)"
        ) from exc
    normalized_range = (float(tilt_low), float(tilt_high))
    if normalized_range[0] > normalized_range[1]:
        raise ValueError(
            "tilt_range must satisfy low <= high, "
            f"got {normalized_range}"
        )
    return normalized_range


def build_nocturne_ctrlsim_env_config(
    *,
    scenario_index_path: str,
    opponent_checkpoint: str,
    scenario_data_dir: str,
    preprocess_dir: str,
    kwargs: Mapping[str, Any],
) -> NocturneCtrlSimEnvConfig:
    """Build a validated config object from constructor kwargs."""
    student_accel_discretization = _validate_min_int(
        "student_accel_discretization",
        kwargs["student_accel_discretization"],
        2,
    )
    student_steer_discretization = _validate_min_int(
        "student_steer_discretization",
        kwargs["student_steer_discretization"],
        2,
    )
    action_repeat_frequency = _validate_min_int(
        "action_repeat_frequency",
        kwargs.get("action_repeat_frequency", 2),
        1,
    )
    kl_loss_computation_frequency = _validate_min_int(
        "kl_loss_computation_frequency",
        kwargs.get("kl_loss_computation_frequency", 2),
        1,
    )
    reweighting_frequency = _validate_min_int(
        "reweighting_frequency",
        kwargs.get("reweighting_frequency", 1),
        1,
    )
    opponent_k = int(kwargs.get("opponent_k", 7))
    if opponent_k < 0:
        raise ValueError(f"opponent_k must be non-negative, got {opponent_k}")

    tilting_mode = str(kwargs.get("tilting_mode", "per_vehicle"))
    if tilting_mode not in {"global", "per_vehicle", "none"}:
        raise ValueError(
            "tilting_mode must be 'global', 'per_vehicle', or 'none', "
            f"got {tilting_mode}"
        )
    mutation_mode = str(kwargs.get("mutation_mode", "all"))
    if mutation_mode not in {"one", "all"}:
        raise ValueError(
            "mutation_mode must be 'one' or 'all', "
            f"got {mutation_mode}"
        )

    policy_reweighting_target = str(kwargs.get("policy_reweighting_target", "rtg"))
    if policy_reweighting_target not in {"rtg", "action"}:
        raise ValueError(
            "policy_reweighting_target must be 'rtg' or 'action', "
            f"got {policy_reweighting_target}"
        )

    use_policy_reweighting = bool(
        kwargs.get("use_policy_reweighting", False)
    )
    if use_policy_reweighting and tilting_mode != "none":
        raise ValueError(
            "use_policy_reweighting requires tilting_mode='none', "
            f"got tilting_mode={tilting_mode}"
        )
    policy_reweighting_config = AdversarialRTGConfig(
        enabled=use_policy_reweighting,
        reward_scale=float(kwargs.get("policy_reweighting_reward_scale", 1.0)),
        epsilon=float(kwargs.get("policy_reweighting_epsilon", 1e-6)),
    )

    return NocturneCtrlSimEnvConfig(
        scenario_index_path=scenario_index_path,
        opponent_checkpoint=opponent_checkpoint,
        scenario_data_dir=scenario_data_dir,
        preprocess_dir=preprocess_dir,
        vehicle_map_path=kwargs.get("vehicle_map_path", "data/vehicle_map_valid.json"),
        preproc_cache_size=int(kwargs.get("preproc_cache_size", 64)),
        opponent_k=opponent_k,
        max_episode_steps=kwargs.get("max_episode_steps", 90),
        device=str(kwargs.get("device", "cuda")),
        cfg=kwargs.get("cfg"),
        seed=int(kwargs.get("seed", 0)),
        fixed_environment=bool(kwargs.get("fixed_environment", False)),
        random_z_dim=int(kwargs.get("random_z_dim", 50)),
        dynamic_scenario_pool=bool(kwargs.get("dynamic_scenario_pool", False)),
        max_scenario_pool_size=int(kwargs.get("max_scenario_pool_size", 10000)),
        tilting_mode=tilting_mode,
        mutation_mode=mutation_mode,
        mutation_range=float(kwargs.get("mutation_range", 5.0)),
        show_tilting_params=bool(kwargs.get("show_tilting_params", True)),
        show_vehicle_ids=bool(kwargs.get("show_vehicle_ids", True)),
        show_ego_vehicle_selection=bool(kwargs.get("show_ego_vehicle_selection", True)),
        remove_background_vehicles=bool(kwargs.get("remove_background_vehicles", True)),
        requested_obs_dim=kwargs.get("obs_dim"),
        student_accel_discretization=student_accel_discretization,
        student_steer_discretization=student_steer_discretization,
        tilt_range=_normalize_tilt_range(kwargs.get("tilt_range")),
        action_repeat_frequency=action_repeat_frequency,
        kl_loss_computation_frequency=kl_loss_computation_frequency,
        sparse_inference_action_repeat=bool(
            kwargs.get("sparse_inference_action_repeat", False)
        ),
        use_ego_ctrlsim_kl_loss=bool(kwargs.get("use_ego_ctrlsim_kl_loss", False)),
        enhanced_regret=bool(kwargs.get("enhanced_regret", False)),
        use_policy_reweighting=use_policy_reweighting,
        policy_reweighting_target=policy_reweighting_target,
        reweighting_frequency=reweighting_frequency,
        policy_reweighting_config=policy_reweighting_config,
        inference_precision=str(kwargs.get("inference_precision", "fp32")),
        opponent_runtime_mode=str(kwargs.get("opponent_runtime_mode", "normal")),
        done_on_position_reached_only=bool(
            kwargs.get("done_on_position_reached_only", True)
        ),
        goal_pos_tolerance=float(kwargs.get("goal_pos_tolerance", 2.0)),
        solvable_progress_threshold=float(
            kwargs.get("solvable_progress_threshold", 0.85)
        ),
        student_num_neighbors=int(kwargs.get("student_num_neighbors", 16)),
        student_top_k_road=int(kwargs.get("student_top_k_road", 64)),
        veh_veh_collision_rew_multiplier=float(
            kwargs.get("veh_veh_collision_rew_multiplier", 10.0)
        ),
        veh_edge_collision_rew_multiplier=float(
            kwargs.get("veh_edge_collision_rew_multiplier", 10.0)
        ),
        pos_target_achieved_rew_multiplier=float(
            kwargs.get("pos_target_achieved_rew_multiplier", 10.0)
        ),
        use_persistent_position_reward=bool(
            kwargs.get("use_persistent_position_reward", False)
        ),
        use_pos_shaped=bool(kwargs.get("use_pos_shaped", True)),
        use_approaching_goal=bool(kwargs.get("use_approaching_goal", True)),
        use_speed_shaped=bool(kwargs.get("use_speed_shaped", True)),
        use_heading_shaped=bool(kwargs.get("use_heading_shaped", True)),
        use_speed_heading_target=bool(kwargs.get("use_speed_heading_target", True)),
        shaped_goal_reward=bool(kwargs.get("shaped_goal_reward", True)),
        shaped_goal_distance_scaling=float(
            kwargs.get("shaped_goal_distance_scaling", 0.2)
        ),
        approaching_goal_scaling=float(kwargs.get("approaching_goal_scaling", 1.0)),
        use_veh_veh_shaped=bool(kwargs.get("use_veh_veh_shaped", True)),
        use_veh_edge_shaped=bool(kwargs.get("use_veh_edge_shaped", True)),
        max_veh_veh_distance=float(kwargs.get("max_veh_veh_distance", 15.0)),
        veh_edge_reward_distance_clip=float(
            kwargs.get("veh_edge_reward_distance_clip", 5.0)
        ),
        reset_random_max_retries=int(kwargs.get("reset_random_max_retries", 10)),
    )
