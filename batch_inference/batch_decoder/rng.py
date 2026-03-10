from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def get_device_rng_state(device: Any) -> torch.Tensor:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        return torch.cuda.get_rng_state(torch_device).clone()
    return torch.get_rng_state().clone()


def get_action_reservation_weights(
    teacher: Any,
    nucleus_sampling: bool,
    nucleus_threshold: float,
) -> torch.Tensor:
    action_dim = int(teacher.accel_discretization) * int(teacher.steer_discretization)
    cache = getattr(teacher, "_action_reservation_weights_cache", None)
    if cache is None:
        cache = {}
        teacher._action_reservation_weights_cache = cache

    cache_key = (action_dim, bool(nucleus_sampling), float(nucleus_threshold), str(teacher.device))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    weights = torch.ones(action_dim, dtype=torch.float32, device=teacher.device)
    if nucleus_sampling:
        sorted_probs, sorted_indices = torch.sort(weights / float(action_dim), descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        selected = cum_probs < nucleus_threshold
        selected = torch.cat(
            [selected.new_ones(selected.shape[:-1] + (1,)), selected[..., :-1]],
            dim=-1,
        )
        selected_probs = sorted_probs[selected]
        selected_probs = selected_probs / selected_probs.sum()
        weights = torch.zeros(action_dim, dtype=torch.float32, device=teacher.device)
        weights[sorted_indices[selected]] = selected_probs

    cache[cache_key] = weights
    return weights


def get_decode_generator(teacher: Any) -> torch.Generator:
    generator = getattr(teacher, "_decode_generator", None)
    device = torch.device(teacher.device)
    if generator is None or generator.device != device:
        generator = torch.Generator(device=device)
        teacher._decode_generator = generator
    return generator


def as_rng_state_tensor(worker_rng_state: Any) -> Optional[torch.Tensor]:
    if worker_rng_state is None:
        return None
    if isinstance(worker_rng_state, torch.Tensor):
        return worker_rng_state.detach().cpu().clone().to(dtype=torch.uint8)
    return torch.as_tensor(
        np.asarray(worker_rng_state, dtype=np.uint8),
        dtype=torch.uint8,
    )


def get_env_sampling_generator(
    teacher: Any,
    env_idx: int,
    step_t: int,
    worker_rng_state: Any,
) -> torch.Generator:
    cache = getattr(teacher, "_env_sampling_generators", None)
    if cache is None:
        cache = {}
        teacher._env_sampling_generators = cache

    cached_entry = cache.get(int(env_idx))
    if cached_entry is not None and int(cached_entry["step_t"]) == int(step_t):
        return cached_entry["generator"]

    generator = torch.Generator(device=torch.device(teacher.device))
    rng_state = as_rng_state_tensor(worker_rng_state)
    if rng_state is None:
        rng_state = get_device_rng_state(teacher.device)
    generator.set_state(rng_state)
    cache[int(env_idx)] = {
        "step_t": int(step_t),
        "generator": generator,
    }
    return generator


def get_next_worker_rng_state(
    teacher: Any,
    env_idx: int,
    step_t: int,
    fallback_rng_state: Any,
) -> np.ndarray:
    cache = getattr(teacher, "_env_sampling_generators", None)
    if cache is not None:
        cached_entry = cache.get(int(env_idx))
        if cached_entry is not None and int(cached_entry["step_t"]) == int(step_t):
            return (
                cached_entry["generator"]
                .get_state()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
    if fallback_rng_state is not None:
        return np.asarray(fallback_rng_state, dtype=np.uint8)
    return (
        get_device_rng_state(teacher.device)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )


def reserve_action_rng_states_for_job(
    teacher: Any,
    job: Dict[str, Any],
    env_generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    data_veh_ids = job["focal_batch"].get("data_veh_ids", [])
    sampling = job["prepared"].get("sampling")
    if not data_veh_ids or sampling is None:
        return {}
    if env_generator is None:
        env_generator = get_decode_generator(teacher)
        env_generator.set_state(get_device_rng_state(teacher.device))
    reservation_weights = get_action_reservation_weights(
        teacher=teacher,
        nucleus_sampling=bool(sampling["nucleus_sampling"]),
        nucleus_threshold=float(sampling["nucleus_threshold"]),
    )
    reserved_states: Dict[int, torch.Tensor] = {}
    for veh_id in data_veh_ids:
        reserved_states[int(veh_id)] = env_generator.get_state().clone()
        torch.multinomial(
            reservation_weights,
            1,
            generator=env_generator,
        )
    return reserved_states
