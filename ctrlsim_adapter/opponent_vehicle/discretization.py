"""
负责 CtRL-Sim 运行时共享的 tilting logits、动作与 RTG 的离散化转换。
该模块被适配器侧和 batch inference 侧共同复用，保证解码语义与原始模型一致。
Provides shared helpers for tilt logits plus action/RTG discretization conversions in CtRL-Sim runtime code.
Keeps decoding semantics aligned with the model across both the adapter and batch-inference paths.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_tilt_logits(
    rtg_discretization: int,
    goal_tilt: int,
    veh_tilt: int,
    road_tilt: int,
) -> np.ndarray:
    """Build RTG tilt logits offsets."""
    rtg_bin_values = np.zeros((rtg_discretization, 3))
    rtg_bin_values[:, 0] = goal_tilt * np.linspace(0, 1, rtg_discretization)
    rtg_bin_values[:, 1] = veh_tilt * np.linspace(0, 1, rtg_discretization)
    rtg_bin_values[:, 2] = road_tilt * np.linspace(0, 1, rtg_discretization)
    return rtg_bin_values


def undiscretize_actions(
    actions: np.ndarray,
    accel_discretization: int,
    steer_discretization: int,
    min_accel: float,
    max_accel: float,
    min_steer: float,
    max_steer: float,
) -> np.ndarray:
    """Map discrete action indices to continuous accel/steer values."""
    continuous = np.zeros(actions.shape + (2,))
    continuous[..., 0] = actions // steer_discretization
    continuous[..., 1] = actions % steer_discretization

    continuous[..., 0] /= accel_discretization - 1
    continuous[..., 1] /= steer_discretization - 1

    continuous[..., 0] = continuous[..., 0] * (max_accel - min_accel) + min_accel
    continuous[..., 1] = continuous[..., 1] * (max_steer - min_steer) + min_steer
    return continuous


def undiscretize_rtgs(
    rtgs: np.ndarray,
    rtg_discretization: int,
    min_rtg_pos: float,
    max_rtg_pos: float,
    min_rtg_veh: float,
    max_rtg_veh: float,
    min_rtg_road: float,
    max_rtg_road: float,
) -> np.ndarray:
    """Map discrete RTG indices to continuous RTG values."""
    continuous = np.zeros_like(rtgs, dtype=float)
    continuous[..., 0] = rtgs[..., 0] / (rtg_discretization - 1)
    continuous[..., 1] = rtgs[..., 1] / (rtg_discretization - 1)
    continuous[..., 2] = rtgs[..., 2] / (rtg_discretization - 1)

    continuous[..., 0] = continuous[..., 0] * (max_rtg_pos - min_rtg_pos) + min_rtg_pos
    continuous[..., 1] = continuous[..., 1] * (max_rtg_veh - min_rtg_veh) + min_rtg_veh
    continuous[..., 2] = continuous[..., 2] * (max_rtg_road - min_rtg_road) + min_rtg_road
    return continuous


def undiscretize_action_index(
    action_idx: int,
    accel_discretization: int,
    steer_discretization: int,
    min_accel: float,
    max_accel: float,
    min_steer: float,
    max_steer: float,
) -> tuple[float, float]:
    """Map one discrete action index to continuous accel/steer values."""
    accel_idx = action_idx // steer_discretization
    steer_idx = action_idx % steer_discretization
    accel = accel_idx / (accel_discretization - 1)
    steer = steer_idx / (steer_discretization - 1)
    accel = accel * (max_accel - min_accel) + min_accel
    steer = steer * (max_steer - min_steer) + min_steer
    return float(accel), float(steer)


def undiscretize_rtg_indices(
    goal_idx: int,
    veh_idx: int,
    road_idx: int,
    rtg_discretization: int,
    min_rtg_pos: float,
    max_rtg_pos: float,
    min_rtg_veh: float,
    max_rtg_veh: float,
    min_rtg_road: float,
    max_rtg_road: float,
) -> tuple[float, float, float]:
    """Map one set of discrete RTG indices to continuous RTG values."""
    goal = goal_idx / (rtg_discretization - 1)
    veh = veh_idx / (rtg_discretization - 1)
    road = road_idx / (rtg_discretization - 1)
    goal = goal * (max_rtg_pos - min_rtg_pos) + min_rtg_pos
    veh = veh * (max_rtg_veh - min_rtg_veh) + min_rtg_veh
    road = road * (max_rtg_road - min_rtg_road) + min_rtg_road
    return float(goal), float(veh), float(road)


def decode_predicted_rtg(
    rtg_logits_3: torch.Tensor,
    tilt_logits_np,
    rtg_discretization: int,
    min_rtg_pos: float,
    max_rtg_pos: float,
    min_rtg_veh: float,
    max_rtg_veh: float,
    min_rtg_road: float,
    max_rtg_road: float,
    device: str = "cuda",
    generator: torch.Generator | None = None,
):
    """Sample one RTG tuple from logits and undiscretize it."""
    if isinstance(tilt_logits_np, torch.Tensor):
        tilt = tilt_logits_np
    else:
        tilt = torch.from_numpy(tilt_logits_np).to(device)

    goal_dis = F.softmax(rtg_logits_3[:, 0] + tilt[:, 0], dim=0)
    veh_dis = F.softmax(rtg_logits_3[:, 1] + tilt[:, 1], dim=0)
    road_dis = F.softmax(rtg_logits_3[:, 2] + tilt[:, 2], dim=0)

    goal_idx = torch.multinomial(goal_dis, 1, generator=generator)
    veh_idx = torch.multinomial(veh_dis, 1, generator=generator)
    road_idx = torch.multinomial(road_dis, 1, generator=generator)

    goal_idx_int = int(goal_idx)
    veh_idx_int = int(veh_idx)
    road_idx_int = int(road_idx)
    continuous = undiscretize_rtg_indices(
        goal_idx_int,
        veh_idx_int,
        road_idx_int,
        rtg_discretization,
        min_rtg_pos,
        max_rtg_pos,
        min_rtg_veh,
        max_rtg_veh,
        min_rtg_road,
        max_rtg_road,
    )

    return (goal_idx_int, veh_idx_int, road_idx_int), continuous


def decode_predicted_action(
    action_logits: torch.Tensor,
    action_temperature: float,
    nucleus_sampling: bool,
    nucleus_threshold: float,
    accel_discretization: int,
    steer_discretization: int,
    min_accel: float,
    max_accel: float,
    min_steer: float,
    max_steer: float,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Sample one action and undiscretize it."""
    if nucleus_sampling:
        action_probs = F.softmax(action_logits / action_temperature, dim=0)
        sorted_probs, sorted_indices = torch.sort(action_probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        selected = cum_probs < nucleus_threshold
        selected = torch.cat(
            [selected.new_ones(selected.shape[:-1] + (1,)), selected[..., :-1]],
            dim=-1,
        )
        new_probs = sorted_probs[selected]
        new_probs = new_probs / new_probs.sum()
        action_dis = torch.zeros_like(action_logits)
        action_dis[sorted_indices[selected]] = new_probs
    else:
        action_dis = F.softmax(action_logits / action_temperature, dim=0)

    action_idx = torch.multinomial(action_dis, 1, generator=generator)
    return undiscretize_action_index(
        int(action_idx),
        accel_discretization,
        steer_discretization,
        min_accel,
        max_accel,
        min_steer,
        max_steer,
    )
