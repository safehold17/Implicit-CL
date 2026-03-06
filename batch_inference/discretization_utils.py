"""
离散化工具纯函数。

从 ctrl-sim dataset.py 和 policy.py 中提取的纯数学操作，
主进程 ExternalTeacher 与子进程 PerVehicleAutoregressivePolicy 共用，
避免逻辑漂移。所有函数仅依赖 cfg 中的离散化参数，无需加载 dataset 实例。
"""
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# tilt logits
# ---------------------------------------------------------------------------

def get_tilt_logits(rtg_discretization: int,
                    goal_tilt: int, veh_tilt: int, road_tilt: int) -> np.ndarray:
    """计算 tilt 偏移 logits。

    对齐: datasets/rl_waymo/dataset.py  get_tilt_logits()

    Returns:
        shape (rtg_discretization, 3) 的 numpy 数组
    """
    rtg_bin_values = np.zeros((rtg_discretization, 3))
    rtg_bin_values[:, 0] = goal_tilt * np.linspace(0, 1, rtg_discretization)
    rtg_bin_values[:, 1] = veh_tilt * np.linspace(0, 1, rtg_discretization)
    rtg_bin_values[:, 2] = road_tilt * np.linspace(0, 1, rtg_discretization)
    return rtg_bin_values


# ---------------------------------------------------------------------------
# undiscretize
# ---------------------------------------------------------------------------

def undiscretize_actions(actions: np.ndarray,
                         accel_discretization: int, steer_discretization: int,
                         min_accel: float, max_accel: float,
                         min_steer: float, max_steer: float) -> np.ndarray:
    """离散 action index → 连续 (accel, steer)。

    对齐: datasets/rl_waymo/dataset.py  undiscretize_actions()

    Args:
        actions: shape (...) 的整数 numpy 数组，combined action index
        其余参数来自 cfg.dataset.waymo
    Returns:
        shape (*actions.shape, 2) 的连续值数组 [accel, steer]
    """
    actions_shape = actions.shape + (2,)
    continuous = np.zeros(actions_shape)

    continuous[..., 0] = actions // steer_discretization   # accel component
    continuous[..., 1] = actions % steer_discretization    # steer component

    continuous[..., 0] /= (accel_discretization - 1)
    continuous[..., 1] /= (steer_discretization - 1)

    continuous[..., 0] = continuous[..., 0] * (max_accel - min_accel) + min_accel
    continuous[..., 1] = continuous[..., 1] * (max_steer - min_steer) + min_steer

    return continuous


def undiscretize_rtgs(rtgs: np.ndarray,
                      rtg_discretization: int,
                      min_rtg_pos: float, max_rtg_pos: float,
                      min_rtg_veh: float, max_rtg_veh: float,
                      min_rtg_road: float, max_rtg_road: float) -> np.ndarray:
    """离散 RTG index → 连续 RTG 值。

    对齐: datasets/rl_waymo/dataset.py  undiscretize_rtgs()

    Args:
        rtgs: shape (..., 3) 的整数 numpy 数组
        其余参数来自 cfg.dataset.waymo
    Returns:
        shape (..., 3) 的连续值数组 [goal, veh, road]
    """
    continuous = np.zeros_like(rtgs, dtype=float)
    continuous[..., 0] = rtgs[..., 0] / (rtg_discretization - 1)
    continuous[..., 1] = rtgs[..., 1] / (rtg_discretization - 1)
    continuous[..., 2] = rtgs[..., 2] / (rtg_discretization - 1)

    continuous[..., 0] = continuous[..., 0] * (max_rtg_pos - min_rtg_pos) + min_rtg_pos
    continuous[..., 1] = continuous[..., 1] * (max_rtg_veh - min_rtg_veh) + min_rtg_veh
    continuous[..., 2] = continuous[..., 2] * (max_rtg_road - min_rtg_road) + min_rtg_road

    return continuous


def undiscretize_action_index(action_idx: int,
                              accel_discretization: int, steer_discretization: int,
                              min_accel: float, max_accel: float,
                              min_steer: float, max_steer: float) -> tuple[float, float]:
    accel_idx = action_idx // steer_discretization
    steer_idx = action_idx % steer_discretization
    accel = accel_idx / (accel_discretization - 1)
    steer = steer_idx / (steer_discretization - 1)
    accel = accel * (max_accel - min_accel) + min_accel
    steer = steer * (max_steer - min_steer) + min_steer
    return float(accel), float(steer)


def undiscretize_rtg_indices(goal_idx: int,
                             veh_idx: int,
                             road_idx: int,
                             rtg_discretization: int,
                             min_rtg_pos: float, max_rtg_pos: float,
                             min_rtg_veh: float, max_rtg_veh: float,
                             min_rtg_road: float, max_rtg_road: float) -> tuple[float, float, float]:
    goal = goal_idx / (rtg_discretization - 1)
    veh = veh_idx / (rtg_discretization - 1)
    road = road_idx / (rtg_discretization - 1)
    goal = goal * (max_rtg_pos - min_rtg_pos) + min_rtg_pos
    veh = veh * (max_rtg_veh - min_rtg_veh) + min_rtg_veh
    road = road * (max_rtg_road - min_rtg_road) + min_rtg_road
    return float(goal), float(veh), float(road)


# ---------------------------------------------------------------------------
# RTG decode (logits → sample → undiscretize)
# ---------------------------------------------------------------------------

def decode_predicted_rtg(rtg_logits_3: torch.Tensor,
                         tilt_logits_np,
                         rtg_discretization: int,
                         min_rtg_pos: float, max_rtg_pos: float,
                         min_rtg_veh: float, max_rtg_veh: float,
                         min_rtg_road: float, max_rtg_road: float,
                         device='cuda',
                         generator: torch.Generator = None):
    """RTG 解码：logits → tilt → softmax → multinomial → undiscretize。

    对齐: policies/policy.py  process_predicted_rtg() 中 RTG 解码部分

    Args:
        rtg_logits_3: shape (rtg_discretization, 3) — 单个 agent 在 token_index 处的 RTG logits，
                      已 reshape 好的 (rtg_disc, num_reward_components)
        tilt_logits_np: shape (rtg_discretization, 3) — get_tilt_logits() 的输出
        rtg_discretization, min/max_rtg_*: 离散化参数
        device: 计算设备
        generator: 可选的 torch.Generator（用于可复现采样）

    Returns:
        (next_rtg_discrete, next_rtg_continuous)
        next_rtg_discrete: (goal_idx, veh_idx, road_idx) — 离散索引 tuple[int, int, int]
        next_rtg_continuous: (goal_val, veh_val, road_val) — 连续值 tuple[float, float, float]
    """
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


# ---------------------------------------------------------------------------
# action decode (logits → sample → undiscretize)
# ---------------------------------------------------------------------------

def decode_predicted_action(action_logits: torch.Tensor,
                            action_temperature: float,
                            nucleus_sampling: bool,
                            nucleus_threshold: float,
                            accel_discretization: int, steer_discretization: int,
                            min_accel: float, max_accel: float,
                            min_steer: float, max_steer: float,
                            generator: torch.Generator = None):
    """Action 解码：logits → temperature → (nucleus) → softmax → multinomial → undiscretize。

    对齐: policies/autoregressive_policy.py  predict() 中 action 采样部分

    Args:
        action_logits: shape (action_dim,) — 单个 agent 在 token_index 处的 action logits
        action_temperature: 温度参数
        nucleus_sampling: 是否 nucleus sampling
        nucleus_threshold: nucleus 阈值
        accel/steer_discretization, min/max_accel/steer: 离散化参数
        generator: 可选的 torch.Generator

    Returns:
        (accel, steer) — 连续值 tuple[float, float]
    """
    if nucleus_sampling:
        action_probs = F.softmax(action_logits / action_temperature, dim=0)
        sorted_probs, sorted_indices = torch.sort(action_probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        selected = cum_probs < nucleus_threshold
        selected = torch.cat([
            selected.new_ones(selected.shape[:-1] + (1,)),
            selected[..., :-1],
        ], dim=-1)
        new_probs = sorted_probs[selected]
        new_probs = new_probs / new_probs.sum()
        action_dis = torch.zeros_like(action_logits)
        action_dis[sorted_indices[selected]] = new_probs
    else:
        action_dis = F.softmax(action_logits / action_temperature, dim=0)

    action_idx = torch.multinomial(action_dis, 1, generator=generator)
    action_idx_int = int(action_idx)
    return undiscretize_action_index(
        action_idx_int,
        accel_discretization,
        steer_discretization,
        min_accel,
        max_accel,
        min_steer,
        max_steer,
    )
