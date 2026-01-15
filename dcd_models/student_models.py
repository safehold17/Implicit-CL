
"""
Student 驾驶策略网络（Late Fusion 架构）

适用于 Nocturne + ctrl-sim 环境的 Student 策略训练。

参考实现:
- gpudrive/networks/late_fusion.py: Late Fusion 网络结构
- dcd_models/walker_models.py: DCD Policy 接口规范
"""

import numpy as np
import torch
import torch.nn as nn

from .common import DeviceAwareModule, init, init_tanh_
from .walker_models import DiagGaussian, FixedNormal

# ============== 观测空间常量 ==============
# 与 gpudrive/env/constants.py 保持一致
# Ego: [speed, length, width, rel_goal_x, rel_goal_y, collision_state]
# Partner: [speed, rel_pos_x, rel_pos_y, rel_orientation, length, width]
# Road graph: [pos_x, pos_y, length, scale_x, scale_y, orientation, type_onehot(7)]
EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
ROAD_GRAPH_FEAT_DIM = 13


class LateFusionBase(nn.Module):
    """
    Late Fusion 特征提取基类
    
    将 Ego、Partner、Road Graph 三种模态分别嵌入后融合。
    参考: gpudrive/networks/late_fusion.py 的 NeuralNet 类
    
    Args:
        input_dim: 各模态嵌入维度
        hidden_dim: 融合后的隐藏层维度
        max_controlled_agents: 最大可控智能体数
        top_k_road_points: 最近道路点数量
        dropout: Dropout 概率
        act_func: 激活函数 ("tanh" 或 "gelu")
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        max_controlled_agents: int = 64,
        top_k_road_points: int = 200,
        dropout: float = 0.0,
        act_func: str = "tanh",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.top_k_road_points = top_k_road_points
        self.num_modes = 3  # Ego, Partner, Road Graph
        
        # 激活函数选择
        if act_func == "tanh":
            self.act_func = nn.Tanh()
        elif act_func == "gelu":
            self.act_func = nn.GELU()
        else:
            self.act_func = nn.ReLU()
        
        # 计算观测向量中各部分的索引
        self.ego_state_idx = EGO_FEAT_DIM
        self.partner_obs_idx = EGO_FEAT_DIM + PARTNER_FEAT_DIM * self.max_observable_agents
        
        # Ego 状态嵌入
        self.ego_embed = nn.Sequential(
            self._layer_init(nn.Linear(EGO_FEAT_DIM, input_dim)),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # Partner 观测嵌入
        self.partner_embed = nn.Sequential(
            self._layer_init(nn.Linear(PARTNER_FEAT_DIM, input_dim)),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # Road Graph 嵌入
        self.road_map_embed = nn.Sequential(
            self._layer_init(nn.Linear(ROAD_GRAPH_FEAT_DIM, input_dim)),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # 融合层
        self.shared_embed = nn.Sequential(
            nn.Linear(input_dim * self.num_modes, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """权重初始化（参考 pufferlib.pytorch.layer_init）"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def unpack_obs(self, obs_flat: torch.Tensor):
        """
        解包扁平化观测向量
        
        Args:
            obs_flat: (batch_size, obs_dim) 扁平化观测
        
        Returns:
            ego_state: (batch_size, EGO_FEAT_DIM)
            road_objects: (batch_size, max_observable_agents, PARTNER_FEAT_DIM)
            road_graph: (batch_size, top_k_road_points, ROAD_GRAPH_FEAT_DIM)
        """
        # 提取各部分
        ego_state = obs_flat[:, :self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx:self.partner_obs_idx]
        road_graph_obs = obs_flat[:, self.partner_obs_idx:]
        
        # 重塑为多维张量
        road_objects = partner_obs.view(
            -1, self.max_observable_agents, PARTNER_FEAT_DIM
        )
        road_graph = road_graph_obs.view(
            -1, self.top_k_road_points, ROAD_GRAPH_FEAT_DIM
        )
        
        return ego_state, road_objects, road_graph
    
    def encode_observations(self, observation: torch.Tensor) -> torch.Tensor:
        """
        编码观测（Late Fusion 核心逻辑）
        
        Args:
            observation: (batch_size, obs_dim) 扁平化观测
        
        Returns:
            hidden: (batch_size, hidden_dim) 融合后的特征
        """
        ego_state, road_objects, road_graph = self.unpack_obs(observation)
        
        # 各模态独立嵌入
        ego_embed = self.ego_embed(ego_state)
        
        # Partner: 嵌入后 max pooling
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        
        # Road Graph: 嵌入后 max pooling
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        
        # 拼接所有嵌入
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)
        
        # 融合层
        return self.shared_embed(embed)
    
    @property
    def output_size(self):
        """输出特征维度"""
        return self.hidden_dim


class StudentPolicy(DeviceAwareModule):
    """
    Student 驾驶策略（DCD Policy 接口）
    
    这个类实现了 DCD 框架要求的完整策略接口:
    - act(): Rollout 时生成动作
    - get_value(): 计算状态价值
    - evaluate_actions(): PPO 更新时评估动作
    
    网络架构使用 Late Fusion，与 gpudrive 保持一致。
    适用于 Nocturne + ctrl-sim 等驾驶环境。
    
    Args:
        obs_shape: 观测空间形状
        action_space: 动作空间
        input_dim: 各模态嵌入维度
        hidden_dim: 融合后的隐藏层维度
        max_controlled_agents: 最大可控智能体数
        top_k_road_points: 最近道路点数量
        dropout: Dropout 概率
        act_func: 激活函数
        recurrent: 是否使用循环网络（暂不支持）
        base_kwargs: 额外参数
    """
    
    def __init__(
        self,
        obs_shape,
        action_space,
        input_dim: int = 64,
        hidden_dim: int = 128,
        max_controlled_agents: int = 64,
        top_k_road_points: int = 200,
        dropout: float = 0.0,
        act_func: str = "tanh",
        recurrent: bool = False,
        base_kwargs=None,
    ):
        super().__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        # 特征提取基础网络
        self.base = LateFusionBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_controlled_agents=max_controlled_agents,
            top_k_road_points=top_k_road_points,
            dropout=dropout,
            act_func=act_func,
        )
        
        # 连续动作维度（accel, steer 等）
        action_dim = action_space.shape[0]
        self.dist = DiagGaussian(hidden_dim, action_dim)
        
        # Critic: 输出状态价值
        self.critic = self._layer_init(
            nn.Linear(hidden_dim, 1), std=1.0
        )
        
        # 循环网络支持（暂不实现）
        self._recurrent = recurrent
        if recurrent:
            raise NotImplementedError(
                "Recurrent policies not yet supported for Student driving policy"
            )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """权重初始化"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    @property
    def is_recurrent(self):
        """是否使用循环网络"""
        return self._recurrent
    
    @property
    def recurrent_hidden_state_size(self):
        """循环隐藏状态大小（非循环返回 1）"""
        return 1
    
    def forward(self, inputs):
        """简化的前向传播（用于推理）"""
        value, action, action_log_probs, rnn_hxs = self.act(
            inputs, rnn_hxs=None, masks=None, deterministic=False
        )
        return action
    
    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        """
        根据观测生成动作（Rollout 阶段调用）
        
        Args:
            inputs: 观测 (batch_size, obs_dim)
            rnn_hxs: 循环隐藏状态（当前未使用）
            masks: Episode 掩码（当前未使用）
            deterministic: 是否确定性采样
        
        Returns:
            value: 状态价值 (batch_size, 1)
            action: 动作 (batch_size, action_dim)
            action_log_probs: 动作对数概率 (batch_size, 1)
            rnn_hxs: 更新后的隐藏状态
        """
        # 特征提取
        hidden = self.base.encode_observations(inputs)
        
        # Critic: 状态价值
        value = self.critic(hidden)
        
        # Actor: 连续动作
        dist = self.dist(hidden)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        action_log_probs = dist.log_probs(action)
        
        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs=None, masks=None):
        """
        获取状态价值（计算 Advantage 时调用）
        
        Args:
            inputs: 观测 (batch_size, obs_dim)
            rnn_hxs: 循环隐藏状态
            masks: Episode 掩码
        
        Returns:
            value: 状态价值 (batch_size, 1)
        """
        hidden = self.base.encode_observations(inputs)
        return self.critic(hidden)
    
    def evaluate_actions(
        self, inputs, rnn_hxs, masks, action, return_policy_logits=False
    ):
        """
        评估给定动作（PPO 更新时调用）
        
        Args:
            inputs: 观测 (batch_size, obs_dim)
            rnn_hxs: 循环隐藏状态
            masks: Episode 掩码
            action: 要评估的动作 (batch_size, action_dim)
            return_policy_logits: 是否返回完整分布
        
        Returns:
            value: 状态价值 (batch_size, 1)
            action_log_probs: 动作对数概率 (batch_size, 1)
            dist_entropy: 策略熵 (scalar)
            rnn_hxs: 更新后的隐藏状态
            [dist]: 可选，完整分布
        """
        hidden = self.base.encode_observations(inputs)
        value = self.critic(hidden)
        
        dist = self.dist(hidden)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        
        return value, action_log_probs, dist_entropy, rnn_hxs
