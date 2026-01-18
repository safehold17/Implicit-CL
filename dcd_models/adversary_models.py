

"""
Teacher 策略网络（用于 PAIRED/Minimax）

适用于 Nocturne + ctrl-sim 环境的 Teacher（Adversary）策略训练。
Teacher 负责生成关卡参数（scenario_id + 3个tilt参数）。

参考实现:
- dcd_models/walker_models.py: BipedalWalkerAdversaryPolicy
- envs/nocturne_ctrlsim/adversarial.py: Adversary 观测和动作空间定义
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DeviceAwareModule
from .walker_models import DiagGaussian, MLPBase


class NocturneAdversaryPolicy(DeviceAwareModule):
    """
    Nocturne Teacher 策略（关卡生成者）
    
    Teacher 通过 4 步生成关卡：
    - Step 0: 选择 scenario_id（连续动作 [-1, 1] 映射到索引）
    - Step 1-3: 设置 goal_tilt, veh_veh_tilt, veh_edge_tilt（连续动作 [-1, 1]）
    
    观测空间：Dict{'image': level_params, 'time_step': step, 'random_z': z}
    动作空间：Box(1,) 连续动作 [-1, 1]
    
    Args:
        observation_space: Adversary 观测空间（Dict 类型）
        action_space: Adversary 动作空间（Box 类型）
        random: 是否使用随机策略（用于基线对比）
        recurrent: 是否使用循环网络
        base_kwargs: MLPBase 的额外参数（hidden_size 等）
    """
    
    def __init__(
        self, 
        observation_space, 
        action_space, 
        random=False,
        recurrent=False,
        recurrent_arch=None,
        base_kwargs=None
    ):
        super().__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        self.random = random
        # 兼容旧的 boolean recurrent 参数
        if recurrent and recurrent_arch is None:
            recurrent_arch = 'gru'  # 默认使用 GRU
        elif not recurrent:
            recurrent_arch = None
        self._recurrent = recurrent_arch is not None
        self._recurrent_arch = recurrent_arch
        
        # 解析观测空间维度
        self.design_dim = observation_space['image'].shape[0]  # level 参数维度（4）
        self.random_z_dim = observation_space['random_z'].shape[0]  # 随机向量维度
        
        # 总观测维度：design + random_z + time_step
        obs_dim = self.design_dim + self.random_z_dim + 1
        
        # 基础网络（MLP 或 RNN）
        base_kwargs['recurrent_arch'] = recurrent_arch
        if recurrent_arch is not None:
            base_kwargs['recurrent'] = True
        self.base = MLPBase(obs_dim, **base_kwargs)
        
        # 动作维度（连续动作）
        self.action_dim = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, self.action_dim)
        
        # Critic：状态价值
        # 注意：MLPBase 已经包含 critic，这里不需要额外定义
    
    @property
    def is_recurrent(self):
        """是否使用循环网络"""
        return self.base.is_recurrent
    
    @property
    def recurrent_hidden_state_size(self):
        """循环隐藏状态大小"""
        return self.base.recurrent_hidden_state_size
    
    def preprocess(self, inputs):
        """
        预处理 Adversary 观测
        
        Args:
            inputs: Dict{'image': level_params, 'time_step': step, 'random_z': z}
        
        Returns:
            obs: 拼接后的观测向量 (batch_size, obs_dim)
        """
        obs = torch.cat([
            inputs['image'], 
            inputs['random_z'], 
            inputs['time_step']
        ], dim=1)
        return obs
    
    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        """
        根据观测生成动作（Rollout 阶段调用）
        
        Args:
            inputs: Adversary 观测（Dict 类型）
            rnn_hxs: 循环隐藏状态
            masks: Episode 掩码
            deterministic: 是否确定性采样
        
        Returns:
            value: 状态价值 (batch_size, 1)
            action: 动作 (batch_size, action_dim)
            action_log_probs: 动作对数概率 (batch_size, 1)
            rnn_hxs: 更新后的隐藏状态
        """
        inputs = self.preprocess(inputs)
        
        # 随机策略（基线）
        if self.random:
            batch_size = inputs.shape[0]
            action = torch.tensor(
                np.random.uniform(-1, 1, (batch_size, self.action_dim)), 
                device=self.device,
                dtype=torch.float32
            )
            action_log_probs = torch.zeros(batch_size, 1, device=self.device)
            value = torch.zeros(batch_size, 1, device=self.device)
            return value, action, action_log_probs, rnn_hxs
        
        # MLP/RNN 前向传播
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        # 动作分布
        dist = self.dist(actor_features)
        
        # 采样动作
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        # 应用 tanh 将动作限制在 [-1, 1]
        action = torch.tanh(action)
        
        action_log_probs = dist.log_probs(action)
        
        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs=None, masks=None):
        """
        获取状态价值（计算 Advantage 时调用）
        
        Args:
            inputs: Adversary 观测（Dict 类型）
            rnn_hxs: 循环隐藏状态
            masks: Episode 掩码
        
        Returns:
            value: 状态价值 (batch_size, 1)
        """
        inputs = self.preprocess(inputs)
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value
    
    def evaluate_actions(
        self, 
        inputs, 
        rnn_hxs, 
        masks, 
        action, 
        return_policy_logits=False
    ):
        """
        评估给定动作（PPO 更新时调用）
        
        Args:
            inputs: Adversary 观测（Dict 类型）
            rnn_hxs: 循环隐藏状态
            masks: Episode 掩码
            action: 要评估的动作 (batch_size, action_dim)
            return_policy_logits: 是否返回完整分布（未使用）
        
        Returns:
            value: 状态价值 (batch_size, 1)
            action_log_probs: 动作对数概率 (batch_size, 1)
            dist_entropy: 策略熵 (scalar)
            rnn_hxs: 更新后的隐藏状态
        """
        inputs = self.preprocess(inputs)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        dist = self.dist(actor_features)
        
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy, rnn_hxs
