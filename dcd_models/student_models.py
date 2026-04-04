
"""
Student model,Late Fusion architecture in gpudrive

- gpudrive/networks/late_fusion.py: Late Fusion
- dcd_models/walker_models.py: DCD Policy 
"""

import numpy as np
import torch
import torch.nn as nn

from .common import DeviceAwareModule
from .distributions import Categorical
from envs.nocturne_ctrlsim.student.observation_action import (
    StudentObservationConfig,
    get_student_obs_dim,
    split_student_observation,
)


class LateFusionBase(nn.Module):
    """
    Late Fusion feature extraction base class
    
    Embed Ego, Partner, and Road Graph modalities separately and then fuse.
    Reference: NeuralNet class in gpudrive/networks/late_fusion.py
    
    Args:
        input_dim: Embedding dimension for each modality
        hidden_dim: Hidden dimension after fusion
        max_controlled_agents: Maximum number of controllable agents
        top_k_road_points: Number of recent road points
        dropout: Dropout probability
        act_func: Activation function ("tanh" or "gelu")
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
        self.observation_config = StudentObservationConfig(
            max_neighbors=self.max_observable_agents,
            top_k_road_points=self.top_k_road_points,
        )
        
        # activation function
        if act_func == "tanh":
            self.act_func = nn.Tanh()
        elif act_func == "gelu":
            self.act_func = nn.GELU()
        else:
            self.act_func = nn.ReLU()
        
        # Indices for different observation vector parts
        # Ego state embedding
        self.ego_embed = nn.Sequential(
            self._layer_init(
                nn.Linear(self.observation_config.ego_feat_dim, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # Partner observation embedding
        self.partner_embed = nn.Sequential(
            self._layer_init(
                nn.Linear(self.observation_config.partner_feat_dim, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # Road Graph embedding
        self.road_map_embed = nn.Sequential(
            self._layer_init(
                nn.Linear(self.observation_config.road_graph_feat_dim, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        
        # Fusion layer
        self.shared_embed = nn.Sequential(
            nn.Linear(input_dim * self.num_modes, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Weight initialization (pufferlib.pytorch.layer_init)"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def unpack_obs(self, obs_flat: torch.Tensor):
        """
        Unpack flattened observation vector
        
        Args:
            obs_flat: (batch_size, obs_dim) Flattened observation
        
        Returns:
            ego_state: (batch_size, EGO_FEAT_DIM)
            road_objects: (batch_size, max_observable_agents, PARTNER_FEAT_DIM)
            road_graph: (batch_size, top_k_road_points, ROAD_GRAPH_FEAT_DIM)
        """
        ego_state, road_objects, road_graph = split_student_observation(
            obs_flat,
            self.observation_config,
        )
        return ego_state, road_objects, road_graph
    
    def encode_observations(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode observations
        
        Args:
            observation: (batch_size, obs_dim) Flattened observation
        
        Returns:
            hidden: (batch_size, hidden_dim) Fused features
        """
        ego_state, road_objects, road_graph = self.unpack_obs(observation)
        
        # Independent embedding for each modality
        ego_embed = self.ego_embed(ego_state)
        
        # Partner: embed then max pooling
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        
        # Road Graph: embed then max pooling
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        
        # Concatenate all embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)
        
        # Fusion layer
        return self.shared_embed(embed)
    
    @property
    def output_size(self):
        """Output feature dimension"""
        return self.hidden_dim

    @property
    def expected_obs_dim(self):
        """Expected flattened observation dimension."""
        return get_student_obs_dim(self.observation_config)


class StudentPolicy(DeviceAwareModule):
    """
    Student driving policy (DCD Policy interface)
    
    This class implements the full policy interface required by the DCD framework:
    - act(): Generate actions during rollout
    - get_value(): Compute state value
    - evaluate_actions(): Evaluate actions during PPO update
    
    The network architecture uses Late Fusion, consistent with gpudrive.
    Suitable for driving environments like Nocturne + ctrl-sim.
    
    Args:
        obs_shape: Observation space shape
        action_space: Action space
        input_dim: Input embedding dimension for each modality
        hidden_dim: Hidden dimension after fusion
        max_controlled_agents: Maximum number of controllable agents
        top_k_road_points: Number of nearest road points
        dropout: Dropout probability
        act_func: Activation function
        recurrent: Whether to use recurrent network (not implemented)
        base_kwargs: Additional kwargs for base network
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
        
        # Feature extraction base network
        self.base = LateFusionBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_controlled_agents=max_controlled_agents,
            top_k_road_points=top_k_road_points,
            dropout=dropout,
            act_func=act_func,
        )

        if isinstance(obs_shape, (tuple, list)):
            if len(obs_shape) != 1:
                raise ValueError(f"StudentPolicy expects 1D obs_shape, got {obs_shape}")
            obs_dim = int(obs_shape[0])
        else:
            obs_dim = int(obs_shape)
        expected_obs_dim = self.base.expected_obs_dim
        if obs_dim != expected_obs_dim:
            raise ValueError(
                "StudentPolicy obs_shape mismatch: "
                f"expected {expected_obs_dim} = "
                f"{self.base.observation_config.ego_feat_dim} + "
                f"{self.base.observation_config.max_neighbors}*{self.base.observation_config.partner_feat_dim} + "
                f"{self.base.observation_config.top_k_road_points}*{self.base.observation_config.road_graph_feat_dim}, "
                f"got {obs_dim}. "
                "Please align the centralized student observation config across env and model."
            )
        
        if action_space.__class__.__name__ != "Discrete":
            raise ValueError(
                "StudentPolicy expects a Discrete action space, "
                f"got {action_space.__class__.__name__}"
            )
        self.action_dim = int(action_space.n)
        self.dist = Categorical(hidden_dim, self.action_dim)
        
        # Critic: output state value
        self.critic = self._layer_init(
            nn.Linear(hidden_dim, 1), std=1.0
        )
        
        # Recurrent network support (not implemented)
        self._recurrent = recurrent
        if recurrent:
            raise NotImplementedError(
                "Recurrent policies not yet supported for Student driving policy"
            )
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Weight initialization"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    @property
    def is_recurrent(self):
        """Whether using recurrent network"""
        return self._recurrent
    
    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size (1 if not recurrent)"""
        return 1
    
    def forward(self, inputs):
        """Simplified forward pass (for inference)"""
        value, action, action_log_probs, rnn_hxs = self.act(
            inputs, rnn_hxs=None, masks=None, deterministic=False
        )
        return action
    
    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        """
        Generate actions based on observations (called during Rollout)
        
        Args:
            inputs: Observations (batch_size, obs_dim)
            rnn_hxs: Recurrent hidden states (currently unused)
            masks: Episode masks (currently unused)
            deterministic: Whether to sample deterministically
        
        Returns:
            value: State value (batch_size, 1)
            action: Action ids (batch_size, 1)
            action_log_dist: Action logits (batch_size, action_dim)
            rnn_hxs: Updated hidden states
        """
        # Feature extraction
        hidden = self.base.encode_observations(inputs)
        
        # Critic: state value
        value = self.critic(hidden)
        
        # Actor: discrete action
        dist = self.dist(hidden)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        action_log_dist = dist.logits
        
        return value, action, action_log_dist, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs=None, masks=None):
        """
        Compute state value (called during Rollout)
        
        Args:
            inputs: Observations (batch_size, obs_dim)
            rnn_hxs: Recurrent hidden states
            masks: Episode masks
        
        Returns:
            value: State value  (batch_size, 1)
        """
        hidden = self.base.encode_observations(inputs)
        return self.critic(hidden)
    
    def evaluate_actions(
        self, inputs, rnn_hxs, masks, action, return_policy_logits=False
    ):
        """
        Evaluate actions (called during PPO update)
        
        Args:
            inputs: Observations (batch_size, obs_dim)
            rnn_hxs: Recurrent hidden states
            masks: Episode masks
            action: Actions to evaluate (batch_size, action_dim)
            return_policy_logits: Whether to return the full distribution
        
        Returns:
            value: State value (batch_size, 1)
            action_log_probs: Action log probabilities (batch_size, 1)
            dist_entropy: Policy entropy (scalar)
            rnn_hxs: Updated hidden states
            [dist]: Optional, full distribution
        """
        hidden = self.base.encode_observations(inputs)
        value = self.critic(hidden)
        
        dist = self.dist(hidden)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        
        return value, action_log_probs, dist_entropy, rnn_hxs
