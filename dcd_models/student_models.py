
"""
Student model,Late Fusion architecture in gpudrive

- gpudrive/networks/late_fusion.py: Late Fusion
- dcd_models/walker_models.py: DCD Policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DeviceAwareModule, RNN
from .distributions import Categorical
from envs.nocturne_ctrlsim.student.observation_action import (
    StudentObservationConfig,
    get_student_obs_dim,
    split_student_observation,
)


class QueryAttentionPooling(nn.Module):
    """Pool a token set with an ego-conditioned attention query."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        num_heads = 4 if dim % 4 == 0 else 1
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool tokens into a single embedding per batch item."""
        if tokens.shape[1] == 0:
            return torch.zeros_like(query)

        mask = key_padding_mask.to(dtype=torch.bool)
        fully_padded = mask.all(dim=1)
        if fully_padded.any():
            # MultiheadAttention becomes unstable when every token is masked for
            # a batch item. Temporarily expose one token to keep attention
            # numerically defined. This does not change the final semantics
            # because fully padded rows are overwritten with zeros below.
            mask = mask.clone()
            mask[fully_padded, 0] = False

        attn_out, _ = self.attn(
            query.unsqueeze(1),
            tokens,
            tokens,
            key_padding_mask=mask,
        )
        pooled = self.norm1(query + attn_out.squeeze(1))
        pooled = self.norm2(pooled + self.ffn(pooled))

        if fully_padded.any():
            pooled = pooled.clone()
            # Restore the intended semantics: rows with no valid tokens should
            # contribute a zero pooled embedding.
            pooled[fully_padded] = 0.0
        return pooled


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
        student_partner_pooling: str = "attention",
        student_road_pooling: str = "attention",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.top_k_road_points = top_k_road_points
        self.num_modes = 3  # Ego, Partner, Road Graph
        self.road_type_feat_dim = 7
        self.student_partner_pooling = student_partner_pooling
        self.student_road_pooling = student_road_pooling
        self.observation_config = StudentObservationConfig(
            max_neighbors=self.max_observable_agents,
            top_k_road_points=self.top_k_road_points,
        )
        self.road_geom_feat_dim = (
            self.observation_config.road_graph_feat_dim - self.road_type_feat_dim
        )
        if self.student_partner_pooling not in {"attention", "max"}:
            raise ValueError(
                "student_partner_pooling must be 'attention' or 'max', "
                f"got {self.student_partner_pooling}"
            )
        if self.student_road_pooling not in {"attention", "max"}:
            raise ValueError(
                "student_road_pooling must be 'attention' or 'max', "
                f"got {self.student_road_pooling}"
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
        self.partner_attention_embed = QueryAttentionPooling(
            input_dim,
            dropout=dropout,
        )
        
        # Road geometry embedding
        self.road_geom_embed = nn.Sequential(
            self._layer_init(
                nn.Linear(self.road_geom_feat_dim, input_dim)
            ),
            nn.LayerNorm(input_dim),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        self.road_type_embed = nn.Sequential(
            self._layer_init(
                nn.Linear(self.road_type_feat_dim, input_dim)
            ),
            self.act_func,
            nn.Dropout(dropout),
            self._layer_init(nn.Linear(input_dim, input_dim)),
        )
        self.road_attention_embed = QueryAttentionPooling(
            input_dim,
            dropout=dropout,
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

    def _build_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return True for padding rows that should be ignored."""
        # Padding is encoded as an all-zero raw token row in the flattened obs.
        # Build this mask before the embedding MLP because the downstream
        # Linear/LayerNorm stack destroys that zero pattern. After embedding we
        # can no longer reliably distinguish padding rows from real tokens.
        return tokens.eq(0).all(dim=-1)

    def _encode_road_tokens(self, road_graph: torch.Tensor) -> torch.Tensor:
        """Encode road tokens with separate geometry and type branches."""
        road_geom = road_graph[..., : self.road_geom_feat_dim]
        road_type = road_graph[..., self.road_geom_feat_dim :]
        road_geom_tokens = self.road_geom_embed(road_geom)
        road_type_tokens = self.road_type_embed(road_type)
        return road_geom_tokens + road_type_tokens
    
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
        # Build masks from the raw token tensors, not from the embedded tokens.
        # Raw zero rows are the only stable representation of padding.
        partner_padding_mask = self._build_padding_mask(road_objects)
        road_padding_mask = self._build_padding_mask(road_graph)
        partner_tokens = self.partner_embed(road_objects)
        road_tokens = self._encode_road_tokens(road_graph)

        if self.student_partner_pooling == "attention":
            partner_embed = self.partner_attention_embed(
                ego_embed,
                partner_tokens,
                partner_padding_mask,
            )
        else:
            partner_embed, _ = partner_tokens.max(dim=1)

        if self.student_road_pooling == "attention":
            road_map_embed = self.road_attention_embed(
                ego_embed,
                road_tokens,
                road_padding_mask,
            )
        else:
            road_map_embed, _ = road_tokens.max(dim=1)
        
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
        student_partner_pooling: str = "attention",
        student_road_pooling: str = "attention",
        recurrent: bool = False,
        recurrent_arch: str = "lstm",
        recurrent_hidden_size: int = 256,
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
            student_partner_pooling=student_partner_pooling,
            student_road_pooling=student_road_pooling,
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
        self._recurrent = recurrent
        self.rnn = None
        if recurrent:
            self.rnn = RNN(
                input_size=hidden_dim,
                hidden_size=recurrent_hidden_size,
                arch=recurrent_arch,
            )

        actor_input_size = (
            self.rnn.output_size if self.rnn is not None else hidden_dim
        )
        self.dist = Categorical(actor_input_size, self.action_dim)
        
        # Critic: output state value
        self.critic = self._layer_init(
            nn.Linear(actor_input_size, 1), std=1.0
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
        if not self.is_recurrent:
            return 1
        return self.rnn.recurrent_hidden_state_size

    def _init_recurrent_state(self, batch_size: int, device: torch.device, dtype):
        """Create a zero recurrent state for the current RNN architecture."""
        zeros = torch.zeros(
            batch_size,
            self.recurrent_hidden_state_size,
            device=device,
            dtype=dtype,
        )
        if self.rnn.is_lstm:
            return zeros, torch.zeros_like(zeros)
        return zeros

    def _encode_actor_features(self, inputs, rnn_hxs=None, masks=None):
        """Encode observations and optionally run them through the RNN."""
        hidden = self.base.encode_observations(inputs)
        if not self.is_recurrent:
            return hidden, rnn_hxs

        if masks is None:
            masks = torch.ones(
                hidden.size(0),
                1,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        if rnn_hxs is None:
            rnn_hxs = self._init_recurrent_state(
                batch_size=hidden.size(0),
                device=hidden.device,
                dtype=hidden.dtype,
            )

        hidden, rnn_hxs = self.rnn(hidden, rnn_hxs, masks)
        return hidden, rnn_hxs
    
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
            rnn_hxs: Recurrent hidden states used in recurrent mode
            masks: Episode masks used to reset recurrent state
            deterministic: Whether to sample deterministically
        
        Returns:
            value: State value (batch_size, 1)
            action: Action ids (batch_size, 1)
            action_log_dist: Action logits (batch_size, action_dim)
            rnn_hxs: Updated hidden states
        """
        hidden, rnn_hxs = self._encode_actor_features(inputs, rnn_hxs, masks)
        
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
            rnn_hxs: Recurrent hidden states used in recurrent mode
            masks: Episode masks used to reset recurrent state
        
        Returns:
            value: State value  (batch_size, 1)
        """
        hidden, _ = self._encode_actor_features(inputs, rnn_hxs, masks)
        return self.critic(hidden)
    
    def evaluate_actions(
        self, inputs, rnn_hxs, masks, action, return_policy_logits=False
    ):
        """
        Evaluate actions (called during PPO update)
        
        Args:
            inputs: Observations (batch_size, obs_dim)
            rnn_hxs: Recurrent hidden states used in recurrent mode
            masks: Episode masks used to reset recurrent state
            action: Actions to evaluate (batch_size, action_dim)
            return_policy_logits: Whether to return the full distribution
        
        Returns:
            value: State value (batch_size, 1)
            action_log_probs: Action log probabilities (batch_size, 1)
            dist_entropy: Policy entropy (scalar)
            rnn_hxs: Updated hidden states
            [dist]: Optional, full distribution
        """
        hidden, rnn_hxs = self._encode_actor_features(inputs, rnn_hxs, masks)
        value = self.critic(hidden)
        
        dist = self.dist(hidden)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist

        return value, action_log_probs, dist_entropy, rnn_hxs


# =============================================================================
# CtRL-Sim aligned student: CtrlSimStudentEncoder + CtrlSimStudentPolicy
# =============================================================================

class _MLPLayer(nn.Module):
    """Linear → LayerNorm → ReLU → Linear  (matches ctrlsim/utils/layers.py)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CtrlSimStudentEncoder(nn.Module):
    """Scene encoder aligned with CtRL-Sim, with no RTG and no action tokens.

    Token stacking: one state+goal token per (agent, timestep).
    Map encoder: road polylines → encoder_embeddings (cross-attention memory).

    Args:
        num_agents:   ego + neighbours (e.g. 9).
        seq_len:      history length (e.g. 10).
        state_dim:    raw agent state features = 8 (px,py,vx,vy,hdg,len,wid,exist).
        agent_type_dim: one-hot type width = 5.
        goal_dim:     goal feature width = 5.
        max_roads:    road polyline count (e.g. 64).
        road_type_dim: road-type one-hot width = 8.
        hidden_dim:   transformer hidden size (e.g. 256).
        num_heads:    attention heads (e.g. 8).
        dim_feedforward: transformer FFN size (e.g. 1024).
        num_encoder_layers: TransformerEncoder depth (e.g. 2).
        max_timestep: timestep embedding table size (>= seq_len).
    """

    def __init__(
        self,
        num_agents: int,
        seq_len: int,
        state_dim: int = 8,
        agent_type_dim: int = 5,
        goal_dim: int = 5,
        max_roads: int = 64,
        road_type_dim: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 2,
        max_timestep: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.max_roads = max_roads

        # State embedding: drop existence flag → 7 features + agent_type_dim
        fused_state_dim = (state_dim - 1) + agent_type_dim  # e.g. 7 + 5 = 12
        self.embed_state = _MLPLayer(fused_state_dim, hidden_dim, hidden_dim)
        self.embed_goal = _MLPLayer(goal_dim, hidden_dim, hidden_dim)
        self.embed_state_goal = nn.Linear(hidden_dim * 2, hidden_dim)

        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_agent_id = nn.Embedding(num_agents, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Map encoder: attention-pool each polyline, then combine with road type.
        map_attr = 2  # x, y (valid flag in last dim is mask, not a feature)
        self.map_seeds = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.map_seeds)
        self.road_pts_encoder = _MLPLayer(map_attr, hidden_dim, hidden_dim)
        self.road_pts_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.road_norm1 = nn.LayerNorm(hidden_dim)
        self.road_norm2 = nn.LayerNorm(hidden_dim)
        self.road_feats = _MLPLayer(hidden_dim, hidden_dim, hidden_dim)
        self.road_type_encoder = _MLPLayer(road_type_dim, hidden_dim, hidden_dim)
        self.road_combine = _MLPLayer(hidden_dim * 2, hidden_dim, hidden_dim)

        # TransformerEncoder: encodes map polylines + initial agent states.
        # enable_nested_tensor=False prevents PyTorch from using a nested-tensor
        # optimization that changes the output seq-length when masks are present,
        # which would break the cross-attention memory shape in the decoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

    # ------------------------------------------------------------------
    # Map encoding
    # ------------------------------------------------------------------

    def _encode_map(
        self, road_points: torch.Tensor, road_types: torch.Tensor
    ):
        """Return (batch, max_roads, hidden) and (batch, max_roads) valid mask."""
        B, R, P, _ = road_points.shape

        # Valid flag is last dim of road_points.
        valid_flag = road_points[..., -1]           # (B, R, P)
        pts_xy = road_points[..., :2].float()       # (B, R, P, 2)

        # Per-point mask: True = padding (to match nn.MultiheadAttention convention).
        # Flatten to (B*R, P) first so the NaN-guard and attention share the same layout.
        road_pts_mask = (valid_flag < 0.5).view(B * R, P)   # True = invalid

        # Road-level valid mask MUST be computed before the NaN-guard below.
        # The guard forces point-0 to be "unmasked" on fully-empty rows so attention
        # does not produce NaN, but that must not make an empty polyline appear valid.
        road_valid = ~road_pts_mask.all(dim=-1).view(B, R)  # (B, R)  True = has valid pts

        # NaN-guard: ensure every row has at least one unmasked key so attention
        # never sees an all-True key_padding_mask (which produces NaN softmax).
        all_invalid_rows = road_pts_mask.all(dim=-1)         # (B*R,)
        road_pts_mask[all_invalid_rows, 0] = False

        # Encode points: (B, R, P, 2) → (B*R, P, hidden).
        pts_enc = self.road_pts_encoder(pts_xy).view(B * R, P, self.hidden_dim)
        pts_enc = pts_enc.permute(1, 0, 2)          # (P, B*R, hidden)

        seeds = self.map_seeds.repeat(1, B * R, 1)  # (1, B*R, hidden)
        kpm = road_pts_mask                          # (B*R, P)

        seg_emb = self.road_pts_attn(
            query=seeds, key=pts_enc, value=pts_enc, key_padding_mask=kpm
        )[0]                                         # (1, B*R, hidden)
        seg_emb = self.road_norm1(seg_emb)
        seg_emb = seg_emb + self.road_feats(seg_emb)
        seg_emb = self.road_norm2(seg_emb)           # (1, B*R, hidden)

        # Road type embedding.
        rt_emb = self.road_type_encoder(road_types.float())  # (B, R, hidden)
        rt_emb = rt_emb.view(1, B * R, self.hidden_dim)

        # Combine.
        combined = self.road_combine(
            torch.cat([seg_emb, rt_emb], dim=-1)
        )                                            # (1, B*R, hidden)
        road_emb = combined.view(B, R, self.hidden_dim)
        return road_emb, road_valid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data, eval: bool = True):
        """Encode the scene and return embeddings for the decoder.

        Args:
            data: object with .agent and .map attributes (from unpack_obs).
            eval:  passed through for consistency with CtRL-Sim API.

        Returns dict with keys:
            stacked_embeddings  : (B, seq_len*num_agents, hidden)
            encoder_embeddings  : (B, max_roads + num_agents, hidden)
            src_key_padding_mask: (B, max_roads + num_agents)  True = ignore
            existence_mask_flat : (B, seq_len*num_agents, 1)
        """
        agent_states = data.agent.agent_states.float()  # (B, A, T, 8)
        agent_types = data.agent.agent_types.float()    # (B, A, 5)
        goals = data.agent.goals.float()                # (B, A, 5)
        road_points = data.map.road_points.float()      # (B, R, P, 3)
        road_types = data.map.road_types.float()        # (B, R, D)

        B, A, T, _ = agent_states.shape

        # Existence mask from last dim of agent_states.
        existence_mask = agent_states[:, :, :, -1:]    # (B, A, T, 1)
        # Drop existence flag; cat with agent_type.
        states_no_exist = agent_states[:, :, :, :-1]  # (B, A, T, 7)
        # Tile agent_types over timesteps: (B, A, T, type_dim)
        types_tiled = agent_types.unsqueeze(2).expand(B, A, T, -1)
        # Combined state: (B, A, T, 12) → transpose → (B, T, A, 12)
        states_full = torch.cat([states_no_exist, types_tiled], dim=-1).transpose(1, 2)
        # Goals tiled: (B, T, A, 5)
        goals_tiled = goals.unsqueeze(1).expand(B, T, A, -1)

        # Flatten time × agents: (B, T*A, 12) and (B, T*A, 5)
        TA = T * A
        states_flat = states_full.reshape(B, TA, -1)
        goals_flat = goals_tiled.reshape(B, TA, -1)

        # Embeddings.
        state_emb = self.embed_state(states_flat)           # (B, T*A, H)
        goal_emb = self.embed_goal(goals_flat)              # (B, T*A, H)
        state_goal_emb = self.embed_state_goal(
            torch.cat([state_emb, goal_emb], dim=-1)
        )                                                    # (B, T*A, H)

        # Positional embeddings.
        t_idx = torch.arange(T, device=agent_states.device)
        a_idx = torch.arange(A, device=agent_states.device)
        # Broadcast: timestep repeats per agent, agent_id repeats per timestep.
        t_idx_flat = t_idx.unsqueeze(1).expand(T, A).reshape(TA)  # (T*A,)
        a_idx_flat = a_idx.unsqueeze(0).expand(T, A).reshape(TA)  # (T*A,)
        t_emb = self.embed_timestep(t_idx_flat).unsqueeze(0)  # (1, T*A, H)
        a_emb = self.embed_agent_id(a_idx_flat).unsqueeze(0)  # (1, T*A, H)

        state_tokens = state_goal_emb + t_emb + a_emb       # (B, T*A, H)

        # Apply existence mask (zero out padded tokens).
        exist_flat = existence_mask.transpose(1, 2).reshape(B, TA, 1)  # (B, T*A, 1)
        state_tokens = state_tokens * exist_flat.float()
        stacked_embeddings = self.embed_ln(state_tokens)     # (B, T*A, H)

        # Map encoding.
        road_emb, road_valid = self._encode_map(road_points, road_types)

        # Initial-state tokens (t=0 slice): (B, A, H)
        initial_tokens = stacked_embeddings[:, :A, :]
        initial_exist = exist_flat[:, :A, 0].bool()          # (B, A)

        # Concatenate for TransformerEncoder: (B, R+A, H)
        pre_enc = torch.cat([road_emb, initial_tokens], dim=1)
        road_invalid = ~road_valid                            # (B, R)  True=mask
        initial_invalid = ~initial_exist                      # (B, A)  True=mask
        src_key_padding_mask = torch.cat(
            [road_invalid, initial_invalid], dim=1
        )                                                     # (B, R+A)

        encoder_embeddings = self.transformer_encoder(
            pre_enc, src_key_padding_mask=src_key_padding_mask
        )                                                     # (B, R+A, H)

        return {
            "stacked_embeddings": stacked_embeddings,
            "encoder_embeddings": encoder_embeddings,
            "src_key_padding_mask": src_key_padding_mask,
            "existence_mask_flat": exist_flat,
        }


class CtrlSimStudentPolicy(nn.Module):
    """PPO-compatible student policy using CtRL-Sim state encoder architecture.

    Observation: flat float32 vector produced by CtrlSimObsBuilder.
    Action:      discrete index into (accel_bins × steer_bins).
    Value:       scalar from ego agent's last-step decoded token.

    act() / get_value() / evaluate_actions() match the interface expected by
    adversarial_runner.py and the PPO / storage implementations.
    """

    # Required by ppo.py and agent.py — this policy is feed-forward, not recurrent.
    is_recurrent = False

    @property
    def recurrent_hidden_state_size(self):
        """Match the common policy interface expected by rollout/eval code."""
        return 1

    def __init__(
        self,
        obs_cfg,                 # CtrlSimObsConfig
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        max_timestep: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        from envs.nocturne_ctrlsim.student.ctrlsim_observation import unpack_obs
        self._obs_cfg = obs_cfg
        self._unpack_obs_fn = unpack_obs
        self.hidden_dim = hidden_dim
        self.num_agents = obs_cfg.num_agents
        self.seq_len = obs_cfg.seq_len
        self.action_dim = action_dim

        self.encoder = CtrlSimStudentEncoder(
            num_agents=obs_cfg.num_agents,
            seq_len=obs_cfg.seq_len,
            state_dim=obs_cfg.state_dim,
            agent_type_dim=obs_cfg.agent_type_dim,
            goal_dim=obs_cfg.goal_dim,
            max_roads=obs_cfg.max_roads,
            road_type_dim=obs_cfg.road_type_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            max_timestep=max_timestep,
            dropout=dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Causal mask: each position attends only to positions at earlier or
        # equal timesteps (across all agents at the same timestep).
        # Shape: (seq_len*num_agents, seq_len*num_agents)
        T, A = obs_cfg.seq_len, obs_cfg.num_agents
        causal = self._build_causal_mask(T, A)
        self.register_buffer("causal_mask", causal)

        self.action_head = _MLPLayer(hidden_dim, hidden_dim, action_dim)
        self.value_head = _MLPLayer(hidden_dim, hidden_dim, 1)

    @staticmethod
    def _build_causal_mask(T: int, A: int) -> torch.Tensor:
        """Block-lower-triangular causal mask over (T*A) tokens.

        Token order: [t0_a0, t0_a1, ..., t0_aN, t1_a0, ..., tT_aN].
        Position (t*A+a) can attend to all positions (t'*A+a') where t' <= t.
        """
        TA = T * A
        mask = torch.zeros(TA, TA)
        for i in range(TA):
            t_i = i // A
            for j in range(TA):
                t_j = j // A
                if t_j > t_i:
                    mask[i, j] = float("-inf")
        return mask

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + decoder and return `(ego_logits, value)`."""
        data = self._unpack_obs_fn(obs, self._obs_cfg)
        scene_enc = self.encoder(data, eval=True)

        stacked_emb = scene_enc["stacked_embeddings"]      # (B, T*A, H)
        encoder_emb = scene_enc["encoder_embeddings"]      # (B, R+A, H)
        src_mask = scene_enc["src_key_padding_mask"]       # (B, R+A)

        decoded = self.transformer_decoder(
            stacked_emb,
            encoder_emb,
            tgt_mask=self.causal_mask,
            memory_key_padding_mask=src_mask,
        )                                                   # (B, T*A, H)

        # Ego agent (index 0) at last timestep.
        # Token index: (T-1)*A + 0
        ego_idx = (self.seq_len - 1) * self.num_agents     # = (T-1)*A
        ego_token = decoded[:, ego_idx, :]                  # (B, H)

        ego_logits = self.action_head(ego_token)            # (B, action_dim)
        value = self.value_head(ego_token)                  # (B, 1)
        return ego_logits, value

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def act(self, obs, rnn_hxs=None, masks=None, deterministic: bool = False):
        """Sample an action.

        Returns:
            value         : (batch, 1)
            action        : (batch, 1)
            action_log_dist: (batch, action_dim)  full log-prob vector
            rnn_hxs       : unchanged (this policy is not recurrent)
        """
        ego_logits, value = self._forward(obs)
        log_dist = F.log_softmax(ego_logits, dim=-1)

        if deterministic:
            action = ego_logits.argmax(dim=-1, keepdim=True)
        else:
            action = torch.distributions.Categorical(logits=ego_logits).sample()
            action = action.unsqueeze(-1)

        return value, action, log_dist, rnn_hxs

    def get_value(self, obs, rnn_hxs=None, masks=None):
        """Return the state value.

        Returns:
            value: (batch, 1)
        """
        _, value = self._forward(obs)
        return value

    def evaluate_actions(self, obs, rnn_hxs, masks, action, return_policy_logits: bool = False):
        """Evaluate log-prob, value, and entropy of given actions (PPO update).

        Args:
            return_policy_logits: when True return a 5th element — a
                ``torch.distributions.Categorical`` whose ``.logits`` attribute
                is the raw action logits. Required by PPO when KL loss is active.

        Returns (4-tuple, or 5-tuple when return_policy_logits=True):
            value           : (batch, 1)
            action_log_probs: (batch, 1)
            dist_entropy    : scalar
            rnn_hxs         : unchanged
            [dist]          : Categorical distribution (only if return_policy_logits)
        """
        ego_logits, value = self._forward(obs)
        dist = torch.distributions.Categorical(logits=ego_logits)
        log_dist = F.log_softmax(ego_logits, dim=-1)
        action_log_probs = log_dist.gather(-1, action.long())
        dist_entropy = dist.entropy().mean()
        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        return value, action_log_probs, dist_entropy, rnn_hxs
