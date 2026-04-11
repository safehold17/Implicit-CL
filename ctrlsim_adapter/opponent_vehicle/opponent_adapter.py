"""
负责把 ctrl-sim 的 AutoregressivePolicy 封装成 DCD 环境可直接调用的对手策略适配器。
该模块统一管理配置、模型、状态服务、inference bridge 与 tilting，并复用原始 ctrl-sim 推理语义。
Wraps ctrl-sim's AutoregressivePolicy into an opponent-policy adapter that can be called directly from DCD.
Centrally manages config, model, state service, inference bridge, and tilting while preserving ctrl-sim inference semantics.
"""
from ctrlsim_adapter.ctrlsim_path import ctrlsim_path

ctrlsim_path()

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from policies.autoregressive_policy import AutoregressivePolicy
from . import opponent_inference_io as _batch_io
from .services.policy_service import OpponentPolicyService
from .services.reward_service import OpponentRewardService
from .services.state_service import OpponentStateService
from .sparse_inference import SparseInferenceConfig, SparseInferenceController
from .tilting import TiltConfig
from ctrlsim_adapter.policy_reweighting_helpers import AdversarialRTGConfig


class CtrlSimOpponentAdapter:
    """
    适配器：将 ctrl-sim AutoregressivePolicy 封装为 DCD 可调用的对手策略
    Adapter that wraps ctrl-sim AutoregressivePolicy as a DCD-callable opponent policy.

    关键设计：
    Key design points:
    1. 保持与 PolicyEvaluator.evaluate_policy() 相同的数据流
    1. Keep the same data flow as PolicyEvaluator.evaluate_policy().
    2. 支持动态设置 tilting 参数
    2. Support dynamically updating tilting parameters.
    3. 复用 AutoregressivePolicy 的完整推理逻辑
    3. Reuse the complete inference logic of AutoregressivePolicy.

    使用示例:
    Example usage:
    ```python
    adapter = CtrlSimOpponentAdapter(cfg, checkpoint_path)
    adapter.set_tilting(goal_tilt=10, veh_veh_tilt=-5, veh_edge_tilt=0)
    adapter.reset(scenario, vehicles, gt_data_dict, preproc_data, opponent_ids)

    for t in range(max_steps):
    prepared = adapter.prepare_step(t, vehicles)
    actions = adapter.apply_predictions(model_outputs)
    for veh_id, (accel, steer) in actions.items():
    adapter.apply_action(vehicle_map[veh_id], (accel, steer))
    sim.step(dt)
    ```
    """

    def __init__(
        self,
        cfg: Any,
        checkpoint_path: str,
        device: str = "cuda",
        action_temperature: float = 1.0,
        nucleus_sampling: bool = False,
        nucleus_threshold: float = 0.8,
        action_repeat_frequency: int = 2,
        kl_loss_computation_frequency: int = 2,
        sparse_inference_action_repeat: bool = False,
        use_ego_ctrlsim_kl_loss: bool = False,
        enhanced_regret: bool = False,
        use_policy_reweighting: bool = False,
        policy_reweighting_reward_scale: float = 1.0,
        policy_reweighting_epsilon: float = 1e-6,
        policy_reweighting_target: str = "rtg",
        reweighting_frequency: int = 1,
        load_on_init: bool = True,
    ):
        """
        Args:
        cfg: Hydra 配置对象（需包含 nocturne, dataset.waymo, model 等配置）
        cfg: Hydra config object that must include nocturne, dataset.waymo, model, and related settings.
        checkpoint_path: ctrl-sim 模型 checkpoint 路径
        checkpoint_path: path to the ctrl-sim model checkpoint.
        device: 推理设备
        device: inference device.
        action_temperature: 动作采样温度（参考: cfgs/policy/ctrl_sim.yaml）
        action_temperature: action sampling temperature; see cfgs/policy/ctrl_sim.yaml.
        nucleus_sampling: 是否使用 nucleus sampling
        nucleus_sampling: whether to use nucleus sampling.
        nucleus_threshold: nucleus sampling 阈值
        nucleus_threshold: threshold for nucleus sampling.
        action_repeat_frequency: 对手推理动作复用周期 N
        action_repeat_frequency: action-repeat cycle length N for opponent inference.
        kl_loss_computation_frequency: ego KL loss 采样周期 N
        kl_loss_computation_frequency: cycle length N for ego KL-loss collection.
        sparse_inference_action_repeat: 是否启用动作复用节奏
        sparse_inference_action_repeat: whether to enable the action-reuse cadence.
        use_ego_ctrlsim_kl_loss: whether to collect ego teacher logits for KL.
        enhanced_regret: whether to export ego teacher RTGs for regret enhancement.
        use_policy_reweighting: 是否启用 opponent policy delayed reweighting
        use_policy_reweighting: whether to enable delayed opponent policy reweighting.
        policy_reweighting_reward_scale: policy reweighting scalar multiplier.
        policy_reweighting_epsilon: numerical-stability term for policy reweighting.
        policy_reweighting_target: delayed reweighting target, either rtg or action.
        reweighting_frequency: cadence for active policy reweighting steps.
        load_on_init: 是否在初始化时立即加载模型/数据集
        load_on_init: whether to load the model and dataset during initialization.
        """
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.dataset = None
        self._checkpoint_cfg = None
        self._policy_service = OpponentPolicyService(self)
        self._reward_service = OpponentRewardService(self)
        self._state_service = OpponentStateService(self)

        # 策略配置
        # Policy config.
        self.action_temperature = action_temperature
        self.nucleus_sampling = nucleus_sampling
        self.nucleus_threshold = nucleus_threshold
        self.action_repeat_frequency = int(action_repeat_frequency)
        self.kl_loss_computation_frequency = int(kl_loss_computation_frequency)
        self.sparse_inference_action_repeat = bool(sparse_inference_action_repeat)
        self.use_ego_ctrlsim_kl_loss = bool(use_ego_ctrlsim_kl_loss)
        self.enhanced_regret = bool(enhanced_regret)
        self.use_policy_reweighting = bool(use_policy_reweighting)
        self.policy_reweighting_target = str(policy_reweighting_target)
        self.reweighting_frequency = int(reweighting_frequency)
        self.policy_reweighting_config = AdversarialRTGConfig(
            enabled=self.use_policy_reweighting,
            reward_scale=float(policy_reweighting_reward_scale),
            epsilon=float(policy_reweighting_epsilon),
        )
        self.sparse_inference_cfg = SparseInferenceConfig(
            enabled=self.sparse_inference_action_repeat,
            interval=self.action_repeat_frequency,
        )
        self.sparse_inference = SparseInferenceController(self.sparse_inference_cfg)

        # 当前 tilting 配置
        # Current tilting config.
        self.current_tilt = TiltConfig()

        # Per-vehicle tilting mapping: {veh_id: (goal_tilt, veh_veh_tilt, veh_edge_tilt)}
        self.per_vehicle_tilting: Optional[Dict[int, Tuple[int, int, int]]] = None

        # 内部策略实例（在 reset 时创建）
        # Internal policy instance, created during reset.
        self._policy: Optional[AutoregressivePolicy] = None

        # 运行时状态
        # Runtime state.
        self._vehicle_data_dict: Dict = {}
        self._gt_data_dict: Dict = {}
        self._gt_traj_by_id: Dict[int, np.ndarray] = {}
        self._gt_action_target_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._gt_action_runtime_cache: Dict[Tuple[Any, ...], Tuple[float, float]] = {}
        self._preproc_data: Dict = {}
        self._vehicles_to_control: List[int] = []
        self._vehicles_to_control_sorted: List[int] = []
        self._vehicles_to_control_set: set[int] = set()
        self._road_edge_polylines: List = []
        self._road_edge_polylines_cpu: Tuple[np.ndarray, ...] = ()
        self._constant_road_edge_reward_by_id: Dict[int, float] = {}
        self._goal_dict: Dict = {}
        self._goal_dist_normalizer: Dict = {}
        self._ego_id: Optional[int] = None
        self._veh_id_to_preproc_idx: Optional[Dict[int, int]] = None
        self._opponent_vehicle_exits: Dict[int, bool] = {}
        self._opponent_last_valid_pos: Dict[int, Tuple[float, float]] = {}
        self._opponent_goal_hold_until: Dict[int, Optional[int]] = {}
        self._goal_pos_tolerance: float = 2.0
        self._goal_hold_steps: int = 5
        self._all_vehicle_ids: List[int] = []
        self._controlled_vehicle_ids_present: List[int] = []
        self._controlled_vehicle_ids_step: List[int] = []
        self._moving_agent_mask_cache: Optional[np.ndarray] = None
        self._batch_prepare_cache: Dict[str, np.ndarray] = {}
        self._state_update_vehicle_ids_step: List[int] = []
        self._zero_reward_template: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._step_vehicle_by_id: Dict[int, Any] = {}
        self._vehicles_by_id_step: Dict[int, Any] = {}
        self._pending_sparse_actions_step_t: Optional[int] = None
        self._pending_sparse_actions: Dict[int, Tuple[float, float]] = {}
        self._ego_action_scale: float = 1.0
        self._ego_reweight_tilt: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Whether to move non-controlled vehicles out of the scene when GT is missing.
        self.allow_set_position_for_noncontrolled: bool = False
        # 缓存 vehicles 列表，供 apply_predictions warm-up 使用
        # Cache the vehicles list for apply_predictions warm-up.
        self._last_vehicles: Optional[List] = None
        self._last_vehicle_by_id: Dict[int, Any] = {}

        # 从配置中获取时间相关参数
        # Read time-related parameters from the config.
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.history_steps = getattr(cfg.nocturne, "history_steps", 10)
        self._default_initial_rtg = np.array([10.0, 90.0, 90.0], dtype=np.float32)

        if load_on_init:
            self._load_checkpoint_cfg()
            self._validate_external_cfg_compatibility()
            self._load_dataset_only()

    # ===== policy service delegation =====

    def _ensure_services(self):
        policy_service = getattr(self, "_policy_service", None)
        reward_service = getattr(self, "_reward_service", None)
        state_service = getattr(self, "_state_service", None)
        if policy_service is not None and reward_service is not None and state_service is not None:
            return

        if policy_service is None:
            self._policy_service = OpponentPolicyService(self)
        if reward_service is None:
            self._reward_service = OpponentRewardService(self)
        if state_service is None:
            self._state_service = OpponentStateService(self)

    def _load_checkpoint_cfg(self):
        self._ensure_services()
        return self._policy_service._load_checkpoint_cfg()

    def _validate_external_cfg_compatibility(self):
        self._ensure_services()
        return self._policy_service._validate_external_cfg_compatibility()

    def _load_dataset_only(self):
        self._ensure_services()
        return self._policy_service._load_dataset_only()

    def _load_model_and_dataset(self):
        self._ensure_services()
        return self._policy_service._load_model_and_dataset()

    def _create_policy(self):
        self._ensure_services()
        return self._policy_service._create_policy()

    def set_tilting(self, goal_tilt: int, veh_veh_tilt: int, veh_edge_tilt: int):
        self._ensure_services()
        return self._policy_service.set_tilting(goal_tilt, veh_veh_tilt, veh_edge_tilt)

    def set_tilting_from_tuple(self, tilt: Tuple[int, int, int]):
        self._ensure_services()
        return self._policy_service.set_tilting_from_tuple(tilt)

    def set_per_vehicle_tilting(self, mapping: Dict[int, Tuple[int, int, int]]):
        self._ensure_services()
        return self._policy_service.set_per_vehicle_tilting(mapping)

    # ===== reward service delegation =====

    def _compute_nearest_dist_all(self, t: int, vehicle_data_dict: Dict):
        self._ensure_services()
        return self._reward_service._compute_nearest_dist_all(t, vehicle_data_dict)

    def _compute_dense_reward(self, t: int, vehicle_data_dict: Dict):
        self._ensure_services()
        return self._reward_service._compute_dense_reward(t, vehicle_data_dict)

    def _compute_nearest_dist_to_all(
        self,
        target_positions: np.ndarray,
        all_positions: np.ndarray,
        all_existence: np.ndarray,
        target_existence: np.ndarray,
        target_all_indices: np.ndarray,
    ):
        self._ensure_services()
        return self._reward_service._compute_nearest_dist_to_all(
            target_positions=target_positions,
            all_positions=all_positions,
            all_existence=all_existence,
            target_existence=target_existence,
            target_all_indices=target_all_indices,
        )

    # ===== runtime state service delegation =====

    def reset(
        self,
        scenario,
        vehicles: List,
        gt_data_dict: Dict,
        preproc_data: Dict,
        vehicles_to_control: List[int],
        ego_id: Optional[int] = None,
    ):
        self._ensure_services()
        return self._state_service.reset(
            scenario,
            vehicles,
            gt_data_dict,
            preproc_data,
            vehicles_to_control,
            ego_id=ego_id,
        )

    def cache_last_valid_positions(self, vehicles: List):
        self._ensure_services()
        return self._state_service.cache_last_valid_positions(vehicles)

    def get_opponent_vehicle_exists(self, veh_id: int):
        self._ensure_services()
        return self._state_service.get_opponent_vehicle_exists(veh_id)

    def post_step_fix_opponent_positions(
        self,
        vehicles: List,
        goal_points_by_id: Optional[Dict[int, np.ndarray]],
        current_step: int,
    ):
        self._ensure_services()
        return self._state_service.post_step_fix_opponent_positions(
            vehicles=vehicles,
            goal_points_by_id=goal_points_by_id,
            current_step=current_step,
        )

    def prepare_step(
        self,
        t: int,
        vehicles: List,
    ):
        self._ensure_services()
        return self._state_service.prepare_step(t, vehicles)

    def prepare_step_pack(
        self,
        t: int,
        vehicles: List,
        ego_id: Optional[int] = None,
        include_ego_ctrlsim_prepared: bool = True,
    ):
        """代理双路 prepared pack 构建。 / Delegate prepared-pack construction."""
        self._ensure_services()
        return _batch_io.prepare_step_pack(
            self,
            t,
            vehicles,
            ego_id=ego_id,
            include_ego_ctrlsim_prepared=include_ego_ctrlsim_prepared,
        )

    def update_policy_state(self, t: int):
        self._ensure_services()
        return self._state_service.update_policy_state(t)

    def apply_predictions(self, model_outputs: Optional[Dict]):
        self._ensure_services()
        return self._state_service.apply_predictions(model_outputs)

    def reset_ego_action_scale(self) -> None:
        """把缓存的 ego_action_scale 收口到 unity。 / Reset the cached ego_action_scale back to unity.

        这是 runtime 层使用的轻量 facade，用于避免外层直接读写 adapter 内部字段。

        This is the lightweight facade used by runtime so outer layers do not
        directly read or write the adapter's internal scale field.
        """
        self._ensure_services()
        _batch_io.reset_ego_action_scale(self)

    def apply_action(self, veh, action: Tuple[float, float]):
        self._ensure_services()
        return self._state_service.apply_action(veh, action)

    def record_action(self, veh_id: int, action: Tuple[float, float]):
        self._ensure_services()
        return self._state_service.record_action(veh_id, action)

    def record_all_actions(
        self,
        t: int,
        vehicles: List,
        applied_actions: Dict[int, Tuple[float, float]],
    ):
        self._ensure_services()
        return self._state_service.record_all_actions(
            t=t,
            vehicles=vehicles,
            applied_actions=applied_actions,
        )

    def _get_gt_action(self, veh_id: int, t: int, veh=None):
        self._ensure_services()
        return self._state_service._get_gt_action(veh_id, t, veh=veh)

    def _get_gt_traj_data(self, veh_id: int):
        self._ensure_services()
        return self._state_service._get_gt_traj_data(veh_id)

    def _extract_road_edge_polylines(self, road_data: List[Dict]):
        self._ensure_services()
        return self._state_service._extract_road_edge_polylines(road_data)

    def _initialize_goal_dict(self, veh, gt_traj_data: np.ndarray):
        self._ensure_services()
        return self._state_service._initialize_goal_dict(veh, gt_traj_data)

    def _initialize_vehicle_data_dict(self, veh, goal_dict: Dict):
        self._ensure_services()
        return self._state_service._initialize_vehicle_data_dict(veh, goal_dict)

    def _compute_goal_dist_normalizer(self, veh, goal_pos: np.ndarray):
        self._ensure_services()
        return self._state_service._compute_goal_dist_normalizer(veh, goal_pos)

    def _get_initial_rtg(self, veh_id: int, veh_idx: int, t: int):
        self._ensure_services()
        return self._state_service._get_initial_rtg(veh_id, veh_idx, t)

    def _update_vehicle_data_dict(
        self,
        t: int,
        vehicles: List,
        vehicle_data_dict: Dict,
    ):
        self._ensure_services()
        return self._state_service._update_vehicle_data_dict(
            t=t,
            vehicles=vehicles,
            vehicle_data_dict=vehicle_data_dict,
        )

    def finalize(self, vehicles: List):
        self._ensure_services()
        return self._state_service.finalize(vehicles)

    @property
    def is_initialized(self) -> bool:
        self._ensure_services()
        return self._state_service.is_initialized

    def get_vehicle_data(self, veh_id: int):
        self._ensure_services()
        return self._state_service.get_vehicle_data(veh_id)


__all__ = [
    "CtrlSimOpponentAdapter",
    "TiltConfig",
]
