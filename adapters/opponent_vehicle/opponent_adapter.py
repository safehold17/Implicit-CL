"""
CtRL-Sim 对手策略适配器

复用 ctrl-sim 的 AutoregressivePolicy，适配 DCD 环境的调用模式。

- evaluators/policy_evaluator.py 第 427-560 行的评估循环
- policies/autoregressive_policy.py 核心推理逻辑
- policies/policy.py Policy 基类
"""
# 必须首先设置路径，在任何其他导入之前
import sys as _sys
from pathlib import Path as _Path

_CTRLSIM_PATH = _Path(__file__).resolve().parents[2] / "ctrlsim"
_CTRLSIM_PATH_STR = str(_CTRLSIM_PATH)
if _CTRLSIM_PATH_STR not in _sys.path:
    _sys.path.insert(0, _CTRLSIM_PATH_STR)

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from policies.autoregressive_policy import AutoregressivePolicy
from utils.sim import compute_reward

from .existence_logic import (
    _compute_goal_hold_until,
    _keep_exists_on_invalid,
    _should_drop_after_goal,
)
from .opponent_policy import OpponentPolicyService
from .opponent_reward import OpponentRewardService
from .opponent_state import OpponentStateService
from .sparse_inference import SparseInferenceConfig, SparseInferenceController
from .tilting import TiltConfig


class CtrlSimOpponentAdapter:
    """
    适配器：将 ctrl-sim AutoregressivePolicy 封装为 DCD 可调用的对手策略

    关键设计：
    1. 保持与 PolicyEvaluator.evaluate_policy() 相同的数据流
    2. 支持动态设置 tilting 参数
    3. 复用 AutoregressivePolicy 的完整推理逻辑

    使用示例:
    ```python
    adapter = CtrlSimOpponentAdapter(cfg, checkpoint_path)
    adapter.set_tilting(goal_tilt=10, veh_veh_tilt=-5, veh_edge_tilt=0)
    adapter.reset(scenario, vehicles, gt_data_dict, preproc_data, opponent_ids)

    for t in range(max_steps):
        actions = adapter.step(t, vehicles)
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
        opponent_sparse_inference_enabled: bool = False,
        opponent_sparse_inference_interval: int = 2,
        sparse_inference_action_repeat: bool = False,
        load_on_init: bool = True,
    ):
        """
        Args:
            cfg: Hydra 配置对象（需包含 nocturne, dataset.waymo, model 等配置）
            checkpoint_path: ctrl-sim 模型 checkpoint 路径
            device: 推理设备
            action_temperature: 动作采样温度（参考: cfgs/policy/ctrl_sim.yaml）
            nucleus_sampling: 是否使用 nucleus sampling
            nucleus_threshold: nucleus sampling 阈值
            opponent_sparse_inference_enabled: 是否启用稀疏推理节奏
            opponent_sparse_inference_interval: 稀疏推理周期 N（每 N 步中最后一步为稀疏步）
            sparse_inference_action_repeat: 稀疏步是否直接复用上一仿真步动作
            load_on_init: 是否在初始化时立即加载模型/数据集
        """
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.dataset = None
        self.batch_inference = False
        self._checkpoint_cfg = None
        self._policy_service = OpponentPolicyService(self)
        self._reward_service = OpponentRewardService(self)
        self._state_service = OpponentStateService(self)

        # 策略配置
        self.action_temperature = action_temperature
        self.nucleus_sampling = nucleus_sampling
        self.nucleus_threshold = nucleus_threshold
        self.sparse_inference_cfg = SparseInferenceConfig(
            enabled=bool(opponent_sparse_inference_enabled),
            interval=int(opponent_sparse_inference_interval),
        )
        self.sparse_inference = SparseInferenceController(self.sparse_inference_cfg)
        self.sparse_inference_action_repeat = bool(sparse_inference_action_repeat)

        # 当前 tilting 配置
        self.current_tilt = TiltConfig()

        # Per-vehicle tilting mapping: {veh_id: (goal_tilt, veh_veh_tilt, veh_edge_tilt)}
        self.per_vehicle_tilting: Optional[Dict[int, Tuple[int, int, int]]] = None

        # 内部策略实例（在 reset 时创建）
        self._policy: Optional[AutoregressivePolicy] = None

        # 运行时状态
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
        self._goal_pos_tolerance: float = 1.0
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
        # Whether to move non-controlled vehicles out of the scene when GT is missing.
        self.allow_set_position_for_noncontrolled: bool = False
        # 缓存 vehicles 列表，供 apply_predictions warm-up 使用
        self._last_vehicles: Optional[List] = None
        self._last_vehicle_by_id: Dict[int, Any] = {}

        # 从配置中获取时间相关参数
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.history_steps = getattr(cfg.nocturne, "history_steps", 10)
        self._default_initial_rtg = np.array([10.0, 90.0, 90.0], dtype=np.float32)

        if load_on_init:
            self._load_model_and_dataset()

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
        worker_rng_state: Optional[np.ndarray] = None,
    ):
        self._ensure_services()
        return self._state_service.prepare_step(t, vehicles, worker_rng_state=worker_rng_state)

    def update_policy_state(self, t: int):
        self._ensure_services()
        return self._state_service.update_policy_state(t)

    def apply_predictions(self, model_outputs: Optional[Dict]):
        self._ensure_services()
        return self._state_service.apply_predictions(model_outputs)

    def step(self, t: int, vehicles: List):
        self._ensure_services()
        return self._state_service.step(t, vehicles)

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
        controlled_actions: Dict[int, Tuple[float, float]],
    ):
        self._ensure_services()
        return self._state_service.record_all_actions(
            t=t,
            vehicles=vehicles,
            controlled_actions=controlled_actions,
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
    "_compute_goal_hold_until",
    "_should_drop_after_goal",
    "_keep_exists_on_invalid",
]
