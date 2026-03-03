"""
CtRL-Sim 对手策略适配器  

复用 ctrl-sim 的 AutoregressivePolicy，适配 DCD 环境的调用模式。

- evaluators/policy_evaluator.py 第 427-560 行的评估循环
- policies/autoregressive_policy.py 核心推理逻辑
- policies/policy.py Policy 基类
"""
# 必须首先设置路径，在任何其他导入之前
import os as _os
import sys as _sys
_CTRLSIM_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), 'third_party', 'ctrl-sim')
if _CTRLSIM_PATH not in _sys.path:
    _sys.path.insert(0, _CTRLSIM_PATH)

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from models.ctrl_sim import CtRLSim
from policies.autoregressive_policy import AutoregressivePolicy
from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from utils.sim import get_road_data, get_moving_vehicles, compute_reward
from utils.data import get_object_type_str, add_batch_dim, from_numpy
from tools.safe_bicycle import safe_backward_action_from_states
from .existence import sim_position_exists

from batch_inference.discretization_utils import (
    get_tilt_logits as _get_tilt_logits,
    undiscretize_rtgs as _undiscretize_rtgs,
    decode_predicted_rtg as _decode_predicted_rtg,
)
from batch_inference import prepare_and_apply as _batch_io


def _compute_goal_hold_until(
    prev_hold_until: Optional[int],
    current_step: int,
    reached_goal: bool,
    hold_steps: int,
) -> Optional[int]:
    if prev_hold_until is None and reached_goal:
        return current_step + hold_steps
    return prev_hold_until


def _should_drop_after_goal(current_step: int, hold_until: Optional[int]) -> bool:
    return hold_until is not None and current_step >= hold_until


def _keep_exists_on_invalid(sim_exists: bool, prev_exists: bool) -> bool:
    if sim_exists:
        return True
    return bool(prev_exists)


@dataclass
class TiltConfig:
    """
    Domain tilting 配置
    
    tilting 通过修改 RTG 预测的 logits 来实现
    参考: datasets/rl_waymo/dataset.py 第 347-352 行 get_tilt_logits()
    
    参数范围: [-25, 25]
    - 正值: 更激进的行为
    - 负值: 更保守的行为
    """
    goal_tilt: int = 0       # 目标导向程度
    veh_veh_tilt: int = 0    # 车-车交互激进度
    veh_edge_tilt: int = 0   # 车-边界交互激进度
    
    def __post_init__(self):
        """验证参数范围"""
        for name, val in [
            ('goal_tilt', self.goal_tilt), 
            ('veh_veh_tilt', self.veh_veh_tilt),
            ('veh_edge_tilt', self.veh_edge_tilt)
        ]:
            if not (-25 <= val <= 25):
                raise ValueError(f"{name} must be in [-25, 25], got {val}")
    
    def to_dict(self) -> Dict:
        """转换为 ctrl-sim 期望的 tilt_dict 格式"""
        return {
            'tilt': True,
            'goal_tilt': self.goal_tilt,
            'veh_veh_tilt': self.veh_veh_tilt,
            'veh_edge_tilt': self.veh_edge_tilt
        }
    
    @classmethod
    def from_tuple(cls, tilt_tuple: Tuple[int, int, int]) -> 'TiltConfig':
        """从元组创建"""
        return cls(
            goal_tilt=tilt_tuple[0],
            veh_veh_tilt=tilt_tuple[1],
            veh_edge_tilt=tilt_tuple[2]
        )


class _DummyModel:
    """替代 GPU 模型，仅提供 Policy.__init__ 需要的 cfg 属性和 eval() 方法。"""
    def __init__(self, checkpoint_cfg):
        self.cfg = checkpoint_cfg
    def eval(self):
        return self


class PerVehicleAutoregressivePolicy(AutoregressivePolicy):
    """
    Per-vehicle tilting policy subclass
    
    覆写 process_predicted_rtg 方法以支持每个车辆使用独立的 tilt
    """
    
    def process_predicted_rtg(self, rtg_logits, token_index, veh_id, dset, vehicle_data_dict, 
                            data, agent_idx_dict, is_tilted=False, device='cuda'):
        """
        处理预测的 RTG，应用 per-vehicle tilting
        
        使用 batch_inference.discretization_utils 中的纯函数，
        替代直接调用 dset.get_tilt_logits() / dset.undiscretize_rtgs()。
        """
        idx = agent_idx_dict[self.veh_id_to_idx[veh_id]]
        
        rtg_logits_3 = rtg_logits[0, idx, token_index].reshape(
            self.cfg_rl_waymo.rtg_discretization, self.cfg_model.num_reward_components
        )
        
        # 获取 per-vehicle tilt 参数
        per_vehicle_map = self.tilt_dict.get('per_vehicle', {})
        if is_tilted and self.tilt_dict.get('tilt', False):
            if veh_id in per_vehicle_map:
                g, v, e = per_vehicle_map[veh_id]
            else:
                g = getattr(self, 'goal_tilt', 0)
                v = getattr(self, 'veh_veh_tilt', 0)
                e = getattr(self, 'veh_edge_tilt', 0)
        else:
            g, v, e = 0, 0, 0

        rtg_discretization = self.cfg_rl_waymo.rtg_discretization
        tilt_logits_np = _get_tilt_logits(rtg_discretization, g, v, e)
        
        (goal_idx, veh_idx, road_idx), (goal_val, veh_val, road_val) = _decode_predicted_rtg(
            rtg_logits_3, tilt_logits_np,
            rtg_discretization,
            self.cfg_rl_waymo.min_rtg_pos, self.cfg_rl_waymo.max_rtg_pos,
            self.cfg_rl_waymo.min_rtg_veh, self.cfg_rl_waymo.max_rtg_veh,
            self.cfg_rl_waymo.min_rtg_road, self.cfg_rl_waymo.max_rtg_road,
            device=device,
        )
        
        vehicle_data_dict[veh_id]['next_rtg_goal'] = goal_val
        vehicle_data_dict[veh_id]['next_rtg_veh'] = veh_val
        vehicle_data_dict[veh_id]['next_rtg_road'] = road_val
        
        # append predicted RTG to data dictionary before making action prediction
        data['agent'].rtgs[0, idx, token_index, 0] = goal_idx
        data['agent'].rtgs[0, idx, token_index, 1] = veh_idx
        data['agent'].rtgs[0, idx, token_index, 2] = road_idx
        
        next_rtgs = [goal_idx, veh_idx, road_idx]
        
        return vehicle_data_dict, data, next_rtgs


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
        device: str = 'cuda',
        action_temperature: float = 1.0,
        nucleus_sampling: bool = False,
        nucleus_threshold: float = 0.8,
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
            load_on_init: 是否在初始化时立即加载模型/数据集
        """
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.dataset = None
        self.batch_inference = False
        self._checkpoint_cfg = None
        if load_on_init:
            self._load_model_and_dataset()
        
        # 策略配置
        self.action_temperature = action_temperature
        self.nucleus_sampling = nucleus_sampling
        self.nucleus_threshold = nucleus_threshold
        
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
        self._preproc_data: Dict = {}
        self._vehicles_to_control: List[int] = []
        self._vehicles_to_control_sorted: List[int] = []
        self._vehicles_to_control_set: set[int] = set()
        self._road_edge_polylines: List = []
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
        self._veh_id_to_all_idx: Dict[int, int] = {}
        self._controlled_vehicle_ids_present: List[int] = []
        self._controlled_vehicle_ids_step: List[int] = []
        self._controlled_all_indices: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._moving_agent_mask_cache: Optional[np.ndarray] = None
        self._batch_prepare_cache: Dict[str, np.ndarray] = {}
        self._controlled_reward_prefix: Optional[np.ndarray] = None
        self._step_vehicle_by_id: Dict[int, Any] = {}
        # Whether to move non-controlled vehicles out of the scene when GT is missing.
        self.allow_set_position_for_noncontrolled: bool = False
        # 缓存 vehicles 列表，供 apply_predictions warm-up 使用
        self._last_vehicles: Optional[List] = None
        self._last_vehicle_by_id: Dict[int, Any] = {}
        
        # 从配置中获取时间相关参数
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.history_steps = getattr(cfg.nocturne, 'history_steps', 10)
        self._default_initial_rtg = np.array([10.0, 90.0, 90.0], dtype=np.float32)

    def _load_checkpoint_cfg(self):
        """从 checkpoint 中只读取 cfg，不加载模型权重到 GPU。"""
        if self._checkpoint_cfg is not None:
            return
        tmp = CtRLSim.load_from_checkpoint(self.checkpoint_path, map_location='cpu')
        self._checkpoint_cfg = tmp.cfg
        del tmp

    def _validate_external_cfg_compatibility(self):
        """校验 checkpoint cfg 与 adapter cfg 的关键字段一致性。"""
        if self._checkpoint_cfg is None:
            return
        ckpt_ds = self._checkpoint_cfg.dataset.waymo
        ckpt_model = self._checkpoint_cfg.model
        adapter_ds = self.cfg.dataset.waymo
        checks = [
            ('train_context_length', getattr(ckpt_ds, 'train_context_length', None),
             getattr(adapter_ds, 'train_context_length', None)),
            ('rtg_discretization', getattr(ckpt_ds, 'rtg_discretization', None),
             getattr(adapter_ds, 'rtg_discretization', None)),
            ('accel_discretization', getattr(ckpt_ds, 'accel_discretization', None),
             getattr(adapter_ds, 'accel_discretization', None)),
            ('steer_discretization', getattr(ckpt_ds, 'steer_discretization', None),
             getattr(adapter_ds, 'steer_discretization', None)),
            ('max_num_agents', getattr(ckpt_model, 'max_num_agents', None),
             getattr(self.cfg.model, 'max_num_agents', None)),
            ('num_reward_components', getattr(ckpt_model, 'num_reward_components', None),
             getattr(self.cfg.model, 'num_reward_components', None)),
        ]
        for name, ckpt_val, adapter_val in checks:
            if ckpt_val is not None and adapter_val is not None and ckpt_val != adapter_val:
                raise ValueError(
                    f"External teacher cfg mismatch: {name} "
                    f"checkpoint={ckpt_val} vs adapter={adapter_val}"
                )

    def _load_dataset_only(self):
        """仅加载 dataset，不加载模型。"""
        self.dataset = RLWaymoDatasetCtRLSim(self.cfg, split_name='test', mode='eval')

    def _load_model_and_dataset(self):
        # 加载模型（参考: eval_sim.py 第 35 行）
        print(f"Loading CtRL-Sim model from {self.checkpoint_path}...")
        self.model = CtRLSim.load_from_checkpoint(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

        # 初始化数据集（用于数据处理，参考: policy_evaluator.py 第 40 行）
        # 注：mode='eval' 用于推理时的数据预处理
        self._load_dataset_only()
    
    def _create_policy(self) -> AutoregressivePolicy:
        """
        创建 AutoregressivePolicy 实例
        
        参考: eval_sim.py 第 42-66 行
        """
        key_dict = {
            'next_acceleration': 'next_acceleration',
            'next_steering': 'next_steering',
            'rtgs': 'rtgs'
        }
        # batch_inference 模式使用 _DummyModel 替代真模型
        if self.batch_inference:
            if self._checkpoint_cfg is None:
                self._load_checkpoint_cfg()
                self._validate_external_cfg_compatibility()
            model = _DummyModel(self._checkpoint_cfg)
        else:
            if self.model is None:
                raise RuntimeError('CtrlSim model is not loaded.')
            model = self.model
        
        # 根据是否有 per_vehicle_tilting 决定使用哪个 policy 类
        if self.per_vehicle_tilting is not None:
            # 使用 per-vehicle policy
            # 需要包含全局 tilt 参数作为默认值
            tilt_dict = {
                'tilt': True,
                'goal_tilt': self.current_tilt.goal_tilt,
                'veh_veh_tilt': self.current_tilt.veh_veh_tilt,
                'veh_edge_tilt': self.current_tilt.veh_edge_tilt,
                'per_vehicle': self.per_vehicle_tilting
            }
            return PerVehicleAutoregressivePolicy(
                cfg=self.cfg,
                model_path=self.checkpoint_path,
                model=model,
                use_rtg=True,
                predict_rtgs=True,
                discretize_rtgs=True,
                real_time_rewards=True,
                privileged_return=False,
                max_return=False,
                min_return=False,
                key_dict=key_dict,
                tilt_dict=tilt_dict,
                name='ctrl_sim',
                action_temperature=self.action_temperature,
                nucleus_sampling=self.nucleus_sampling,
                nucleus_threshold=self.nucleus_threshold,
                device=self.device
            )
        else:
            # 使用全局 tilting policy
            return AutoregressivePolicy(
                cfg=self.cfg,
                model_path=self.checkpoint_path,
                model=model,
                use_rtg=True,
                predict_rtgs=True,
                discretize_rtgs=True,
                real_time_rewards=True,
                privileged_return=False,
                max_return=False,
                min_return=False,
                key_dict=key_dict,
                tilt_dict=self.current_tilt.to_dict(),
                name='ctrl_sim',
                action_temperature=self.action_temperature,
                nucleus_sampling=self.nucleus_sampling,
                nucleus_threshold=self.nucleus_threshold,
                device=self.device
            )
    
    def set_tilting(
        self, 
        goal_tilt: int, 
        veh_veh_tilt: int, 
        veh_edge_tilt: int
    ):
        """
        设置 domain tilting 参数
        
        注：tilting 通过修改 RTG 预测的 logits 来实现
        参考: datasets/rl_waymo/dataset.py 第 347-352 行 get_tilt_logits()
        
        Args:
            goal_tilt: 目标导向程度 [-25, 25]
            veh_veh_tilt: 车-车交互激进度 [-25, 25]
            veh_edge_tilt: 车-边界交互激进度 [-25, 25]
        """
        self.current_tilt = TiltConfig(
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt
        )
        
        # 如果策略已存在，更新其 tilt_dictcheckpoint_path
        if self._policy is not None:
            self._policy.tilt_dict = self.current_tilt.to_dict()
            self._policy.goal_tilt = goal_tilt
            self._policy.veh_veh_tilt = veh_veh_tilt
            self._policy.veh_edge_tilt = veh_edge_tilt
    
    def set_tilting_from_tuple(self, tilt: Tuple[int, int, int]):
        """从元组设置 tilting（便捷接口）"""
        self.set_tilting(tilt[0], tilt[1], tilt[2])
    
    def set_per_vehicle_tilting(self, mapping: Dict[int, Tuple[int, int, int]]):
        """
        设置 per-vehicle tilting
        
        Args:
            mapping: {veh_id: (goal_tilt, veh_veh_tilt, veh_edge_tilt)} 映射字典
        """
        self.per_vehicle_tilting = mapping
        
        # 如果策略已存在，更新其 tilt_dict
        if self._policy is not None:
            self._policy.tilt_dict = {
                'tilt': True,
                'goal_tilt': self.current_tilt.goal_tilt,
                'veh_veh_tilt': self.current_tilt.veh_veh_tilt,
                'veh_edge_tilt': self.current_tilt.veh_edge_tilt,
                'per_vehicle': mapping
            }
            self._policy.goal_tilt = self.current_tilt.goal_tilt
            self._policy.veh_veh_tilt = self.current_tilt.veh_veh_tilt
            self._policy.veh_edge_tilt = self.current_tilt.veh_edge_tilt
    
    def reset(
        self,
        scenario,
        vehicles: List,
        gt_data_dict: Dict,
        preproc_data: Dict,
        vehicles_to_control: List[int],
        ego_id: Optional[int] = None,
    ):
        """
        在每个 episode 开始时调用，初始化策略状态
        
        参考: policy_evaluator.py 第 500-510 行的初始化逻辑
        
        Args:
            scenario: Nocturne scenario 对象
            vehicles: 场景中的所有车辆列表
            gt_data_dict: Ground truth 数据（由 get_ground_truth_states 生成）
            preproc_data: 预处理数据（包含 RTG 和道路信息）
            vehicles_to_control: 要控制的车辆 ID 列表（对手车辆）
        """
        # 仅在有对手车辆时才需要策略；若前序以非 normal 启动，首次转 normal 时在此加载。
        if vehicles_to_control:
            if self.batch_inference:
                # external teacher 模式：只需 dataset + DummyModel
                if self.dataset is None:
                    self._load_dataset_only()
            else:
                # 内置路径：需要真模型 + dataset
                if self.model is None or self.dataset is None:
                    self._load_model_and_dataset()
            self._policy = self._create_policy()
        else:
            self._policy = None
        
        # 存储运行时状态
        self._gt_data_dict = gt_data_dict
        self._gt_traj_by_id = {
            veh_id: np.asarray(data['traj'])
            for veh_id, data in gt_data_dict.items()
            if isinstance(data, dict) and 'traj' in data
        }
        self._preproc_data = preproc_data
        self._vehicles_to_control = list(vehicles_to_control)
        self._vehicles_to_control_set = set(self._vehicles_to_control)
        self._vehicles_to_control_sorted = sorted(
            self._vehicles_to_control,
            key=lambda veh_id: int(self._get_gt_traj_data(veh_id)[:, 4].sum())
            if self._get_gt_traj_data(veh_id) is not None
            else -1,
            reverse=True,
        )
        self._ego_id = ego_id
        self._last_vehicles = None
        self._last_vehicle_by_id = {}
        
        # 提取道路数据（参考: policy_evaluator.py 第 496-497 行）
        road_data = get_road_data(scenario)
        self._road_edge_polylines = self._extract_road_edge_polylines(road_data)
        
        # 初始化 vehicle_data_dict（参考: policy_evaluator.py 第 500-506 行）
        self._vehicle_data_dict = {}
        self._goal_dict = {}
        self._goal_dist_normalizer = {}
        self._opponent_vehicle_exits = {}
        self._opponent_last_valid_pos = {}
        self._opponent_goal_hold_until = {}
        self._moving_agent_mask_cache = None
        self._batch_prepare_cache = {}
        self._controlled_reward_prefix = None
        
        # 创建车辆索引映射
        self._veh_id_to_idx = {}
        for idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            self._veh_id_to_idx[veh_id] = idx
            gt_traj_data = self._get_gt_traj_data(veh_id)
            if gt_traj_data is None:
                raise KeyError(f"Missing gt traj data for veh_id={veh_id}")
            self._goal_dict[veh_id] = self._initialize_goal_dict(veh, gt_traj_data)
            self._vehicle_data_dict[veh_id] = self._initialize_vehicle_data_dict(
                veh, self._goal_dict[veh_id]
            )
            self._goal_dist_normalizer[veh_id] = self._compute_goal_dist_normalizer(
                veh, self._goal_dict[veh_id]['pos']
            )
            if veh_id in self._vehicles_to_control_set:
                pos = veh.getPosition()
                sim_exists = sim_position_exists(pos.x, pos.y)
                self._opponent_vehicle_exits[veh_id] = bool(sim_exists)
                if sim_exists:
                    self._opponent_last_valid_pos[veh_id] = (
                        float(pos.x),
                        float(pos.y),
                    )
                self._opponent_goal_hold_until[veh_id] = None

        self._all_vehicle_ids = list(self._vehicle_data_dict.keys())
        self._veh_id_to_all_idx = {
            veh_id: idx for idx, veh_id in enumerate(self._all_vehicle_ids)
        }
        self._controlled_vehicle_ids_present = [
            veh_id
            for veh_id in self._vehicles_to_control
            if veh_id in self._veh_id_to_all_idx
        ]
        if self._controlled_vehicle_ids_present:
            self._controlled_all_indices = np.asarray(
                [
                    self._veh_id_to_all_idx[veh_id]
                    for veh_id in self._controlled_vehicle_ids_present
                ],
                dtype=np.int64,
            )
        else:
            self._controlled_all_indices = np.zeros((0,), dtype=np.int64)
        self._controlled_vehicle_ids_step = []
        self._step_vehicle_by_id = {}
        
        # 重置策略内部状态（参考: policy.py 第 45-58 行）
        if self._policy is not None:
            self._policy.reset(self._vehicle_data_dict)

    def cache_last_valid_positions(self, vehicles: List):
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id not in self._vehicles_to_control_set:
                continue
            pos = veh.getPosition()
            if sim_position_exists(pos.x, pos.y):
                self._opponent_last_valid_pos[veh_id] = (
                    float(pos.x),
                    float(pos.y),
                )

    def get_opponent_vehicle_exists(self, veh_id: int) -> Optional[bool]:
        """
        获取对手车辆的存在标记。

        仅对对手车辆返回有效值；非对手车辆返回 None。
        """
        if veh_id not in self._vehicles_to_control_set:
            return None
        if veh_id in self._opponent_vehicle_exits:
            return bool(self._opponent_vehicle_exits[veh_id])
        exists_hist = self._vehicle_data_dict.get(veh_id, {}).get("existence")
        if exists_hist:
            return bool(exists_hist[-1])
        return None

    def post_step_fix_opponent_positions(
        self,
        vehicles: List,
        goal_points_by_id: Optional[Dict[int, np.ndarray]],
        current_step: int,
    ):
        if goal_points_by_id is None:
            goal_points_by_id = {}
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id not in self._vehicles_to_control_set:
                continue
            pos = veh.getPosition()
            sim_exists = sim_position_exists(pos.x, pos.y)
            prev_exists = self._opponent_vehicle_exits.get(veh_id, bool(sim_exists))

            if sim_exists:
                self._opponent_last_valid_pos[veh_id] = (
                    float(pos.x),
                    float(pos.y),
                )
            elif prev_exists and veh_id in self._opponent_last_valid_pos:
                last_x, last_y = self._opponent_last_valid_pos[veh_id]
                try:
                    veh.setPosition(last_x, last_y)
                    pos = veh.getPosition()
                    sim_exists = True
                except Exception:
                    pass

            goal_pos = goal_points_by_id.get(veh_id)
            reached_goal = False
            if goal_pos is not None and sim_exists:
                try:
                    goal_arr = np.asarray(goal_pos, dtype=np.float32)
                    dist = np.linalg.norm(
                        goal_arr[:2] - np.array([pos.x, pos.y], dtype=np.float32)
                    )
                    reached_goal = dist < self._goal_pos_tolerance
                except Exception:
                    reached_goal = False

            hold_until = self._opponent_goal_hold_until.get(veh_id)
            hold_until = _compute_goal_hold_until(
                hold_until,
                current_step=current_step,
                reached_goal=reached_goal,
                hold_steps=self._goal_hold_steps,
            )
            self._opponent_goal_hold_until[veh_id] = hold_until

            if _should_drop_after_goal(current_step, hold_until):
                self._opponent_vehicle_exits[veh_id] = False
                try:
                    veh.setPosition(-1000000.0, -1000000.0)
                except Exception:
                    pass
                continue

            self._opponent_vehicle_exits[veh_id] = _keep_exists_on_invalid(
                sim_exists, prev_exists
            )
    
    # ========== External teacher (batch_inference) 接口 ==========

    def prepare_step(self, t: int, vehicles: List) -> Optional[Dict]:
        """构建 prepared_dict，委托给 batch_inference.prepare_and_apply。"""
        return _batch_io.prepare_step(self, t, vehicles)

    def apply_predictions(self, model_outputs: Optional[Dict]) -> Dict[int, Tuple[float, float]]:
        """接收推理结果并返回动作，委托给 batch_inference.prepare_and_apply。"""
        return _batch_io.apply_predictions(self, model_outputs)

    def step(self, t: int, vehicles: List) -> Dict[int, Tuple[float, float]]:
        """
        执行一步推理，返回所有被控车辆的动作
        
        参考: policy_evaluator.py 第 515-542 行的仿真循环
        
        Args:
            t: 当前时间步
            vehicles: 场景中的所有车辆列表
        
        Returns:
            actions: {veh_id: (acceleration, steering)} 动作字典
        """
        # 边界情况: 没有车辆时直接返回空字典
        if len(vehicles) == 0 or self._policy is None:
            return {}
        
        # 1. 更新 vehicle_data_dict（参考: policy_evaluator.py 第 516-524 行）
        self._vehicle_data_dict = self._update_vehicle_data_dict(
            t, vehicles, self._vehicle_data_dict
        )
        
        # 2. 更新策略内部状态（参考: policy_evaluator.py 第 526 行）
        self._policy.update_state(
            self._vehicle_data_dict, 
            self._vehicles_to_control, 
            t
        )
        
        # 3. 执行推理（warm-up 阶段仅使用 GT 动作，不做模型推理）
        if t >= self.history_steps - 1:
            self._vehicle_data_dict = self._policy.predict(
                self._vehicle_data_dict,
                self._gt_data_dict,
                self._preproc_data,
                self.dataset,
                self._vehicles_to_control,
                t
            )
        
        # 4. 提取动作（参考: policy_evaluator.py 的 warm-up 逻辑）
        use_model_action = t >= self.history_steps - 1
        actions = {}
        for veh_id in self._controlled_vehicle_ids_step:
            veh_data = self._vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if use_model_action:
                veh_exists = veh_data['existence'][-1]
                if veh_exists:
                    accel = veh_data['next_acceleration']
                    steer = veh_data['next_steering']
                else:
                    accel, steer = 0.0, 0.0
                actions[veh_id] = (accel, steer)
            else:
                veh = self._step_vehicle_by_id.get(veh_id)
                if veh is not None:
                    actions[veh_id] = self._get_gt_action(veh_id, t, veh)
        
        return actions
    
    def apply_action(self, veh, action: Tuple[float, float]):
        """
        将动作应用到车辆
        
        参考: autoregressive_policy.py 第 256-274 行 act()
        
        Args:
            veh: Nocturne vehicle 对象
            action: (acceleration, steering) 元组
        """
        acceleration, steering = action
        
        if acceleration > 0.0:
            veh.acceleration = acceleration
        else:
            veh.brake(np.abs(acceleration))
        veh.steering = steering
    
    def record_action(self, veh_id: int, action: Tuple[float, float]):
        """
        记录已应用的动作到 vehicle_data_dict
        
        Args:
            veh_id: 车辆 ID
            action: (acceleration, steering) 元组
        """
        if veh_id in self._vehicle_data_dict:
            self._vehicle_data_dict[veh_id]["acceleration"].append(action[0])
            self._vehicle_data_dict[veh_id]["steering"].append(action[1])
    
    def record_all_actions(self, t: int, vehicles: List, controlled_actions: Dict[int, Tuple[float, float]]):
        """
        记录所有车辆的动作（包括非控车辆使用 ground truth）
        
        参考: policy_evaluator.py 第 532-540 行
        
        Args:
            t: 当前时间步
            vehicles: 所有车辆列表
            controlled_actions: 被控车辆的动作字典
        """
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id in controlled_actions:
                # 被控车辆使用预测的动作
                action = controlled_actions[veh_id]
            else:
                # 非控车辆使用 ground truth 动作
                action = self._get_gt_action(veh_id, t, veh)
            self.record_action(veh_id, action)
    
    def _get_gt_action(self, veh_id: int, t: int, veh=None) -> Tuple[float, float]:
        """
        获取 ground truth 动作
        
        参考: policy_evaluator.py apply_gt_action()
        """
        gt_traj = self._get_gt_traj_data(veh_id)
        if gt_traj is None:
            return (0.0, 0.0)
        if t + 1 >= len(gt_traj):
            return (0.0, 0.0)
        
        # action is only defined if state at next timestep is defined
        protected = (veh_id == self._ego_id) or (veh_id in self._vehicles_to_control_set)
        if protected:
            if veh_id in self._vehicles_to_control_set:
                exists = self.get_opponent_vehicle_exists(veh_id)
                veh_exists = 1 if exists else 0
            else:
                if veh is not None:
                    pos = veh.getPosition()
                    veh_exists = 1 if sim_position_exists(pos.x, pos.y) else 0
                else:
                    veh_exists = 0
        else:
            veh_exists = gt_traj[t, 4] and gt_traj[t + 1, 4]
        # once we encounter the first missing timestep, all future timesteps are also missing
        if t > 0 and self._vehicle_data_dict.get(veh_id, {}).get("existence") and self._vehicle_data_dict[veh_id]["existence"][-1] == 0:
            veh_exists = 0
        
        if not veh_exists:
            if veh is not None and protected:
                return (0.0, 0.0)
            if veh is not None:
                # For opponents, keep position even if GT action is missing.
                if self.allow_set_position_for_noncontrolled and veh_id not in self._vehicles_to_control_set:
                    veh.setPosition(-1000000, -1000000)
            return (0.0, 0.0)
        
        if veh is None:
            return (0.0, 0.0)
        
        accel, steer = safe_backward_action_from_states(
            prev_pos=(veh.getPosition().x, veh.getPosition().y),
            prev_theta=veh.getHeading(),
            prev_vel=veh.getSpeed(),
            curr_pos=(gt_traj[t + 1, 0], gt_traj[t + 1, 1]),
            curr_theta=gt_traj[t + 1, 2],
            curr_vel=gt_traj[t + 1, 3],
            wheel_base=gt_traj[t + 1, -1],
            dt=self.dt,
        )
        
        return (float(accel), float(steer))
    
    def _get_gt_traj_data(self, veh_id: int) -> Optional[np.ndarray]:
        """返回缓存的 GT 轨迹数组；首次访问时按需从 _gt_data_dict 建立缓存。"""
        gt_traj = self._gt_traj_by_id.get(veh_id)
        if gt_traj is not None:
            return gt_traj
        data = self._gt_data_dict.get(veh_id)
        if not isinstance(data, dict) or 'traj' not in data:
            return None
        gt_traj = np.asarray(data['traj'])
        self._gt_traj_by_id[veh_id] = gt_traj
        return gt_traj

    # ========== 辅助方法（复用 PolicyEvaluator 逻辑）==========
    
    def _extract_road_edge_polylines(self, road_data: List[Dict]) -> List:
        """
        提取道路边界多边形
        
        参考: evaluator.py 第 112-125 行 extract_road_edge_polylines()
        """
        road_edge_polylines = []
        for road in road_data:
            if road['type'] == 'road_edge':
                geometry = road['geometry']
                if isinstance(geometry, list):
                    polyline = np.array([[pt['x'], pt['y']] for pt in geometry])
                    road_edge_polylines.append(polyline)
        return road_edge_polylines
    
    def _initialize_goal_dict(self, veh, gt_traj_data: np.ndarray) -> Dict:
        """
        初始化目标字典
        
        参考: evaluator.py 第 60-73 行 initialize_goal_dict()
        """
        goal_pos = np.array([veh.target_position.x, veh.target_position.y])
        goal_heading = veh.target_heading
        goal_speed = veh.target_speed
        
        idx_disappear = np.where(gt_traj_data[:, 4] == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1
            if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj_data[idx_goal, :2]
                goal_heading = gt_traj_data[idx_goal, 2]
                goal_speed = gt_traj_data[idx_goal, 3]
        
        return {
            'pos': goal_pos,
            'heading': goal_heading,
            'speed': goal_speed
        }
    
    def _initialize_vehicle_data_dict(self, veh, goal_dict: Dict) -> Dict:
        """
        初始化车辆数据字典
        
        参考: policy_evaluator.py 第 70-97 行 initialize_vehicle_data_dict()
        """
        return {
            "gt_position": [],
            "gt_speed": [],
            "gt_heading": [],
            "gt_acceleration": [],
            "gt_nearest_dist": [],
            "position": [],
            "velocity": [],
            "heading": [],
            "nearest_dist": [],
            "existence": [],
            "acceleration": [0.0],  # 初始化为0，避免 t=0 时访问 [-1] 出错
            "steering": [0.0],      # 初始化为0，避免 t=0 时访问 [-1] 出错
            "reward": [],
            "dense_reward": [],
            "goal_position": {'x': goal_dict['pos'][0], 'y': goal_dict['pos'][1]},
            "goal_heading": goal_dict['heading'],
            "goal_speed": goal_dict['speed'],
            "width": veh.getWidth(),
            "length": veh.getLength(),
            "type": get_object_type_str(veh),
            "timestep": [],
            "rtgs": [],
            "next_acceleration": 0.,
            "next_steering": 0.
        }
    
    def _compute_goal_dist_normalizer(self, veh, goal_pos: np.ndarray) -> float:
        """
        计算目标距离归一化因子
        
        参考: evaluator.py 第 76-81 行 compute_goal_dist_normalizer()
        """
        obj_pos = veh.getPosition()
        obj_pos = np.array([obj_pos.x, obj_pos.y])
        dist = np.linalg.norm(obj_pos - goal_pos)
        return dist if dist > 0 else 1.0

    def _get_initial_rtg(self, veh_id: int, veh_idx: int, t: int) -> np.ndarray:
        """读取初始 RTG；缺失或越界时返回默认值。"""
        preproc_data = self._preproc_data
        if preproc_data is None or 'rtgs' not in preproc_data:
            return self._default_initial_rtg.copy()

        rtgs_array = preproc_data['rtgs']
        if not hasattr(rtgs_array, 'shape'):
            return self._default_initial_rtg.copy()

        if (
            isinstance(self._veh_id_to_preproc_idx, dict)
            and veh_id in self._veh_id_to_preproc_idx
        ):
            preproc_idx = int(self._veh_id_to_preproc_idx[veh_id])
        else:
            warnings.warn(
                f"veh_id_to_preproc_idx missing for veh_id={veh_id}; "
                f"fallback to veh_idx={veh_idx}.",
                UserWarning,
                stacklevel=3,
            )
            preproc_idx = veh_idx

        num_agents_in_rtg = rtgs_array.shape[0]
        if not (0 <= preproc_idx < num_agents_in_rtg):
            return self._default_initial_rtg.copy()

        try:
            unnormalized_rtg = rtgs_array[preproc_idx, t]
            return np.concatenate(
                [unnormalized_rtg[:1], unnormalized_rtg[3:]],
                axis=-1,
            )
        except (IndexError, KeyError) as exc:
            print(
                f"Warning: Failed to get RTG for preproc_idx={preproc_idx}, "
                f"veh_idx={veh_idx}, veh_id={veh_id}: {exc}"
            )
            return self._default_initial_rtg.copy()
    
    def _update_vehicle_data_dict(
        self, t: int, vehicles: List, vehicle_data_dict: Dict
    ) -> Dict:
        """
        更新车辆数据字典
        
        参考: policy_evaluator.py 第 99-146 行 update_vehicle_data_dict()
        """
        vehicles_to_control_set = self._vehicles_to_control_set
        ego_id = self._ego_id
        rew_cfg = self.cfg.nocturne['rew_cfg']
        collision_fix = getattr(self.cfg.nocturne, 'collision_fix', True)
        step_vehicle_by_id: Dict[int, Any] = {}
        controlled_vehicle_ids_step: List[int] = []
        for veh_idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            gt_traj_data = self._get_gt_traj_data(veh_id)
            if gt_traj_data is None:
                continue
            veh_data = vehicle_data_dict[veh_id]
            
            # 更新 ground truth 信息
            veh_data["gt_position"].append({
                'x': gt_traj_data[t, 0], 
                'y': gt_traj_data[t, 1]
            })
            veh_data["gt_heading"].append(gt_traj_data[t, 2])
            veh_data["gt_speed"].append(gt_traj_data[t, 3])
            
            # 计算 ground truth 加速度（中心差分）
            if t > 0 and t < self.steps - 1:
                gt_accel = (gt_traj_data[t+1, 3] - gt_traj_data[t-1, 3]) / (2 * self.dt)
                veh_data["gt_acceleration"].append(gt_accel)
            else:
                veh_data["gt_acceleration"].append(0)
            
            # 更新当前状态
            pos = veh.getPosition()
            velocity = veh.velocity()
            veh_data['position'].append({'x': pos.x, 'y': pos.y})
            veh_data["velocity"].append({'x': velocity.x, 'y': velocity.y})
            veh_data["heading"].append(veh.getHeading())
            veh_data["timestep"].append(t)
            
            # 更新存在状态
            is_controlled = veh_id in vehicles_to_control_set
            if is_controlled:
                controlled_vehicle_ids_step.append(veh_id)
                step_vehicle_by_id[veh_id] = veh
            protected = (veh_id == ego_id) or is_controlled
            if protected:
                if is_controlled:
                    sim_exists = sim_position_exists(pos.x, pos.y)
                    prev_exists = self._opponent_vehicle_exits.get(veh_id, bool(sim_exists))
                    hold_until = self._opponent_goal_hold_until.get(veh_id)
                    exists = _keep_exists_on_invalid(sim_exists, prev_exists)
                    if _should_drop_after_goal(t, hold_until):
                        exists = False
                    self._opponent_vehicle_exits[veh_id] = bool(exists)
                    veh_exists = 1 if exists else 0
                else:
                    veh_exists = 1 if sim_position_exists(pos.x, pos.y) else 0
            else:
                veh_exists = gt_traj_data[t, 4]
            if (
                t > 0
                and not is_controlled
                and veh_data["existence"][-1] == 0
            ):
                veh_exists = 0
            veh_data["existence"].append(veh_exists)
            
            # 初始化/更新 RTG（参考: policy_evaluator.py 第 121-143 行）
            if t == 0:
                veh_data["rtgs"].append(self._get_initial_rtg(veh_id, veh_idx, t))
            else:
                # 计算 dense reward 并更新 RTG
                dense_rewards = veh_data["dense_reward"]
                if dense_rewards:
                    veh_data["rtgs"].append(veh_data["rtgs"][-1] - dense_rewards[-1])
            
            # 计算 reward（参考: policy_evaluator.py 第 144-146 行）
            reward = compute_reward(
                rew_cfg,
                veh,
                self._goal_dict[veh_id],
                self._goal_dist_normalizer[veh_id],
                vehicle_data_dict,
                collision_fix=collision_fix,
            )
            veh_data["reward"].append(reward)
        self._controlled_vehicle_ids_step = controlled_vehicle_ids_step
        self._step_vehicle_by_id = step_vehicle_by_id
        
        # 计算 dense reward / 最近距离（对齐 ctrl-sim）
        if self._policy.real_time_rewards:
            vehicle_data_dict = self._compute_dense_reward(t, vehicle_data_dict)
        else:
            vehicle_data_dict = self._compute_nearest_dist_all(t, vehicle_data_dict)
        
        return vehicle_data_dict

    def _compute_nearest_dist_all(self, t: int, vehicle_data_dict: Dict) -> Dict:
        """
        计算车-车最近距离（对齐 ctrl-sim evaluator.py compute_nearest_dist_all）
        """
        veh_ids = self._all_vehicle_ids
        if not veh_ids:
            veh_ids = list(vehicle_data_dict.keys())
        if not veh_ids:
            return vehicle_data_dict

        veh_data_list = [vehicle_data_dict[veh_id] for veh_id in veh_ids]
        ag_data_xy_exist = np.array(
            [
                [
                    veh_data["position"][t]['x'],
                    veh_data["position"][t]['y'],
                    veh_data["existence"][t],
                ]
                for veh_data in veh_data_list
            ]
        )[:, np.newaxis, :]
        all_existence = ag_data_xy_exist[:, 0, 2]
        existence_scale = all_existence[:, np.newaxis].astype(float)

        veh_veh_dist_rewards = (
            self.dataset.compute_dist_to_nearest_vehicle_rewards(
                ag_data_xy_exist,
                normalize=False,
            )
            * existence_scale
        )

        gt_ag_data = np.array(
            [
                [
                    veh_data["gt_position"][t]['x'],
                    veh_data["gt_position"][t]['y'],
                    veh_data["existence"][t],
                ]
                for veh_data in veh_data_list
            ]
        )[:, np.newaxis, :]
        veh_veh_dist_rewards_gt = (
            self.dataset.compute_dist_to_nearest_vehicle_rewards(
                gt_ag_data,
                normalize=False,
            )
            * existence_scale
        )

        for idx, veh_data in enumerate(veh_data_list):
            veh_data["nearest_dist"].append(veh_veh_dist_rewards[idx, 0])
            veh_data["gt_nearest_dist"].append(veh_veh_dist_rewards_gt[idx, 0])

        return vehicle_data_dict
    
    def _compute_dense_reward(
        self, t: int, vehicle_data_dict: Dict
    ) -> Dict:
        """
        计算 dense reward（仅对受控车辆）

        参考: evaluator.py 第 127-170 行 compute_dense_reward()
        """
        veh_ids = self._all_vehicle_ids
        if not veh_ids:
            veh_ids = list(vehicle_data_dict.keys())
        if not veh_ids:
            return vehicle_data_dict

        controlled_ids = self._controlled_vehicle_ids_present
        veh_data_list = [vehicle_data_dict[veh_id] for veh_id in veh_ids]
        all_positions = np.array(
            [
                [
                    veh_data["position"][t]['x'],
                    veh_data["position"][t]['y'],
                ]
                for veh_data in veh_data_list
            ],
            dtype=np.float32,
        )
        all_gt_positions = np.array(
            [
                [
                    veh_data["gt_position"][t]['x'],
                    veh_data["gt_position"][t]['y'],
                ]
                for veh_data in veh_data_list
            ],
            dtype=np.float32,
        )
        all_existence = np.array(
            [veh_data["existence"][t] for veh_data in veh_data_list],
            dtype=np.float32,
        )
        num_agents = len(veh_ids)

        nearest_dist_values = np.zeros(num_agents, dtype=np.float32)
        gt_nearest_dist_values = np.zeros(num_agents, dtype=np.float32)
        dense_rewards_by_idx: List[Optional[np.ndarray]] = [None] * num_agents

        cfg_dataset = self.cfg.dataset.waymo
        dense_template = np.zeros(self.cfg.model.num_reward_components, dtype=np.float32)

        if controlled_ids:
            controlled_all_indices = self._controlled_all_indices
            controlled_positions = all_positions[controlled_all_indices]
            controlled_gt_positions = all_gt_positions[controlled_all_indices]
            controlled_existence = all_existence[controlled_all_indices]

            if len(self._road_edge_polylines) > 0:
                controlled_xy = controlled_positions[:, np.newaxis, :]
                veh_edge_dist_rewards = self.dataset.compute_dist_to_nearest_road_edge_rewards(
                    controlled_xy,
                    self._road_edge_polylines,
                )
                veh_edge_dist_rewards = (
                    veh_edge_dist_rewards
                    * controlled_existence[:, np.newaxis].astype(float)
                )
            else:
                veh_edge_dist_rewards = np.zeros((len(controlled_ids), 1), dtype=float)

            veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
                target_positions=controlled_positions,
                all_positions=all_positions,
                all_existence=all_existence,
                target_existence=controlled_existence,
                target_all_indices=controlled_all_indices,
            )
            veh_veh_dist_rewards_gt = self._compute_nearest_dist_to_all(
                target_positions=controlled_gt_positions,
                all_positions=all_gt_positions,
                all_existence=all_existence,
                target_existence=controlled_existence,
                target_all_indices=controlled_all_indices,
            )

            max_veh_veh_distance = cfg_dataset.max_veh_veh_distance
            nearest_dist_values[controlled_all_indices] = (
                veh_veh_dist_rewards[:, 0] * max_veh_veh_distance
            )
            gt_nearest_dist_values[controlled_all_indices] = (
                veh_veh_dist_rewards_gt[:, 0] * max_veh_veh_distance
            )

            veh_veh_dist_rewards_norm = np.clip(
                veh_veh_dist_rewards,
                a_min=0.0,
                a_max=max_veh_veh_distance,
            )
            veh_veh_dist_rewards_norm = veh_veh_dist_rewards_norm / max_veh_veh_distance

            if (
                self._controlled_reward_prefix is None
                or self._controlled_reward_prefix.shape[0] != len(controlled_ids)
            ):
                self._controlled_reward_prefix = np.asarray(
                    [vehicle_data_dict[veh_id]["reward"][0] for veh_id in controlled_ids]
                )[:, np.newaxis, :]
            processed_rewards = self._controlled_reward_prefix
            processed_rewards = (
                processed_rewards
                * controlled_existence[:, np.newaxis, np.newaxis].astype(float)
            )
            controlled_ag_data = np.concatenate(
                [
                    controlled_positions,
                    controlled_existence[:, np.newaxis],
                ],
                axis=1,
            )[:, np.newaxis, :]
            controlled_rewards = self.dataset.compute_rewards(
                controlled_ag_data,
                processed_rewards,
                veh_edge_dist_rewards,
                veh_veh_dist_rewards_norm,
            )
            controlled_rewards = np.concatenate(
                [controlled_rewards[:, :, :1], controlled_rewards[:, :, 3:]],
                axis=-1,
            )

            dense_template = np.zeros_like(controlled_rewards[0, 0], dtype=np.float32)
            for controlled_idx, all_idx in enumerate(controlled_all_indices):
                dense_rewards_by_idx[int(all_idx)] = controlled_rewards[controlled_idx, 0]

        for idx, veh_data in enumerate(veh_data_list):
            veh_data["nearest_dist"].append(float(nearest_dist_values[idx]))
            veh_data["gt_nearest_dist"].append(float(gt_nearest_dist_values[idx]))
            dense_reward = dense_rewards_by_idx[idx]
            veh_data["dense_reward"].append(
                dense_reward if dense_reward is not None else dense_template.copy()
            )

        return vehicle_data_dict

    @staticmethod
    def _compute_nearest_dist_to_all(
        target_positions: np.ndarray,
        all_positions: np.ndarray,
        all_existence: np.ndarray,
        target_existence: np.ndarray,
        target_all_indices: np.ndarray,
    ) -> np.ndarray:
        """计算目标车辆到全体车辆的最近距离（不含自身）。"""
        if len(target_positions) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        with np.errstate(invalid='ignore'):
            diff = target_positions[:, np.newaxis, :] - all_positions[np.newaxis, :, :]
            squared_dist = np.sum(diff ** 2, axis=-1)

        valid_all = all_existence.astype(bool)
        squared_dist[:, ~valid_all] = np.inf
        row_idx = np.arange(len(target_positions), dtype=np.int64)
        squared_dist[row_idx, target_all_indices] = np.inf

        nearest = np.sqrt(np.min(squared_dist, axis=1))
        nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)
        nearest = nearest * target_existence

        return nearest[:, np.newaxis].astype(np.float32)

    def finalize(self, vehicles: List) -> Dict:
        """
        在 episode 结束后调用，记录最终状态（对齐 policy_evaluator.py）
        
        注意：最后一次step时数据已经更新过，这里不需要再调用_update_vehicle_data_dict
        """
        # 只添加最终的加速度和转向（这些不会在step中被记录）
        for veh in vehicles:
            veh_id = veh.getID()
            # 检查vehicle_data_dict中是否有这个车辆的数据
            if veh_id in self._vehicle_data_dict:
                self._vehicle_data_dict[veh_id]["acceleration"].append(0)
                self._vehicle_data_dict[veh_id]["steering"].append(0)
        return self._vehicle_data_dict
    
    @property
    def is_initialized(self) -> bool:
        """检查适配器是否已初始化"""
        return self._policy is not None
    
    def get_vehicle_data(self, veh_id: int) -> Optional[Dict]:
        """获取指定车辆的数据"""
        return self._vehicle_data_dict.get(veh_id)
