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
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from models.ctrl_sim import CtRLSim
from policies.autoregressive_policy import AutoregressivePolicy
from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from utils.sim import get_road_data, get_moving_vehicles, compute_reward
from utils.data import get_object_type_str, compute_distance_to_road_edge


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


class PerVehicleAutoregressivePolicy(AutoregressivePolicy):
    """
    Per-vehicle tilting policy subclass
    
    覆写 process_predicted_rtg 方法以支持每个车辆使用独立的 tilt
    """
    
    def process_predicted_rtg(self, rtg_logits, token_index, veh_id, dset, vehicle_data_dict, 
                            data, agent_idx_dict, is_tilted=False, device='cuda'):
        """
        处理预测的 RTG，应用 per-vehicle tilting
        
        参考: policies/autoregressive_policy.py process_predicted_rtg() 方法
        
        Args:
            rtg_logits: RTG 预测 logits
            token_index: token 索引
            veh_id: 车辆 ID
            dset: 数据集对象
            vehicle_data_dict: 车辆数据字典
            data: 数据
            agent_idx_dict: agent 索引字典
            is_tilted: 是否应用 tilting
            device: 设备
        """
        idx = agent_idx_dict[self.veh_id_to_idx[veh_id]]
        
        next_rtg_logits = rtg_logits[0, idx, token_index].reshape(
            self.cfg_rl_waymo.rtg_discretization, self.cfg_model.num_reward_components
        )
        next_rtg_goal_logits = next_rtg_logits[:, 0]
        next_rtg_veh_logits = next_rtg_logits[:, 1]
        next_rtg_road_logits = next_rtg_logits[:, 2]
        
        # 获取 per-vehicle mapping
        per_vehicle_map = self.tilt_dict.get('per_vehicle', {})
        
        # is_tilted is whether we tilt the specific agent and self.tilt_dict['tilt'] is whether the model supports tilting
        if is_tilted and self.tilt_dict.get('tilt', False):
            if veh_id in per_vehicle_map:
                g, v, e = per_vehicle_map[veh_id]
            else:
                g = getattr(self, 'goal_tilt', 0)
                v = getattr(self, 'veh_veh_tilt', 0)
                e = getattr(self, 'veh_edge_tilt', 0)
            tilt_logits = torch.from_numpy(dset.get_tilt_logits(g, v, e)).to(device)
        else:
            tilt_logits = torch.from_numpy(dset.get_tilt_logits(0, 0, 0)).to(device)
        
        next_rtg_goal_dis = F.softmax(next_rtg_goal_logits + tilt_logits[:, 0], dim=0)
        next_rtg_goal = torch.multinomial(next_rtg_goal_dis, 1)
        next_rtg_veh_dis = F.softmax(next_rtg_veh_logits + tilt_logits[:, 1], dim=0)
        next_rtg_veh = torch.multinomial(next_rtg_veh_dis, 1)
        next_rtg_road_dis = F.softmax(next_rtg_road_logits + tilt_logits[:, 2], dim=0)
        next_rtg_road = torch.multinomial(next_rtg_road_dis, 1)
        
        next_rtg = torch.cat(
            [next_rtg_goal.reshape(1, 1, 1), next_rtg_veh.reshape(1, 1, 1), next_rtg_road.reshape(1, 1, 1)],
            dim=2
        )
        next_rtg_continuous = dset.undiscretize_rtgs(next_rtg.cpu().numpy())
        vehicle_data_dict[veh_id]['next_rtg_goal'] = next_rtg_continuous[0, 0, 0]
        vehicle_data_dict[veh_id]['next_rtg_veh'] = next_rtg_continuous[0, 0, 1]
        vehicle_data_dict[veh_id]['next_rtg_road'] = next_rtg_continuous[0, 0, 2]
        
        # append predicted RTG to data dictionary before making action prediction
        data['agent'].rtgs[0, idx, token_index, 0] = next_rtg_goal
        data['agent'].rtgs[0, idx, token_index, 1] = next_rtg_veh
        data['agent'].rtgs[0, idx, token_index, 2] = next_rtg_road
        
        next_rtgs = [next_rtg_goal, next_rtg_veh, next_rtg_road]
        
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
    ):
        """
        Args:
            cfg: Hydra 配置对象（需包含 nocturne, dataset.waymo, model 等配置）
            checkpoint_path: ctrl-sim 模型 checkpoint 路径
            device: 推理设备
            action_temperature: 动作采样温度（参考: cfgs/policy/ctrl_sim.yaml）
            nucleus_sampling: 是否使用 nucleus sampling
            nucleus_threshold: nucleus sampling 阈值
        """
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 加载模型（参考: eval_sim.py 第 35 行）
        print(f"Loading CtRL-Sim model from {checkpoint_path}...")
        self.model = CtRLSim.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully.")
        
        # 初始化数据集（用于数据处理，参考: policy_evaluator.py 第 40 行）
        # 注：mode='eval' 用于推理时的数据预处理
        self.dataset = RLWaymoDatasetCtRLSim(cfg, split_name='test', mode='eval')
        
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
        self._preproc_data: Dict = {}
        self._vehicles_to_control: List[int] = []
        self._road_edge_polylines: List = []
        self._goal_dict: Dict = {}
        self._goal_dist_normalizer: Dict = {}
        
        # 从配置中获取时间相关参数
        self.dt = cfg.nocturne.dt
        self.steps = cfg.nocturne.steps
        self.history_steps = getattr(cfg.nocturne, 'history_steps', 10)
    
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
                model=self.model,
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
                model=self.model,
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
        # 创建新的策略实例
        self._policy = self._create_policy()
        
        # 存储运行时状态
        self._gt_data_dict = gt_data_dict
        self._preproc_data = preproc_data
        self._vehicles_to_control = vehicles_to_control
        
        # 提取道路数据（参考: policy_evaluator.py 第 496-497 行）
        road_data = get_road_data(scenario)
        self._road_edge_polylines = self._extract_road_edge_polylines(road_data)
        
        # 初始化 vehicle_data_dict（参考: policy_evaluator.py 第 500-506 行）
        self._vehicle_data_dict = {}
        self._goal_dict = {}
        self._goal_dist_normalizer = {}
        
        # 创建车辆索引映射
        self._veh_id_to_idx = {}
        for idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            self._veh_id_to_idx[veh_id] = idx
            gt_traj_data = np.array(gt_data_dict[veh_id]['traj'])
            self._goal_dict[veh_id] = self._initialize_goal_dict(veh, gt_traj_data)
            self._vehicle_data_dict[veh_id] = self._initialize_vehicle_data_dict(
                veh, self._goal_dict[veh_id]
            )
            self._goal_dist_normalizer[veh_id] = self._compute_goal_dist_normalizer(
                veh, self._goal_dict[veh_id]['pos']
            )
        
        # 重置策略内部状态（参考: policy.py 第 45-58 行）
        self._policy.reset(self._vehicle_data_dict)
    
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
        
        # 3. 执行推理（参考: policy_evaluator.py 第 530 行）
        self._vehicle_data_dict = self._policy.predict(
            self._vehicle_data_dict,
            self._gt_data_dict,
            self._preproc_data,
            self.dataset,
            self._vehicles_to_control,
            t
        )
        
        # 4. 提取动作（参考: autoregressive_policy.py 第 256-274 行 act()）
        actions = {}
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id in self._vehicles_to_control:
                veh_exists = self._vehicle_data_dict[veh_id]['existence'][-1]
                if veh_exists:
                    accel = self._vehicle_data_dict[veh_id]['next_acceleration']
                    steer = self._vehicle_data_dict[veh_id]['next_steering']
                else:
                    accel, steer = 0.0, 0.0
                actions[veh_id] = (accel, steer)
        
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
                action = self._get_gt_action(veh_id, t)
            self.record_action(veh_id, action)
    
    def _get_gt_action(self, veh_id: int, t: int) -> Tuple[float, float]:
        """
        获取 ground truth 动作
        
        参考: policy_evaluator.py apply_gt_action()
        """
        if veh_id not in self._gt_data_dict:
            return (0.0, 0.0)
        
        gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
        
        # 计算加速度（速度差分）
        if t < len(gt_traj) - 1:
            accel = (gt_traj[t+1, 3] - gt_traj[t, 3]) / self.dt
        else:
            accel = 0.0
        
        # 计算转向（航向差分）
        if t < len(gt_traj) - 1:
            steer = (gt_traj[t+1, 2] - gt_traj[t, 2]) / self.dt
        else:
            steer = 0.0
        
        return (float(accel), float(steer))
    
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
            "acceleration": [],
            "steering": [],
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
    
    def _update_vehicle_data_dict(
        self, t: int, vehicles: List, vehicle_data_dict: Dict
    ) -> Dict:
        """
        更新车辆数据字典
        
        参考: policy_evaluator.py 第 99-146 行 update_vehicle_data_dict()
        """
        cfg_model = self.model.cfg.model
        
        for veh_idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            gt_traj_data = np.array(self._gt_data_dict[veh_id]['traj'])
            
            # 更新 ground truth 信息
            vehicle_data_dict[veh_id]["gt_position"].append({
                'x': gt_traj_data[t, 0], 
                'y': gt_traj_data[t, 1]
            })
            vehicle_data_dict[veh_id]["gt_heading"].append(gt_traj_data[t, 2])
            vehicle_data_dict[veh_id]["gt_speed"].append(gt_traj_data[t, 3])
            
            # 计算 ground truth 加速度（中心差分）
            if t > 0 and t < self.steps - 1:
                gt_accel = (gt_traj_data[t+1, 3] - gt_traj_data[t-1, 3]) / (2 * self.dt)
                vehicle_data_dict[veh_id]["gt_acceleration"].append(gt_accel)
            else:
                vehicle_data_dict[veh_id]["gt_acceleration"].append(0)
            
            # 更新当前状态
            vehicle_data_dict[veh_id]['position'].append({
                'x': veh.getPosition().x, 
                'y': veh.getPosition().y
            })
            vehicle_data_dict[veh_id]["velocity"].append({
                'x': veh.velocity().x, 
                'y': veh.velocity().y
            })
            vehicle_data_dict[veh_id]["heading"].append(veh.getHeading())
            vehicle_data_dict[veh_id]["timestep"].append(t)
            
            # 更新存在状态
            veh_exists = gt_traj_data[t, 4]
            if t > 0 and vehicle_data_dict[veh_id]["existence"][-1] == 0:
                veh_exists = 0
            vehicle_data_dict[veh_id]["existence"].append(veh_exists)
            
            # 初始化/更新 RTG（参考: policy_evaluator.py 第 121-143 行）
            if t == 0:
                # 从预处理数据获取初始 RTG
                if self._preproc_data is not None and 'rtgs' in self._preproc_data:
                    unnormalized_rtg = self._preproc_data['rtgs'][veh_idx, t]
                    # 选择 goal, veh_veh, veh_edge 三个维度
                    unnormalized_rtg = np.concatenate([
                        unnormalized_rtg[:1], 
                        unnormalized_rtg[3:5]
                    ], axis=-1)
                else:
                    # 默认 RTG
                    unnormalized_rtg = np.array([10, 90, 90])
                vehicle_data_dict[veh_id]["rtgs"].append(unnormalized_rtg)
            else:
                # 计算 dense reward 并更新 RTG
                if len(vehicle_data_dict[veh_id]["dense_reward"]) > 0:
                    discounted_rtg = (
                        vehicle_data_dict[veh_id]["rtgs"][-1] - 
                        vehicle_data_dict[veh_id]["dense_reward"][-1]
                    )
                    vehicle_data_dict[veh_id]["rtgs"].append(discounted_rtg)
            
            # 计算 reward（参考: policy_evaluator.py 第 144-146 行）
            reward = compute_reward(
                self.cfg.nocturne['rew_cfg'],
                veh,
                self._goal_dict[veh_id],
                self._goal_dist_normalizer[veh_id],
                vehicle_data_dict,
                collision_fix=getattr(self.cfg.nocturne, 'collision_fix', True)
            )
            vehicle_data_dict[veh_id]["reward"].append(reward)
        
        # 计算 dense reward（参考: policy_evaluator.py compute_dense_reward）
        vehicle_data_dict = self._compute_dense_reward(t, vehicle_data_dict)
        
        return vehicle_data_dict
    
    def _compute_dense_reward(
        self, t: int, vehicle_data_dict: Dict
    ) -> Dict:
        """
        计算 dense reward（包含车-车距离和车-边界距离奖励）
        
        参考: evaluator.py 第 127-170 行 compute_dense_reward()
        """
        # 获取所有车辆位置和存在状态
        veh_ids = list(vehicle_data_dict.keys())
        num_vehicles = len(veh_ids)
        
        # 边界情况:没有车辆或只有一个车辆时,无法计算车-车距离
        if num_vehicles == 0:
            return vehicle_data_dict
        
        all_x = np.array([vehicle_data_dict[v]["position"][t]['x'] for v in veh_ids])
        all_y = np.array([vehicle_data_dict[v]["position"][t]['y'] for v in veh_ids])
        all_existence = np.array([vehicle_data_dict[v]["existence"][t] for v in veh_ids])
        
        # 计算车-车距离 (只有多于一个车辆时才计算)
        if num_vehicles > 1:
            ag_data = np.concatenate([
                all_x[:, np.newaxis], 
                all_y[:, np.newaxis], 
                all_existence[:, np.newaxis]
            ], axis=1)[:, np.newaxis, :]
            
            veh_veh_dist = self.dataset.compute_dist_to_nearest_vehicle_rewards(
                ag_data, normalize=False
            ) * all_existence[:, np.newaxis].astype(float)
        else:
            # 只有一个车辆,距离设为0
            veh_veh_dist = np.zeros((1, 1))
        
        # 计算车-边界距离
        if len(self._road_edge_polylines) > 0:
            veh_edge_dist = compute_distance_to_road_edge(
                all_x.reshape(1, -1),
                all_y.reshape(1, -1),
                self._road_edge_polylines
            )
        else:
            veh_edge_dist = np.zeros(len(veh_ids))
        
        # 更新 dense_reward 和 nearest_dist
        cfg_dataset = self.cfg.dataset.waymo
        for i, veh_id in enumerate(veh_ids):
            # 计算归一化的 dense reward
            veh_dist_normalized = np.clip(
                veh_veh_dist[i, 0], 0, cfg_dataset.max_veh_veh_distance
            ) / cfg_dataset.max_veh_veh_distance
            
            edge_dist_normalized = np.clip(
                np.abs(veh_edge_dist[i]) * cfg_dataset.dist_to_road_edge_scaling_factor,
                0, 5
            ) / 5.0
            
            # dense_reward: [goal_reward, veh_veh_reward, veh_edge_reward]
            reward = vehicle_data_dict[veh_id]["reward"][-1] if vehicle_data_dict[veh_id]["reward"] else [0]*8
            
            # Ensure reward has expected format for indexing below
            # 0: goal, 6: veh_veh collision, 7: veh_edge collision
            assert len(reward) >= 8, f"Reward vector length {len(reward)} < 8. Incompatible ctrl-sim version?"
            
            dense_reward = np.array([
                reward[0] * cfg_dataset.pos_target_achieved_rew_multiplier,  # goal
                veh_dist_normalized - reward[6] * cfg_dataset.veh_veh_collision_rew_multiplier,  # veh-veh
                edge_dist_normalized - reward[7] * cfg_dataset.veh_edge_collision_rew_multiplier   # veh-edge
            ])
            
            vehicle_data_dict[veh_id]["dense_reward"].append(dense_reward)
            vehicle_data_dict[veh_id]["nearest_dist"].append(veh_veh_dist[i, 0])
            vehicle_data_dict[veh_id]["gt_nearest_dist"].append(veh_veh_dist[i, 0])  # 简化：使用相同值
        
        return vehicle_data_dict
    
    @property
    def is_initialized(self) -> bool:
        """检查适配器是否已初始化"""
        return self._policy is not None
    
    def get_vehicle_data(self, veh_id: int) -> Optional[Dict]:
        """获取指定车辆的数据"""
        return self._vehicle_data_dict.get(veh_id)
