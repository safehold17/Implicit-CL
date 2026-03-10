import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ctrlsim_adapter.existence import sim_position_exists
from ctrlsim_adapter.gt_helpers import (
    build_gt_action_target_cache,
    compute_goal_dist_normalizer,
    get_gt_traj_array,
    reconstruct_gt_action,
    resolve_goal_state,
    resolve_next_gt_state,
)
from utils.data import get_object_type_str

from . import batch_runtime as _batch_io
from .existence_logic import (
    _compute_goal_hold_until,
    _keep_exists_on_invalid,
    _should_drop_after_goal,
)
from ._opponent_state import policy_sync as _policy_sync_module
from ._opponent_state import reset as _reset_module
from ._opponent_state import update as _update_module


class OpponentStateService:
    def __init__(self, adapter):
        self.adapter = adapter

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
        _reset_module.reset(
            self,
            scenario=scenario,
            vehicles=vehicles,
            gt_data_dict=gt_data_dict,
            preproc_data=preproc_data,
            vehicles_to_control=vehicles_to_control,
            ego_id=ego_id,
            build_gt_action_target_cache_fn=build_gt_action_target_cache,
        )

    def cache_last_valid_positions(self, vehicles: List):
        _reset_module.cache_last_valid_positions(self, vehicles)

    def get_opponent_vehicle_exists(self, veh_id: int) -> Optional[bool]:
        """
        获取对手车辆的存在标记。

        仅对对手车辆返回有效值；非对手车辆返回 None。
        """
        return _reset_module.get_opponent_vehicle_exists(self, veh_id)

    def post_step_fix_opponent_positions(
        self,
        vehicles: List,
        goal_points_by_id: Optional[Dict[int, np.ndarray]],
        current_step: int,
    ):
        _reset_module.post_step_fix_opponent_positions(
            self,
            vehicles=vehicles,
            goal_points_by_id=goal_points_by_id,
            current_step=current_step,
        )

    def prepare_step(
        self,
        t: int,
        vehicles: List,
        worker_rng_state: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """构建 prepared_dict，供 ExternalTeacher 批量推理。"""
        return _batch_io.prepare_step(self.adapter, t, vehicles, worker_rng_state=worker_rng_state)

    def apply_predictions(
        self,
        model_outputs: Optional[Dict],
    ) -> Dict[int, Tuple[float, float]]:
        """接收 batched 推理结果并返回动作。"""
        return _batch_io.apply_predictions(self.adapter, model_outputs)

    def update_policy_state(self, t: int) -> None:
        """仅写入本 step 需要的车辆状态，避免全量 update_state 开销。"""
        _policy_sync_module.update_policy_state(self, t)

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
        if veh_id in self.adapter._vehicle_data_dict:
            self.adapter._vehicle_data_dict[veh_id]["acceleration"].append(action[0])
            self.adapter._vehicle_data_dict[veh_id]["steering"].append(action[1])

    def record_all_actions(
        self,
        t: int,
        vehicles: List,
        controlled_actions: Dict[int, Tuple[float, float]],
    ):
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
                action = controlled_actions[veh_id]
            else:
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
        gt_action_target_cache = getattr(self.adapter, "_gt_action_target_cache", None)
        if gt_action_target_cache is None:
            gt_action_target_cache = _build_gt_action_target_cache(
                getattr(self.adapter, "_gt_traj_by_id", {})
            )
            self.adapter._gt_action_target_cache = gt_action_target_cache

        is_controlled = veh_id in self.adapter._vehicles_to_control_set
        is_protected = (veh_id == self.adapter._ego_id) or is_controlled
        if is_protected:
            if is_controlled:
                veh_exists = 1 if self.get_opponent_vehicle_exists(veh_id) else 0
            elif veh is None:
                veh_exists = 0
            else:
                pos = veh.getPosition()
                veh_exists = 1 if sim_position_exists(pos.x, pos.y) else 0
        else:
            veh_exists = gt_traj[t, 4] and gt_traj[t + 1, 4]
        exists_history = self.adapter._vehicle_data_dict.get(veh_id, {}).get("existence")
        if t > 0 and exists_history and exists_history[-1] == 0:
            veh_exists = 0

        if not veh_exists:
            if (
                veh is not None
                and not is_protected
                and self.adapter.allow_set_position_for_noncontrolled
            ):
                veh.setPosition(-1000000, -1000000)
            return (0.0, 0.0)

        if veh is None:
            return (0.0, 0.0)

        pos = veh.getPosition()
        heading = float(veh.getHeading())
        speed = float(veh.getSpeed())
        runtime_cache = getattr(self.adapter, "_gt_action_runtime_cache", None)
        if runtime_cache is None:
            runtime_cache = {}
            self.adapter._gt_action_runtime_cache = runtime_cache
        cache_key = (
            int(veh_id),
            int(t),
            float(pos.x),
            float(pos.y),
            heading,
            speed,
        )
        cached_action = runtime_cache.get(cache_key)
        if cached_action is not None:
            return cached_action

        next_pos, next_heading, next_speed, wheel_base = resolve_next_gt_state(
            gt_traj,
            gt_action_target_cache,
            veh_id,
            t,
        )
        action = reconstruct_gt_action(
            pos_x=float(pos.x),
            pos_y=float(pos.y),
            heading=heading,
            speed=speed,
            next_pos=next_pos,
            next_heading=next_heading,
            next_speed=next_speed,
            wheel_base=wheel_base,
            dt=self.adapter.dt,
        )
        runtime_cache[cache_key] = action
        return action

    def _get_gt_traj_data(self, veh_id: int) -> Optional[np.ndarray]:
        """返回缓存的 GT 轨迹数组；首次访问时按需从 _gt_data_dict 建立缓存。"""
        gt_traj_by_id = getattr(self.adapter, "_gt_traj_by_id", None)
        if gt_traj_by_id is None:
            gt_traj_by_id = {}
            self.adapter._gt_traj_by_id = gt_traj_by_id

        return get_gt_traj_array(
            getattr(self.adapter, "_gt_data_dict", {}),
            gt_traj_by_id,
            veh_id,
        )

    def _initialize_goal_dict(self, veh, gt_traj_data: np.ndarray) -> Dict:
        """
        初始化目标字典

        参考: evaluator.py 第 60-73 行 initialize_goal_dict()
        """
        return resolve_goal_state(
            target_position=np.array(
                [veh.target_position.x, veh.target_position.y],
                dtype=np.float32,
            ),
            target_heading=veh.target_heading,
            target_speed=veh.target_speed,
            gt_traj_data=gt_traj_data,
        )

    def _initialize_vehicle_data_dict(self, veh, goal_dict: Dict) -> Dict:
        """
        初始化车辆数据字典

        参考: policy_evaluator.py 第 70-97 行 initialize_vehicle_data_dict()
        """
        goal_speed = float(goal_dict["speed"])
        goal_heading = float(goal_dict["heading"])
        goal_velocity_x = goal_speed * np.cos(goal_heading)
        goal_velocity_y = goal_speed * np.sin(goal_heading)
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
            "acceleration": [0.0],
            "steering": [0.0],
            "reward": [],
            "dense_reward": [],
            "goal_position": {"x": goal_dict["pos"][0], "y": goal_dict["pos"][1]},
            "goal_heading": goal_dict["heading"],
            "goal_speed": goal_dict["speed"],
            "goal_velocity_x": goal_velocity_x,
            "goal_velocity_y": goal_velocity_y,
            "width": veh.getWidth(),
            "length": veh.getLength(),
            "type": get_object_type_str(veh),
            "timestep": [],
            "rtgs": [],
            "next_acceleration": 0.0,
            "next_steering": 0.0,
        }

    def _compute_goal_dist_normalizer(self, veh, goal_pos: np.ndarray) -> float:
        """
        计算目标距离归一化因子

        参考: evaluator.py 第 76-81 行 compute_goal_dist_normalizer()
        """
        obj_pos = veh.getPosition()
        return compute_goal_dist_normalizer(
            np.array([obj_pos.x, obj_pos.y], dtype=np.float32),
            goal_pos,
        )

    def _get_initial_rtg(self, veh_id: int, veh_idx: int, t: int) -> np.ndarray:
        """读取初始 RTG；缺失或越界时返回默认值。"""
        preproc_data = self.adapter._preproc_data
        if preproc_data is None or "rtgs" not in preproc_data:
            return self.adapter._default_initial_rtg.copy()

        rtgs_array = preproc_data["rtgs"]
        if not hasattr(rtgs_array, "shape"):
            return self.adapter._default_initial_rtg.copy()

        if (
            isinstance(self.adapter._veh_id_to_preproc_idx, dict)
            and veh_id in self.adapter._veh_id_to_preproc_idx
        ):
            preproc_idx = int(self.adapter._veh_id_to_preproc_idx[veh_id])
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
            return self.adapter._default_initial_rtg.copy()

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
            return self.adapter._default_initial_rtg.copy()

    def _update_vehicle_data_dict(
        self,
        t: int,
        vehicles: List,
        vehicle_data_dict: Dict,
    ) -> Dict:
        """
        更新车辆数据字典

        参考: policy_evaluator.py 第 99-146 行 update_vehicle_data_dict()
        """
        return _update_module.update_vehicle_data_dict(self, t, vehicles, vehicle_data_dict)

    def finalize(self, vehicles: List) -> Dict:
        """
        在 episode 结束后调用，记录最终状态（对齐 policy_evaluator.py）

        注意：最后一次step时数据已经更新过，这里不需要再调用_update_vehicle_data_dict
        """
        adapter = self.adapter
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id in adapter._vehicle_data_dict:
                adapter._vehicle_data_dict[veh_id]["acceleration"].append(0)
                adapter._vehicle_data_dict[veh_id]["steering"].append(0)
        return adapter._vehicle_data_dict

    @property
    def is_initialized(self) -> bool:
        """检查适配器是否已初始化"""
        return self.adapter._policy is not None

    def get_vehicle_data(self, veh_id: int) -> Optional[Dict]:
        """获取指定车辆的数据"""
        return self.adapter._vehicle_data_dict.get(veh_id)
