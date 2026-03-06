import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adapters.existence import sim_position_exists
from batch_inference import prepare_and_apply as _batch_io
from tools.safe_bicycle import safe_backward_action_from_states
from utils.data import get_object_type_onehot, get_object_type_str
from utils.sim import get_road_data

from .existence_logic import (
    _compute_goal_hold_until,
    _keep_exists_on_invalid,
    _should_drop_after_goal,
)
from .opponent_state_helpers import (
    append_gt_state_for_step,
    extract_road_edge_polylines,
    get_sim_state_entries,
    get_state_update_vehicle_ids,
    resolve_vehicle_exists,
    store_next_action,
)


def _build_gt_action_target_cache(gt_traj_by_id: Dict[int, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    cache: Dict[int, Dict[str, np.ndarray]] = {}
    for veh_id, gt_traj in gt_traj_by_id.items():
        traj = np.asarray(gt_traj)
        if traj.ndim != 2 or traj.shape[0] < 2:
            continue
        cache[int(veh_id)] = {
            "curr_exists": traj[:-1, 4].astype(bool, copy=False),
            "next_exists": traj[1:, 4].astype(bool, copy=False),
            "next_pos": traj[1:, :2].astype(np.float32, copy=False),
            "next_heading": traj[1:, 2].astype(np.float32, copy=False),
            "next_speed": traj[1:, 3].astype(np.float32, copy=False),
            "wheel_base": traj[1:, -1].astype(np.float32, copy=False),
        }
    return cache


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
        if vehicles_to_control:
            if self.adapter.batch_inference:
                if self.adapter.dataset is None:
                    self.adapter._load_dataset_only()
            else:
                if self.adapter.model is None or self.adapter.dataset is None:
                    self.adapter._load_model_and_dataset()
            self.adapter._policy = self.adapter._create_policy()
        else:
            self.adapter._policy = None

        self.adapter._gt_data_dict = gt_data_dict
        self.adapter._gt_traj_by_id = {
            veh_id: np.asarray(data["traj"])
            for veh_id, data in gt_data_dict.items()
            if isinstance(data, dict) and "traj" in data
        }
        self.adapter._gt_action_target_cache = _build_gt_action_target_cache(
            self.adapter._gt_traj_by_id
        )
        self.adapter._gt_action_runtime_cache = {}
        self.adapter._preproc_data = preproc_data
        self.adapter._vehicles_to_control = list(vehicles_to_control)
        self.adapter._vehicles_to_control_set = set(self.adapter._vehicles_to_control)
        if self.adapter._vehicles_to_control:
            trajectory_lengths = np.array(
                [
                    int(gt_traj_data[:, 4].sum()) if gt_traj_data is not None else -1
                    for gt_traj_data in (
                        self._get_gt_traj_data(veh_id)
                        for veh_id in self.adapter._vehicles_to_control
                    )
                ],
                dtype=np.int64,
            )
            sorted_idxs = np.argsort(trajectory_lengths)[::-1]
            vehicles_np = np.asarray(self.adapter._vehicles_to_control, dtype=np.int64)
            self.adapter._vehicles_to_control_sorted = list(vehicles_np[sorted_idxs].tolist())
        else:
            self.adapter._vehicles_to_control_sorted = []
        self.adapter._ego_id = ego_id
        self.adapter._last_vehicles = None
        self.adapter._last_vehicle_by_id = {}

        road_data = get_road_data(scenario)
        self.adapter._road_edge_polylines = extract_road_edge_polylines(road_data)

        self.adapter._vehicle_data_dict = {}
        self.adapter._goal_dict = {}
        self.adapter._goal_dist_normalizer = {}
        self.adapter._opponent_vehicle_exits = {}
        self.adapter._opponent_last_valid_pos = {}
        self.adapter._opponent_goal_hold_until = {}
        self.adapter._moving_agent_mask_cache = None
        self.adapter._batch_prepare_cache = {}
        self.adapter._pending_sparse_actions_step_t = None
        self.adapter._pending_sparse_actions = {}
        self.adapter._constant_state_vehicle_ids = set()
        self.adapter._constant_state_by_id = {}
        static_speed_threshold = 1e-3

        self.adapter._veh_id_to_idx = {}
        for idx, veh in enumerate(vehicles):
            veh_id = veh.getID()
            self.adapter._veh_id_to_idx[veh_id] = idx
            gt_traj_data = self._get_gt_traj_data(veh_id)
            if gt_traj_data is None:
                raise KeyError(f"Missing gt traj data for veh_id={veh_id}")
            self.adapter._goal_dict[veh_id] = self._initialize_goal_dict(veh, gt_traj_data)
            self.adapter._vehicle_data_dict[veh_id] = self._initialize_vehicle_data_dict(
                veh,
                self.adapter._goal_dict[veh_id],
            )
            self.adapter._goal_dist_normalizer[veh_id] = self._compute_goal_dist_normalizer(
                veh,
                self.adapter._goal_dict[veh_id]["pos"],
            )
            if veh_id not in self.adapter._vehicles_to_control_set:
                speed = abs(float(veh.getSpeed()))
                if speed <= static_speed_threshold:
                    pos = veh.getPosition()
                    velocity = veh.velocity()
                    gt_state = gt_traj_data[0]
                    self.adapter._constant_state_vehicle_ids.add(veh_id)
                    self.adapter._constant_state_by_id[veh_id] = {
                        "position": {"x": float(pos.x), "y": float(pos.y)},
                        "velocity": {"x": float(velocity.x), "y": float(velocity.y)},
                        "heading": float(veh.getHeading()),
                        "existence": 1 if sim_position_exists(pos.x, pos.y) else 0,
                        "gt_position": {
                            "x": float(gt_state[0]),
                            "y": float(gt_state[1]),
                        },
                        "gt_heading": float(gt_state[2]),
                        "gt_speed": float(gt_state[3]),
                    }
            if veh_id in self.adapter._vehicles_to_control_set:
                pos = veh.getPosition()
                sim_exists = sim_position_exists(pos.x, pos.y)
                self.adapter._opponent_vehicle_exits[veh_id] = bool(sim_exists)
                if sim_exists:
                    self.adapter._opponent_last_valid_pos[veh_id] = (
                        float(pos.x),
                        float(pos.y),
                    )
                self.adapter._opponent_goal_hold_until[veh_id] = None

        self.adapter._reward_service.prepare_road_edge_cache()

        self.adapter._all_vehicle_ids = list(self.adapter._vehicle_data_dict.keys())
        self.adapter._controlled_vehicle_ids_present = [
            veh_id
            for veh_id in self.adapter._vehicles_to_control
            if veh_id in self.adapter._vehicle_data_dict
        ]
        self.adapter._controlled_vehicle_ids_step = []
        self.adapter._state_update_vehicle_ids_step = list(self.adapter._all_vehicle_ids)
        self.adapter._step_vehicle_by_id = {}

        if self.adapter._policy is not None:
            self.adapter._policy.reset(self.adapter._vehicle_data_dict)
        self.adapter.sparse_inference.clear_on_reset()

    def cache_last_valid_positions(self, vehicles: List):
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id not in self.adapter._vehicles_to_control_set:
                continue
            pos = veh.getPosition()
            if sim_position_exists(pos.x, pos.y):
                self.adapter._opponent_last_valid_pos[veh_id] = (
                    float(pos.x),
                    float(pos.y),
                )

    def get_opponent_vehicle_exists(self, veh_id: int) -> Optional[bool]:
        """
        获取对手车辆的存在标记。

        仅对对手车辆返回有效值；非对手车辆返回 None。
        """
        if veh_id not in self.adapter._vehicles_to_control_set:
            return None
        if veh_id in self.adapter._opponent_vehicle_exits:
            return bool(self.adapter._opponent_vehicle_exits[veh_id])
        exists_hist = self.adapter._vehicle_data_dict.get(veh_id, {}).get("existence")
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
            if veh_id not in self.adapter._vehicles_to_control_set:
                continue
            pos = veh.getPosition()
            sim_exists = sim_position_exists(pos.x, pos.y)
            prev_exists = self.adapter._opponent_vehicle_exits.get(veh_id, bool(sim_exists))

            if sim_exists:
                self.adapter._opponent_last_valid_pos[veh_id] = (
                    float(pos.x),
                    float(pos.y),
                )
            elif prev_exists and veh_id in self.adapter._opponent_last_valid_pos:
                last_x, last_y = self.adapter._opponent_last_valid_pos[veh_id]
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
                    reached_goal = dist < self.adapter._goal_pos_tolerance
                except Exception:
                    reached_goal = False

            hold_until = self.adapter._opponent_goal_hold_until.get(veh_id)
            hold_until = _compute_goal_hold_until(
                hold_until,
                current_step=current_step,
                reached_goal=reached_goal,
                hold_steps=self.adapter._goal_hold_steps,
            )
            self.adapter._opponent_goal_hold_until[veh_id] = hold_until

            if _should_drop_after_goal(current_step, hold_until):
                self.adapter._opponent_vehicle_exits[veh_id] = False
                try:
                    veh.setPosition(-1000000.0, -1000000.0)
                except Exception:
                    pass
                continue

            self.adapter._opponent_vehicle_exits[veh_id] = _keep_exists_on_invalid(
                sim_exists,
                prev_exists,
            )

    def prepare_step(
        self,
        t: int,
        vehicles: List,
        worker_rng_state: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """构建 prepared_dict，委托给 batch_inference.prepare_and_apply。"""
        return _batch_io.prepare_step(self.adapter, t, vehicles, worker_rng_state=worker_rng_state)

    def apply_predictions(
        self,
        model_outputs: Optional[Dict],
    ) -> Dict[int, Tuple[float, float]]:
        """接收推理结果并返回动作，委托给 batch_inference.prepare_and_apply。"""
        return _batch_io.apply_predictions(self.adapter, model_outputs)

    def update_policy_state(self, t: int) -> None:
        """仅写入本 step 需要的车辆状态，避免全量 update_state 开销。"""
        policy = self.adapter._policy
        if policy is None:
            return

        states = policy.states
        types = policy.types
        actions = policy.actions
        rtgs = policy.rtgs
        timesteps = policy.timesteps
        goals = policy.goals
        goal_dim = policy.cfg_rl_waymo.goal_dim
        use_rtg = policy.use_rtg
        use_real_time_rtgs = policy.real_time_rewards and policy.use_rtg

        update_vehicle_ids = self.adapter._state_update_vehicle_ids_step
        for veh_id in update_vehicle_ids:
            veh_data = self.adapter._vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if len(veh_data["position"]) <= t:
                continue

            veh_idx = policy.veh_id_to_idx[veh_id]
            state_slot = states[veh_idx, t]
            state_slot[0] = veh_data["position"][t]["x"]
            state_slot[1] = veh_data["position"][t]["y"]
            state_slot[2] = veh_data["velocity"][t]["x"]
            state_slot[3] = veh_data["velocity"][t]["y"]
            state_slot[4] = veh_data["heading"][t]
            state_slot[5] = veh_data["length"]
            state_slot[6] = veh_data["width"]
            state_slot[7] = veh_data["existence"][t]

            if t == 0:
                types[veh_idx] = get_object_type_onehot(veh_data["type"])
            timesteps[veh_idx, t, 0] = veh_data["timestep"][t]

            if t > 0:
                action_slot = actions[veh_idx, t - 1]
                action_slot[0] = veh_data["acceleration"][t - 1]
                action_slot[1] = veh_data["steering"][t - 1]
                rtg_hist = veh_data["rtgs"]
                if use_rtg and len(rtg_hist) > t - 1:
                    rtgs[veh_idx, t - 1] = rtg_hist[t - 1]
            else:
                rtg_hist = veh_data["rtgs"]

            if use_real_time_rtgs and len(rtg_hist) > t:
                rtgs[veh_idx, t] = rtg_hist[t]

            goal_slot = goals[veh_idx, t]
            goal_slot[0] = veh_data["goal_position"]["x"]
            goal_slot[1] = veh_data["goal_position"]["y"]
            if goal_dim > 2:
                goal_slot[2] = veh_data["goal_velocity_x"]
            if goal_dim > 3:
                goal_slot[3] = veh_data["goal_velocity_y"]
            if goal_dim > 4:
                goal_slot[4] = veh_data["goal_heading"]

    def _build_warmup_gt_actions(self, t: int) -> Dict[int, Tuple[float, float]]:
        actions: Dict[int, Tuple[float, float]] = {}
        for veh_id in self.adapter._controlled_vehicle_ids_step:
            veh_data = self.adapter._vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            veh = self.adapter._step_vehicle_by_id.get(veh_id)
            if veh is None:
                continue
            action = self._get_gt_action(veh_id, t, veh)
            store_next_action(veh_data, veh_id, action, actions)
        return actions

    def _build_sparse_repeat_actions(self) -> Dict[int, Tuple[float, float]]:
        actions: Dict[int, Tuple[float, float]] = {}
        for veh_id in self.adapter._controlled_vehicle_ids_step:
            veh_data = self.adapter._vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if not veh_data["existence"][-1]:
                action = (0.0, 0.0)
            else:
                accel_hist = veh_data["acceleration"]
                steer_hist = veh_data["steering"]
                if accel_hist and steer_hist:
                    action = (float(accel_hist[-1]), float(steer_hist[-1]))
                else:
                    action = (0.0, 0.0)
            store_next_action(veh_data, veh_id, action, actions)
        return actions

    def _collect_predicted_actions(self) -> Dict[int, Tuple[float, float]]:
        actions: Dict[int, Tuple[float, float]] = {}
        for veh_id in self.adapter._controlled_vehicle_ids_step:
            veh_data = self.adapter._vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if not veh_data["existence"][-1]:
                action = (0.0, 0.0)
            else:
                action = (
                    float(veh_data["next_acceleration"]),
                    float(veh_data["next_steering"]),
                )
            store_next_action(veh_data, veh_id, action, actions)
        return actions

    def _predict_actions_with_policy(
        self,
        t: int,
        action_only_with_recursive_rtg: bool,
    ) -> Dict[int, Tuple[float, float]]:
        policy = self.adapter._policy
        if policy is None:
            return {}

        original_predict_rtgs = bool(policy.predict_rtgs)
        if action_only_with_recursive_rtg:
            policy.predict_rtgs = False

        # warm-up 跳过 predict 时，首次模型步可能还没有 relevant_agent_idxs 条目。
        for veh_id in self.adapter._controlled_vehicle_ids_present:
            policy.relevant_agent_idxs.setdefault(veh_id, [])
        try:
            self.adapter._vehicle_data_dict = policy.predict(
                self.adapter._vehicle_data_dict,
                self.adapter._gt_data_dict,
                self.adapter._preproc_data,
                self.adapter.dataset,
                self.adapter._vehicles_to_control,
                t,
            )
        finally:
            policy.predict_rtgs = original_predict_rtgs
        return self._collect_predicted_actions()

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
        if len(vehicles) == 0 or self.adapter._policy is None:
            return {}

        self.adapter._vehicle_data_dict = self._update_vehicle_data_dict(
            t,
            vehicles,
            self.adapter._vehicle_data_dict,
        )

        self.update_policy_state(t)

        use_model_action = t >= self.adapter.history_steps - 1
        if not use_model_action:
            self._predict_actions_with_policy(
                t=t,
                action_only_with_recursive_rtg=False,
            )
            return self._build_warmup_gt_actions(t)

        is_sparse_step = self.adapter.sparse_inference.is_sparse_step(
            t=t,
            history_steps=self.adapter.history_steps,
        )
        if not is_sparse_step:
            return self._predict_actions_with_policy(
                t=t,
                action_only_with_recursive_rtg=False,
            )
        if self.adapter.sparse_inference_action_repeat:
            return self._build_sparse_repeat_actions()
        return self._predict_actions_with_policy(
            t=t,
            action_only_with_recursive_rtg=True,
        )

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
        target = gt_action_target_cache.get(veh_id)

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

        if target is None:
            next_pos = gt_traj[t + 1, :2]
            next_heading = float(gt_traj[t + 1, 2])
            next_speed = float(gt_traj[t + 1, 3])
            wheel_base = float(gt_traj[t + 1, -1])
        else:
            next_pos = target["next_pos"][t]
            next_heading = float(target["next_heading"][t])
            next_speed = float(target["next_speed"][t])
            wheel_base = float(target["wheel_base"][t])

        accel, steer = safe_backward_action_from_states(
            prev_pos=(float(pos.x), float(pos.y)),
            prev_theta=heading,
            prev_vel=speed,
            curr_pos=(float(next_pos[0]), float(next_pos[1])),
            curr_theta=next_heading,
            curr_vel=next_speed,
            wheel_base=wheel_base,
            dt=self.adapter.dt,
        )

        action = (float(accel), float(steer))
        runtime_cache[cache_key] = action
        return action

    def _get_gt_traj_data(self, veh_id: int) -> Optional[np.ndarray]:
        """返回缓存的 GT 轨迹数组；首次访问时按需从 _gt_data_dict 建立缓存。"""
        gt_traj_by_id = getattr(self.adapter, "_gt_traj_by_id", None)
        if gt_traj_by_id is None:
            gt_traj_by_id = {}
            self.adapter._gt_traj_by_id = gt_traj_by_id

        gt_traj = gt_traj_by_id.get(veh_id)
        if gt_traj is not None:
            return gt_traj
        gt_data_dict = getattr(self.adapter, "_gt_data_dict", {})
        data = gt_data_dict.get(veh_id)
        if not isinstance(data, dict) or "traj" not in data:
            return None
        gt_traj = np.asarray(data["traj"])
        self.adapter._gt_traj_by_id[veh_id] = gt_traj
        return gt_traj

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
            if (
                idx_goal >= 0
                and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0
            ):
                goal_pos = gt_traj_data[idx_goal, :2]
                goal_heading = gt_traj_data[idx_goal, 2]
                goal_speed = gt_traj_data[idx_goal, 3]

        return {
            "pos": goal_pos,
            "heading": goal_heading,
            "speed": goal_speed,
        }

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
        obj_pos = np.array([obj_pos.x, obj_pos.y])
        dist = np.linalg.norm(obj_pos - goal_pos)
        return dist if dist > 0 else 1.0

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
        adapter = self.adapter
        vehicles_to_control_set = getattr(
            adapter,
            "_vehicles_to_control_set",
            set(getattr(adapter, "_vehicles_to_control", [])),
        )
        ego_id = getattr(adapter, "_ego_id", None)
        rew_cfg = adapter.cfg.nocturne["rew_cfg"]
        collision_fix = getattr(adapter.cfg.nocturne, "collision_fix", True)
        goal_dict = adapter._goal_dict
        goal_dist_normalizer = adapter._goal_dist_normalizer
        from . import opponent_adapter as _opponent_adapter_module

        reward_fn = _opponent_adapter_module.compute_reward
        step_vehicle_by_id: Dict[int, Any] = {}
        controlled_vehicle_ids_step: List[int] = []
        vehicles_by_id = adapter._vehicles_by_id_step
        vehicles_by_id.clear()
        for veh in vehicles:
            vehicles_by_id[veh.getID()] = veh

        update_vehicle_ids = get_state_update_vehicle_ids(
            adapter=adapter,
            t=t,
            vehicles_by_id=vehicles_by_id,
        )
        for veh_id in update_vehicle_ids:
            veh = vehicles_by_id[veh_id]
            gt_traj_data = self._get_gt_traj_data(veh_id)
            if gt_traj_data is None:
                continue
            veh_data = vehicle_data_dict[veh_id]
            veh_idx = adapter._veh_id_to_idx.get(veh_id, 0)
            is_controlled = veh_id in vehicles_to_control_set
            constant_state = (
                adapter._constant_state_by_id.get(veh_id)
                if (not is_controlled and veh_id in adapter._constant_state_vehicle_ids)
                else None
            )

            append_gt_state_for_step(
                veh_data=veh_data,
                gt_traj_data=gt_traj_data,
                t=t,
                steps=adapter.steps,
                dt=adapter.dt,
                constant_state=constant_state,
            )

            pos, pos_entry, velocity_entry, heading = get_sim_state_entries(
                veh=veh,
                constant_state=constant_state,
            )

            veh_data["position"].append(pos_entry)
            veh_data["velocity"].append(velocity_entry)
            veh_data["heading"].append(heading)
            veh_data["timestep"].append(t)

            if is_controlled:
                controlled_vehicle_ids_step.append(veh_id)
                step_vehicle_by_id[veh_id] = veh

            veh_exists = resolve_vehicle_exists(
                adapter=adapter,
                veh_id=veh_id,
                t=t,
                is_controlled=is_controlled,
                ego_id=ego_id,
                gt_traj_data=gt_traj_data,
                pos=pos,
                constant_state=constant_state,
            )
            if t > 0 and not is_controlled and veh_data["existence"][-1] == 0:
                veh_exists = 0
            veh_data["existence"].append(veh_exists)

            if t == 0:
                veh_data["rtgs"].append(self._get_initial_rtg(veh_id, veh_idx, t))
            else:
                dense_rewards = veh_data["dense_reward"]
                if dense_rewards:
                    veh_data["rtgs"].append(veh_data["rtgs"][-1] - dense_rewards[-1])

            reward = reward_fn(
                rew_cfg,
                veh,
                goal_dict[veh_id],
                goal_dist_normalizer[veh_id],
                vehicle_data_dict,
                collision_fix=collision_fix,
            )
            veh_data["reward"].append(reward)

        adapter._controlled_vehicle_ids_step = controlled_vehicle_ids_step
        adapter._state_update_vehicle_ids_step = update_vehicle_ids
        adapter._step_vehicle_by_id = step_vehicle_by_id

        if adapter._policy.real_time_rewards:
            vehicle_data_dict = adapter._compute_dense_reward(t, vehicle_data_dict)
        else:
            vehicle_data_dict = adapter._compute_nearest_dist_all(t, vehicle_data_dict)

        return vehicle_data_dict

    def finalize(self, vehicles: List) -> Dict:
        """
        在 episode 结束后调用，记录最终状态（对齐 policy_evaluator.py）

        注意：最后一次step时数据已经更新过，这里不需要再调用_update_vehicle_data_dict
        """
        for veh in vehicles:
            veh_id = veh.getID()
            if veh_id in self.adapter._vehicle_data_dict:
                self.adapter._vehicle_data_dict[veh_id]["acceleration"].append(0)
                self.adapter._vehicle_data_dict[veh_id]["steering"].append(0)
        return self.adapter._vehicle_data_dict

    @property
    def is_initialized(self) -> bool:
        """检查适配器是否已初始化"""
        return self.adapter._policy is not None

    def get_vehicle_data(self, veh_id: int) -> Optional[Dict]:
        """获取指定车辆的数据"""
        return self.adapter._vehicle_data_dict.get(veh_id)
