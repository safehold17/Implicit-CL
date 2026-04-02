"""
负责按仿真步为对手车辆计算与 ctrl-sim 兼容的奖励分量。
该模块聚合车辆间距离、道路边界距离与目标相关信息，生成状态更新所需奖励。
Computes ctrl-sim-compatible reward terms for opponent vehicles at each simulation step.
Aggregates inter-vehicle distance, road-edge distance, and goal-related signals for state updates.
"""

from typing import Any, Dict, List

import numpy as np

from . import reward_geometry as _reward_helpers

_compute_signed_distance_to_polylines_np = (
    _reward_helpers.compute_signed_distance_to_polylines_np
)


class OpponentRewardService:
    def __init__(self, adapter):
        self.adapter = adapter

    def _road_edge_scaling_factor(self) -> float:
        return float(self.adapter.cfg.dataset.waymo.dist_to_road_edge_scaling_factor)

    @staticmethod
    def _get_step_or_last(entries: List[Any], t: int) -> Any:
        if not entries:
            raise IndexError("Cannot read from empty history.")
        if t < len(entries):
            return entries[t]
        return entries[-1]

    def _get_context_vehicle_ids(self, vehicle_data_dict: Dict) -> List[int]:
        step_tensor_context = getattr(self.adapter, "_step_tensor_context", None)
        if (
            step_tensor_context is not None
            and step_tensor_context.context_vehicle_ids is not None
        ):
            return list(step_tensor_context.context_vehicle_ids)
        veh_ids = self.adapter._all_vehicle_ids
        if not veh_ids:
            veh_ids = list(vehicle_data_dict.keys())

        context_vehicle_ids: List[int] = []
        for veh_id in veh_ids:
            veh_data = vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if not veh_data.get("position"):
                continue
            if not veh_data.get("existence"):
                continue
            context_vehicle_ids.append(veh_id)
        return context_vehicle_ids

    @staticmethod
    def _build_xy_exist(
        veh_data_list: List[Dict],
        t: int,
        pos_key: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                [
                    OpponentRewardService._get_step_or_last(veh_data[pos_key], t)["x"],
                    OpponentRewardService._get_step_or_last(veh_data[pos_key], t)["y"],
                    OpponentRewardService._get_step_or_last(veh_data["existence"], t),
                ]
                for veh_data in veh_data_list
            ],
            dtype=np.float32,
        )[:, np.newaxis, :]

    def prepare_road_edge_cache(self) -> None:
        adapter = self.adapter
        road_edge_polylines = tuple(
            np.asarray(polyline, dtype=np.float32)
            for polyline in adapter._road_edge_polylines
            if len(polyline) >= 2
        )
        adapter._road_edge_polylines_cpu = road_edge_polylines
        adapter._constant_road_edge_reward_by_id = (
            self._build_constant_road_edge_reward_cache()
        )

    def _build_constant_road_edge_reward_cache(self) -> Dict[int, float]:
        adapter = self.adapter
        constant_vehicle_ids = sorted(
            veh_id
            for veh_id in adapter._constant_state_vehicle_ids
            if veh_id in adapter._constant_state_by_id
        )
        if not constant_vehicle_ids or not adapter._road_edge_polylines_cpu:
            return {}

        positions = np.asarray(
            [
                [
                    adapter._constant_state_by_id[veh_id]["position"]["x"],
                    adapter._constant_state_by_id[veh_id]["position"]["y"],
                ]
                for veh_id in constant_vehicle_ids
            ],
            dtype=np.float32,
        )
        rewards = self._compute_road_edge_rewards(
            positions,
        )
        return {
            veh_id: float(rewards[idx, 0])
            for idx, veh_id in enumerate(constant_vehicle_ids)
        }

    def _compute_road_edge_rewards(
        self,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        adapter = self.adapter
        num_targets = len(target_positions)
        if num_targets == 0:
            return np.zeros((0, 1), dtype=np.float32)

        dist_to_road_edge = _compute_signed_distance_to_polylines_np(
            np.asarray(target_positions, dtype=np.float32),
            adapter._road_edge_polylines_cpu,
        )
        rewards = (
            -np.asarray(dist_to_road_edge, dtype=np.float32)
            / self._road_edge_scaling_factor()
        )
        return rewards.reshape(num_targets, 1)

    def _get_step_vehicle_ids(self, t: int, vehicle_data_dict: Dict) -> List[int]:
        step_tensor_context = getattr(self.adapter, "_step_tensor_context", None)
        if (
            step_tensor_context is not None
            and int(step_tensor_context.step_t) == int(t)
            and step_tensor_context.update_vehicle_ids
        ):
            return list(step_tensor_context.update_vehicle_ids)
        veh_ids = self.adapter._state_update_vehicle_ids_step
        if not veh_ids:
            veh_ids = self.adapter._all_vehicle_ids
        if not veh_ids:
            veh_ids = list(vehicle_data_dict.keys())
        step_vehicle_ids: List[int] = []
        for veh_id in veh_ids:
            veh_data = vehicle_data_dict.get(veh_id)
            if veh_data is None:
                continue
            if len(veh_data["position"]) <= t:
                continue
            step_vehicle_ids.append(veh_id)
        return step_vehicle_ids

    def _compute_nearest_dist_all(self, t: int, vehicle_data_dict: Dict) -> Dict:
        """
        计算车-车最近距离（对齐 ctrl-sim evaluator.py compute_nearest_dist_all）
        Compute the nearest vehicle-to-vehicle distance and align with ctrl-sim evaluator.py compute_nearest_dist_all.
        """
        step_vehicle_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not step_vehicle_ids:
            return vehicle_data_dict

        context_vehicle_ids = self._get_context_vehicle_ids(vehicle_data_dict)
        if not context_vehicle_ids:
            return vehicle_data_dict

        step_tensor_context = getattr(self.adapter, "_step_tensor_context", None)
        context_idx_map = {veh_id: idx for idx, veh_id in enumerate(context_vehicle_ids)}
        if (
            step_tensor_context is not None
            and int(step_tensor_context.step_t) == int(t)
            and step_tensor_context.target_vehicle_ids is not None
        ):
            target_vehicle_ids = list(step_tensor_context.target_vehicle_ids)
        else:
            target_vehicle_ids = [
                veh_id for veh_id in step_vehicle_ids if veh_id in context_idx_map
            ]
        if not target_vehicle_ids:
            return vehicle_data_dict

        if (
            step_tensor_context is not None
            and int(step_tensor_context.step_t) == int(t)
            and step_tensor_context.context_positions_xy is not None
            and step_tensor_context.context_existence is not None
            and step_tensor_context.target_context_indices is not None
        ):
            all_positions = step_tensor_context.context_positions_xy
            all_existence = step_tensor_context.context_existence
            target_all_indices = step_tensor_context.target_context_indices
        else:
            context_data_list = [vehicle_data_dict[veh_id] for veh_id in context_vehicle_ids]
            all_xy_exist = self._build_xy_exist(context_data_list, t, "position")
            all_positions = all_xy_exist[:, 0, :2]
            all_existence = all_xy_exist[:, 0, 2]
            target_all_indices = np.asarray(
                [context_idx_map[veh_id] for veh_id in target_vehicle_ids],
                dtype=np.int64,
            )
        target_positions = all_positions[target_all_indices]
        target_existence = all_existence[target_all_indices]

        veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
            target_positions=target_positions,
            all_positions=all_positions,
            all_existence=all_existence,
            target_existence=target_existence,
            target_all_indices=target_all_indices,
        )

        for idx, veh_id in enumerate(target_vehicle_ids):
            veh_data = vehicle_data_dict[veh_id]
            veh_data["nearest_dist"].append(float(veh_veh_dist_rewards[idx, 0]))

        return vehicle_data_dict

    def _compute_dense_reward(
        self,
        t: int,
        vehicle_data_dict: Dict,
    ) -> Dict:
        """
        计算 dense reward
        Compute dense reward.

        参考: evaluator.py 第 127-170 行 compute_dense_reward()
        See evaluator.py lines 127-170 in compute_dense_reward().
        """
        step_vehicle_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not step_vehicle_ids:
            return vehicle_data_dict

        context_vehicle_ids = self._get_context_vehicle_ids(vehicle_data_dict)
        if not context_vehicle_ids:
            return vehicle_data_dict

        adapter = self.adapter
        dataset = adapter.dataset
        step_tensor_context = getattr(adapter, "_step_tensor_context", None)
        context_idx_map = {veh_id: idx for idx, veh_id in enumerate(context_vehicle_ids)}
        if (
            step_tensor_context is not None
            and int(step_tensor_context.step_t) == int(t)
            and step_tensor_context.target_vehicle_ids is not None
        ):
            target_vehicle_ids = [
                veh_id
                for veh_id in step_tensor_context.target_vehicle_ids
                if self._get_step_or_last(vehicle_data_dict[veh_id]["existence"], t)
            ]
            target_indices = np.asarray(
                [context_idx_map[veh_id] for veh_id in target_vehicle_ids],
                dtype=np.int64,
            )
        else:
            target_vehicle_ids = []
            for veh_id in step_vehicle_ids:
                if veh_id not in context_idx_map:
                    continue
                veh_data = vehicle_data_dict[veh_id]
                if not self._get_step_or_last(veh_data["existence"], t):
                    continue
                target_vehicle_ids.append(veh_id)
            target_indices = np.asarray(
                [context_idx_map[veh_id] for veh_id in target_vehicle_ids],
                dtype=np.int64,
            )

        if (
            step_tensor_context is not None
            and int(step_tensor_context.step_t) == int(t)
            and step_tensor_context.context_positions_xy is not None
            and step_tensor_context.context_existence is not None
        ):
            all_positions = step_tensor_context.context_positions_xy
            all_existence = step_tensor_context.context_existence
        else:
            context_data_list = [vehicle_data_dict[veh_id] for veh_id in context_vehicle_ids]
            all_xy_exist = self._build_xy_exist(context_data_list, t, "position")
            all_positions = all_xy_exist[:, 0, :2]
            all_existence = all_xy_exist[:, 0, 2]

        cfg_dataset = adapter.cfg.dataset.waymo
        dense_template = np.zeros(
            adapter.cfg.model.num_reward_components,
            dtype=np.float32,
        )
        nearest_dist_values = np.zeros((len(step_vehicle_ids),), dtype=np.float32)
        dense_reward_values = np.zeros(
            (len(step_vehicle_ids), dense_template.shape[0]),
            dtype=np.float32,
        )

        if target_vehicle_ids:
            target_positions = all_positions[target_indices]
            target_existence = all_existence[target_indices]

            veh_edge_dist_rewards = np.zeros(
                (len(target_vehicle_ids), 1),
                dtype=np.float32,
            )
            if adapter._road_edge_polylines_cpu:
                constant_road_edge_reward_by_id = (
                    adapter._constant_road_edge_reward_by_id
                )
                cached_road_rewards = np.asarray(
                    [
                        float(constant_road_edge_reward_by_id.get(veh_id, np.nan))
                        for veh_id in target_vehicle_ids
                    ],
                    dtype=np.float32,
                )
                cached_reward_mask = ~np.isnan(cached_road_rewards)
                if np.any(cached_reward_mask):
                    veh_edge_dist_rewards[cached_reward_mask, 0] = cached_road_rewards[
                        cached_reward_mask
                    ]
                dynamic_target_mask = ~cached_reward_mask
                if np.any(dynamic_target_mask):
                    dynamic_rewards = self._compute_road_edge_rewards(
                        target_positions[dynamic_target_mask],
                    )
                    veh_edge_dist_rewards[dynamic_target_mask, 0] = dynamic_rewards[:, 0]
                veh_edge_dist_rewards *= target_existence[:, np.newaxis]

            veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
                target_positions=target_positions,
                all_positions=all_positions,
                all_existence=all_existence,
                target_existence=target_existence,
                target_all_indices=target_indices,
            )

            max_veh_veh_distance = cfg_dataset.max_veh_veh_distance
            target_nearest_dist_values = veh_veh_dist_rewards[:, 0] * max_veh_veh_distance

            veh_veh_dist_rewards_norm = np.clip(
                veh_veh_dist_rewards,
                a_min=0.0,
                a_max=max_veh_veh_distance,
            )
            veh_veh_dist_rewards_norm = (
                veh_veh_dist_rewards_norm / max_veh_veh_distance
            )

            if (
                step_tensor_context is not None
                and int(step_tensor_context.step_t) == int(t)
                and step_tensor_context.latest_rewards is not None
                and step_tensor_context.target_update_indices is not None
                and step_tensor_context.latest_rewards.shape[0] > 0
                and step_tensor_context.latest_rewards.shape[1] > 0
            ):
                processed_rewards = np.asarray(
                    step_tensor_context.latest_rewards[
                        step_tensor_context.target_update_indices[: len(target_vehicle_ids)]
                    ],
                    dtype=np.float32,
                )[:, np.newaxis, :]
            else:
                processed_rewards = np.asarray(
                    [
                        (
                            np.asarray(vehicle_data_dict[veh_id]["reward"][-1], dtype=np.float32)
                            if vehicle_data_dict[veh_id]["reward"]
                            else np.asarray(adapter._zero_reward_template, dtype=np.float32)
                        )
                        for veh_id in target_vehicle_ids
                    ],
                    dtype=np.float32,
                )[:, np.newaxis, :]
            processed_rewards = (
                processed_rewards
                * target_existence[:, np.newaxis, np.newaxis]
            )
            target_ag_data = np.concatenate(
                [
                    target_positions,
                    target_existence[:, np.newaxis],
                ],
                axis=1,
            )[:, np.newaxis, :]
            target_rewards = dataset.compute_rewards(
                target_ag_data,
                processed_rewards,
                veh_edge_dist_rewards,
                veh_veh_dist_rewards_norm,
            )
            target_rewards = np.concatenate(
                [target_rewards[:, :, :1], target_rewards[:, :, 3:]],
                axis=-1,
            )

            dense_template = np.zeros_like(target_rewards[0, 0], dtype=np.float32)
            dense_reward_values = np.zeros(
                (len(step_vehicle_ids), dense_template.shape[0]),
                dtype=np.float32,
            )
            step_row_by_vehicle_id = {
                veh_id: row_idx for row_idx, veh_id in enumerate(step_vehicle_ids)
            }
            target_step_indices = np.asarray(
                [step_row_by_vehicle_id[veh_id] for veh_id in target_vehicle_ids],
                dtype=np.int64,
            )
            if target_step_indices.size > 0:
                nearest_dist_values[target_step_indices] = target_nearest_dist_values
                dense_reward_values[target_step_indices] = target_rewards[:, 0]

        for step_idx, veh_id in enumerate(step_vehicle_ids):
            veh_data = vehicle_data_dict[veh_id]
            veh_data["nearest_dist"].append(float(nearest_dist_values[step_idx]))
            veh_data["dense_reward"].append(dense_reward_values[step_idx].copy())

        return vehicle_data_dict

    @staticmethod
    def _compute_nearest_dist_to_all(
        target_positions: np.ndarray,
        all_positions: np.ndarray,
        all_existence: np.ndarray,
        target_existence: np.ndarray,
        target_all_indices: np.ndarray,
    ) -> np.ndarray:
        """
        计算目标车辆到全体车辆的最近距离（不含自身）。
        Compute the nearest distance from the target vehicle to all vehicles, excluding itself.
        """
        return _reward_helpers.compute_nearest_dist_to_all(
            target_positions=target_positions,
            all_positions=all_positions,
            all_existence=all_existence,
            target_existence=target_existence,
            target_all_indices=target_all_indices,
        )
