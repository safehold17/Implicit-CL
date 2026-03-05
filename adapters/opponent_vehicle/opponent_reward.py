from typing import Any, Dict, List

import numpy as np


class OpponentRewardService:
    def __init__(self, adapter):
        self.adapter = adapter

    @staticmethod
    def _get_step_or_last(entries: List[Any], t: int) -> Any:
        if not entries:
            raise IndexError("Cannot read from empty history.")
        if t < len(entries):
            return entries[t]
        return entries[-1]

    def _get_context_vehicle_ids(self, vehicle_data_dict: Dict) -> List[int]:
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
            if not veh_data.get("gt_position"):
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

    @staticmethod
    def _build_xy(
        veh_data_list: List[Dict],
        t: int,
        pos_key: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                [
                    OpponentRewardService._get_step_or_last(veh_data[pos_key], t)["x"],
                    OpponentRewardService._get_step_or_last(veh_data[pos_key], t)["y"],
                ]
                for veh_data in veh_data_list
            ],
            dtype=np.float32,
        )

    def _get_step_vehicle_ids(self, t: int, vehicle_data_dict: Dict) -> List[int]:
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
        """
        step_vehicle_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not step_vehicle_ids:
            return vehicle_data_dict

        context_vehicle_ids = self._get_context_vehicle_ids(vehicle_data_dict)
        if not context_vehicle_ids:
            return vehicle_data_dict

        context_idx_map = {
            veh_id: idx for idx, veh_id in enumerate(context_vehicle_ids)
        }
        target_vehicle_ids = [
            veh_id for veh_id in step_vehicle_ids if veh_id in context_idx_map
        ]
        if not target_vehicle_ids:
            return vehicle_data_dict

        context_data_list = [vehicle_data_dict[veh_id] for veh_id in context_vehicle_ids]
        all_xy_exist = self._build_xy_exist(context_data_list, t, "position")
        all_positions = all_xy_exist[:, 0, :2]
        all_existence = all_xy_exist[:, 0, 2]
        all_gt_positions = self._build_xy(context_data_list, t, "gt_position")

        target_all_indices = np.asarray(
            [context_idx_map[veh_id] for veh_id in target_vehicle_ids],
            dtype=np.int64,
        )
        target_positions = all_positions[target_all_indices]
        target_gt_positions = all_gt_positions[target_all_indices]
        target_existence = all_existence[target_all_indices]

        veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
            target_positions=target_positions,
            all_positions=all_positions,
            all_existence=all_existence,
            target_existence=target_existence,
            target_all_indices=target_all_indices,
        )
        veh_veh_dist_rewards_gt = self._compute_nearest_dist_to_all(
            target_positions=target_gt_positions,
            all_positions=all_gt_positions,
            all_existence=all_existence,
            target_existence=target_existence,
            target_all_indices=target_all_indices,
        )

        for idx, veh_id in enumerate(target_vehicle_ids):
            veh_data = vehicle_data_dict[veh_id]
            veh_data["nearest_dist"].append(float(veh_veh_dist_rewards[idx, 0]))
            veh_data["gt_nearest_dist"].append(float(veh_veh_dist_rewards_gt[idx, 0]))

        return vehicle_data_dict

    def _compute_dense_reward(
        self,
        t: int,
        vehicle_data_dict: Dict,
    ) -> Dict:
        """
        计算 dense reward（仅对受控车辆）

        参考: evaluator.py 第 127-170 行 compute_dense_reward()
        """
        step_vehicle_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not step_vehicle_ids:
            return vehicle_data_dict

        context_vehicle_ids = self._get_context_vehicle_ids(vehicle_data_dict)
        if not context_vehicle_ids:
            return vehicle_data_dict

        adapter = self.adapter
        dataset = adapter.dataset
        context_idx_map = {
            veh_id: idx for idx, veh_id in enumerate(context_vehicle_ids)
        }
        controlled_ids = [
            veh_id
            for veh_id in adapter._controlled_vehicle_ids_step
            if veh_id in context_idx_map
        ]
        controlled_indices = np.asarray(
            [context_idx_map[veh_id] for veh_id in controlled_ids],
            dtype=np.int64,
        )

        context_data_list = [vehicle_data_dict[veh_id] for veh_id in context_vehicle_ids]
        all_xy_exist = self._build_xy_exist(context_data_list, t, "position")
        all_positions = all_xy_exist[:, 0, :2]
        all_existence = all_xy_exist[:, 0, 2]
        all_gt_positions = self._build_xy(context_data_list, t, "gt_position")

        cfg_dataset = adapter.cfg.dataset.waymo
        dense_template = np.zeros(
            adapter.cfg.model.num_reward_components,
            dtype=np.float32,
        )
        nearest_dist_by_context_idx: Dict[int, float] = {}
        gt_nearest_dist_by_context_idx: Dict[int, float] = {}
        dense_rewards_by_context_idx: Dict[int, np.ndarray] = {}

        if controlled_ids:
            controlled_positions = all_positions[controlled_indices]
            controlled_gt_positions = all_gt_positions[controlled_indices]
            controlled_existence = all_existence[controlled_indices]

            if adapter._road_edge_polylines:
                controlled_xy = controlled_positions[:, np.newaxis, :]
                veh_edge_dist_rewards = (
                    dataset.compute_dist_to_nearest_road_edge_rewards(
                        controlled_xy,
                        adapter._road_edge_polylines,
                    )
                )
                veh_edge_dist_rewards = (
                    veh_edge_dist_rewards
                    * controlled_existence[:, np.newaxis]
                )
            else:
                veh_edge_dist_rewards = np.zeros((len(controlled_ids), 1), dtype=np.float32)

            veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
                target_positions=controlled_positions,
                all_positions=all_positions,
                all_existence=all_existence,
                target_existence=controlled_existence,
                target_all_indices=controlled_indices,
            )
            veh_veh_dist_rewards_gt = self._compute_nearest_dist_to_all(
                target_positions=controlled_gt_positions,
                all_positions=all_gt_positions,
                all_existence=all_existence,
                target_existence=controlled_existence,
                target_all_indices=controlled_indices,
            )

            max_veh_veh_distance = cfg_dataset.max_veh_veh_distance
            nearest_dist_values = veh_veh_dist_rewards[:, 0] * max_veh_veh_distance
            gt_nearest_dist_values = (
                veh_veh_dist_rewards_gt[:, 0] * max_veh_veh_distance
            )
            for local_idx, context_idx in enumerate(controlled_indices):
                nearest_dist_by_context_idx[int(context_idx)] = float(
                    nearest_dist_values[local_idx]
                )
                gt_nearest_dist_by_context_idx[int(context_idx)] = float(
                    gt_nearest_dist_values[local_idx]
                )

            veh_veh_dist_rewards_norm = np.clip(
                veh_veh_dist_rewards,
                a_min=0.0,
                a_max=max_veh_veh_distance,
            )
            veh_veh_dist_rewards_norm = (
                veh_veh_dist_rewards_norm / max_veh_veh_distance
            )

            processed_rewards = np.asarray(
                [
                    (
                        np.asarray(vehicle_data_dict[veh_id]["reward"][-1], dtype=np.float32)
                        if vehicle_data_dict[veh_id]["reward"]
                        else np.asarray(adapter._zero_reward_template, dtype=np.float32)
                    )
                    for veh_id in controlled_ids
                ],
                dtype=np.float32,
            )[:, np.newaxis, :]
            processed_rewards = (
                processed_rewards
                * controlled_existence[:, np.newaxis, np.newaxis]
            )
            controlled_ag_data = np.concatenate(
                [
                    controlled_positions,
                    controlled_existence[:, np.newaxis],
                ],
                axis=1,
            )[:, np.newaxis, :]
            controlled_rewards = dataset.compute_rewards(
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
            for controlled_idx, context_idx in enumerate(controlled_indices):
                dense_rewards_by_context_idx[int(context_idx)] = controlled_rewards[
                    controlled_idx,
                    0,
                ]

        for veh_id in step_vehicle_ids:
            context_idx = context_idx_map.get(veh_id)
            if context_idx is None:
                continue
            veh_data = vehicle_data_dict[veh_id]
            veh_data["nearest_dist"].append(
                nearest_dist_by_context_idx.get(context_idx, 0.0)
            )
            veh_data["gt_nearest_dist"].append(
                gt_nearest_dist_by_context_idx.get(context_idx, 0.0)
            )
            veh_data["dense_reward"].append(
                dense_rewards_by_context_idx.get(context_idx, dense_template.copy())
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

        with np.errstate(invalid="ignore"):
            diff = target_positions[:, np.newaxis, :] - all_positions[np.newaxis, :, :]
            squared_dist = np.sum(diff**2, axis=-1)

        valid_all = all_existence.astype(bool)
        squared_dist[:, ~valid_all] = np.inf
        row_idx = np.arange(len(target_positions), dtype=np.int64)
        squared_dist[row_idx, target_all_indices] = np.inf

        nearest = np.sqrt(np.min(squared_dist, axis=1))
        nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)
        nearest = nearest * target_existence

        return nearest[:, np.newaxis].astype(np.float32)
