from typing import Dict, List, Optional

import numpy as np



class OpponentRewardService:
    def __init__(self, adapter):
        self.adapter = adapter

    def _get_step_vehicle_ids(self, t: int, vehicle_data_dict: Dict) -> List[int]:
        veh_ids = self.adapter._state_update_vehicle_ids_step
        if not veh_ids:
            veh_ids = self.adapter._all_vehicle_ids
        if not veh_ids:
            veh_ids = list(vehicle_data_dict.keys())
        return [
            veh_id
            for veh_id in veh_ids
            if veh_id in vehicle_data_dict and len(vehicle_data_dict[veh_id]["position"]) > t
        ]

    def _compute_nearest_dist_all(self, t: int, vehicle_data_dict: Dict) -> Dict:
        """
        计算车-车最近距离（对齐 ctrl-sim evaluator.py compute_nearest_dist_all）
        """
        veh_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not veh_ids:
            return vehicle_data_dict

        veh_data_list = [vehicle_data_dict[veh_id] for veh_id in veh_ids]
        ag_data_xy_exist = np.array(
            [
                [
                    veh_data["position"][t]["x"],
                    veh_data["position"][t]["y"],
                    veh_data["existence"][t],
                ]
                for veh_data in veh_data_list
            ]
        )[:, np.newaxis, :]
        all_existence = ag_data_xy_exist[:, 0, 2]
        existence_scale = all_existence[:, np.newaxis].astype(float)

        veh_veh_dist_rewards = (
            self.adapter.dataset.compute_dist_to_nearest_vehicle_rewards(
                ag_data_xy_exist,
                normalize=False,
            )
            * existence_scale
        )

        gt_ag_data = np.array(
            [
                [
                    veh_data["gt_position"][t]["x"],
                    veh_data["gt_position"][t]["y"],
                    veh_data["existence"][t],
                ]
                for veh_data in veh_data_list
            ]
        )[:, np.newaxis, :]
        veh_veh_dist_rewards_gt = (
            self.adapter.dataset.compute_dist_to_nearest_vehicle_rewards(
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
        self,
        t: int,
        vehicle_data_dict: Dict,
    ) -> Dict:
        """
        计算 dense reward（仅对受控车辆）

        参考: evaluator.py 第 127-170 行 compute_dense_reward()
        """
        veh_ids = self._get_step_vehicle_ids(t, vehicle_data_dict)
        if not veh_ids:
            return vehicle_data_dict

        veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(veh_ids)}
        controlled_ids = [
            veh_id
            for veh_id in self.adapter._controlled_vehicle_ids_step
            if veh_id in veh_id_to_idx
        ]
        controlled_indices = np.asarray(
            [veh_id_to_idx[veh_id] for veh_id in controlled_ids],
            dtype=np.int64,
        )
        veh_data_list = [vehicle_data_dict[veh_id] for veh_id in veh_ids]
        all_positions = np.array(
            [
                [
                    veh_data["position"][t]["x"],
                    veh_data["position"][t]["y"],
                ]
                for veh_data in veh_data_list
            ],
            dtype=np.float32,
        )
        all_gt_positions = np.array(
            [
                [
                    veh_data["gt_position"][t]["x"],
                    veh_data["gt_position"][t]["y"],
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

        cfg_dataset = self.adapter.cfg.dataset.waymo
        dense_template = np.zeros(
            self.adapter.cfg.model.num_reward_components,
            dtype=np.float32,
        )

        if controlled_ids:
            controlled_positions = all_positions[controlled_indices]
            controlled_gt_positions = all_gt_positions[controlled_indices]
            controlled_existence = all_existence[controlled_indices]

            if len(self.adapter._road_edge_polylines) > 0:
                controlled_xy = controlled_positions[:, np.newaxis, :]
                veh_edge_dist_rewards = (
                    self.adapter.dataset.compute_dist_to_nearest_road_edge_rewards(
                        controlled_xy,
                        self.adapter._road_edge_polylines,
                    )
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
            nearest_dist_values[controlled_indices] = (
                veh_veh_dist_rewards[:, 0] * max_veh_veh_distance
            )
            gt_nearest_dist_values[controlled_indices] = (
                veh_veh_dist_rewards_gt[:, 0] * max_veh_veh_distance
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
                [vehicle_data_dict[veh_id]["reward"][0] for veh_id in controlled_ids]
            )[:, np.newaxis, :]
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
            controlled_rewards = self.adapter.dataset.compute_rewards(
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
            for controlled_idx, all_idx in enumerate(controlled_indices):
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
