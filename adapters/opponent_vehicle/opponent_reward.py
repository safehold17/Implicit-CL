from typing import Any, Dict, List

import numpy as np

_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0


def _dot_product_2d_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs[..., 0] * rhs[..., 0] + lhs[..., 1] * rhs[..., 1]


def _cross_product_2d_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def _compute_signed_distance_to_polyline_np(
    xys: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    is_cyclic = (
        np.square(polyline[0] - polyline[-1]).sum()
        < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
    )
    xy_starts = polyline[None, :-1, :2]
    xy_ends = polyline[None, 1:, :2]
    start_to_point = xys[:, None, :2] - xy_starts
    start_to_end = xy_ends - xy_starts

    rel_t = np.nan_to_num(
        _dot_product_2d_np(start_to_point, start_to_end)
        / _dot_product_2d_np(start_to_end, start_to_end)
    )
    n = np.sign(_cross_product_2d_np(start_to_point, start_to_end))
    distance_to_segment = np.linalg.norm(
        start_to_point - (start_to_end * np.clip(rel_t, 0.0, 1.0)[..., None]),
        axis=-1,
    )

    start_to_end_padded = np.concatenate(
        [start_to_end[:, -1:], start_to_end, start_to_end[:, :1]],
        axis=1,
    )
    is_locally_convex = (
        _cross_product_2d_np(
            start_to_end_padded[:, :-1],
            start_to_end_padded[:, 1:],
        )
        > 0.0
    )
    n_prior = np.concatenate(
        [np.where(is_cyclic, n[:, -1:], n[:, :1]), n[:, :-1]],
        axis=-1,
    )
    n_next = np.concatenate(
        [n[:, 1:], np.where(is_cyclic, n[:, :1], n[:, -1:])],
        axis=-1,
    )
    sign_if_before = np.where(
        is_locally_convex[:, :-1],
        np.maximum(n, n_prior),
        np.minimum(n, n_prior),
    )
    sign_if_after = np.where(
        is_locally_convex[:, 1:],
        np.maximum(n, n_next),
        np.minimum(n, n_next),
    )
    sign_to_segment = np.where(
        rel_t < 0.0,
        sign_if_before,
        np.where(rel_t < 1.0, n, sign_if_after),
    )
    min_segment_idx = np.argmin(distance_to_segment, axis=-1)
    distance_sign = np.take_along_axis(
        sign_to_segment,
        min_segment_idx[:, None],
        axis=1,
    )[:, 0]
    return distance_sign * np.min(distance_to_segment, axis=-1)


def _compute_signed_distance_to_polylines_np(
    xys: np.ndarray,
    polylines: tuple[np.ndarray, ...],
) -> np.ndarray:
    distances = [
        _compute_signed_distance_to_polyline_np(xys, polyline)
        for polyline in polylines
        if len(polyline) >= 2
    ]
    if not distances:
        return np.zeros((len(xys),), dtype=np.float32)
    stacked = np.stack(distances, axis=-1)
    nearest_idx = np.argmin(np.abs(stacked), axis=-1)
    return np.take_along_axis(stacked, nearest_idx[:, None], axis=1)[:, 0]


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
        计算 dense reward

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

        if target_vehicle_ids:
            target_positions = all_positions[target_indices]
            target_gt_positions = all_gt_positions[target_indices]
            target_existence = all_existence[target_indices]

            veh_edge_dist_rewards = np.zeros(
                (len(target_vehicle_ids), 1),
                dtype=np.float32,
            )
            if adapter._road_edge_polylines_cpu:
                dynamic_target_indices: List[int] = []
                dynamic_target_positions: List[np.ndarray] = []
                constant_road_edge_reward_by_id = (
                    adapter._constant_road_edge_reward_by_id
                )
                for local_idx, veh_id in enumerate(target_vehicle_ids):
                    cached_reward = constant_road_edge_reward_by_id.get(veh_id)
                    if cached_reward is not None:
                        veh_edge_dist_rewards[local_idx, 0] = cached_reward
                        continue
                    dynamic_target_indices.append(local_idx)
                    dynamic_target_positions.append(target_positions[local_idx])

                if dynamic_target_positions:
                    dynamic_rewards = self._compute_road_edge_rewards(
                        np.asarray(dynamic_target_positions, dtype=np.float32),
                    )
                    veh_edge_dist_rewards[
                        np.asarray(dynamic_target_indices, dtype=np.int64),
                        0,
                    ] = dynamic_rewards[:, 0]
                veh_edge_dist_rewards *= target_existence[:, np.newaxis]

            veh_veh_dist_rewards = self._compute_nearest_dist_to_all(
                target_positions=target_positions,
                all_positions=all_positions,
                all_existence=all_existence,
                target_existence=target_existence,
                target_all_indices=target_indices,
            )
            veh_veh_dist_rewards_gt = self._compute_nearest_dist_to_all(
                target_positions=target_gt_positions,
                all_positions=all_gt_positions,
                all_existence=all_existence,
                target_existence=target_existence,
                target_all_indices=target_indices,
            )

            max_veh_veh_distance = cfg_dataset.max_veh_veh_distance
            nearest_dist_values = veh_veh_dist_rewards[:, 0] * max_veh_veh_distance
            gt_nearest_dist_values = (
                veh_veh_dist_rewards_gt[:, 0] * max_veh_veh_distance
            )
            for local_idx, context_idx in enumerate(target_indices):
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
            for target_idx, context_idx in enumerate(target_indices):
                dense_rewards_by_context_idx[int(context_idx)] = target_rewards[
                    target_idx,
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
