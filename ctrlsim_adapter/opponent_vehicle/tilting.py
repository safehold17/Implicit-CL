from dataclasses import dataclass
from typing import Dict, Tuple

from policies.autoregressive_policy import AutoregressivePolicy

from ctrlsim_adapter.ctrlsim_discretization import (
    decode_predicted_rtg as _decode_predicted_rtg,
)
from ctrlsim_adapter.ctrlsim_discretization import (
    get_tilt_logits as _get_tilt_logits,
)


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

    goal_tilt: int = 0
    veh_veh_tilt: int = 0
    veh_edge_tilt: int = 0

    def __post_init__(self):
        """验证参数范围"""
        for name, val in [
            ("goal_tilt", self.goal_tilt),
            ("veh_veh_tilt", self.veh_veh_tilt),
            ("veh_edge_tilt", self.veh_edge_tilt),
        ]:
            if not (-25 <= val <= 25):
                raise ValueError(f"{name} must be in [-25, 25], got {val}")

    def to_dict(self) -> Dict:
        """转换为 ctrl-sim 期望的 tilt_dict 格式"""
        return {
            "tilt": True,
            "goal_tilt": self.goal_tilt,
            "veh_veh_tilt": self.veh_veh_tilt,
            "veh_edge_tilt": self.veh_edge_tilt,
        }

    @classmethod
    def from_tuple(cls, tilt_tuple: Tuple[int, int, int]) -> "TiltConfig":
        """从元组创建"""
        goal_tilt, veh_veh_tilt, veh_edge_tilt = tilt_tuple
        return cls(
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
        )


class PerVehicleAutoregressivePolicy(AutoregressivePolicy):
    """
    Per-vehicle tilting policy subclass

    覆写 process_predicted_rtg 方法以支持每个车辆使用独立的 tilt
    """

    def process_predicted_rtg(
        self,
        rtg_logits,
        token_index,
        veh_id,
        dset,
        vehicle_data_dict,
        data,
        agent_idx_dict,
        is_tilted=False,
        device="cuda",
    ):
        """
        处理预测的 RTG，应用 per-vehicle tilting

        使用 ctrlsim_discretization 中的纯函数，
        替代直接调用 dset.get_tilt_logits() / dset.undiscretize_rtgs()。
        """
        cfg_rl_waymo = self.cfg_rl_waymo
        idx = agent_idx_dict[self.veh_id_to_idx[veh_id]]

        rtg_logits_3 = rtg_logits[0, idx, token_index].reshape(
            cfg_rl_waymo.rtg_discretization,
            self.cfg_model.num_reward_components,
        )

        tilt_dict = self.tilt_dict
        if is_tilted and tilt_dict.get("tilt", False):
            per_vehicle_map = tilt_dict.get("per_vehicle", {})
            g, v, e = per_vehicle_map.get(
                veh_id,
                (
                    getattr(self, "goal_tilt", 0),
                    getattr(self, "veh_veh_tilt", 0),
                    getattr(self, "veh_edge_tilt", 0),
                ),
            )
        else:
            g, v, e = 0, 0, 0

        rtg_discretization = cfg_rl_waymo.rtg_discretization
        tilt_logits_np = _get_tilt_logits(rtg_discretization, g, v, e)

        (goal_idx, veh_idx, road_idx), (goal_val, veh_val, road_val) = (
            _decode_predicted_rtg(
                rtg_logits_3,
                tilt_logits_np,
                rtg_discretization,
                cfg_rl_waymo.min_rtg_pos,
                cfg_rl_waymo.max_rtg_pos,
                cfg_rl_waymo.min_rtg_veh,
                cfg_rl_waymo.max_rtg_veh,
                cfg_rl_waymo.min_rtg_road,
                cfg_rl_waymo.max_rtg_road,
                device=device,
            )
        )

        vehicle_data_dict[veh_id]["next_rtg_goal"] = goal_val
        vehicle_data_dict[veh_id]["next_rtg_veh"] = veh_val
        vehicle_data_dict[veh_id]["next_rtg_road"] = road_val

        data["agent"].rtgs[0, idx, token_index, 0] = goal_idx
        data["agent"].rtgs[0, idx, token_index, 1] = veh_idx
        data["agent"].rtgs[0, idx, token_index, 2] = road_idx

        next_rtgs = [goal_idx, veh_idx, road_idx]

        return vehicle_data_dict, data, next_rtgs
