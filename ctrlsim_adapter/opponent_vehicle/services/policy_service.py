"""
负责构建对手控制所需的模型、数据集与 policy 实例。
该模块集中处理 checkpoint 加载、tilt 参数注入以及 policy 构造参数的拼装。
Builds the model, dataset, and policy instances needed for opponent control.
Centralizes checkpoint loading, tilt injection, and policy-construction argument assembly.
"""

from typing import Dict, Tuple

from datasets.rl_waymo.dataset_ctrl_sim import RLWaymoDatasetCtRLSim
from models.ctrl_sim import CtRLSim
from policies.autoregressive_policy import AutoregressivePolicy

from ..tilting import PerVehicleAutoregressivePolicy, TiltConfig


class _DummyModel:
    """
    替代 GPU 模型，仅提供 Policy.__init__ 需要的 cfg 属性和 eval() 方法。
    Replacement for the GPU model that only provides the cfg attribute and eval() method required by Policy.__init__.
    """

    def __init__(self, checkpoint_cfg):
        self.cfg = checkpoint_cfg

    def eval(self):
        return self


class OpponentPolicyService:
    def __init__(self, adapter):
        self.adapter = adapter

    def _build_per_vehicle_tilt_dict(
        self,
        mapping: Dict[int, Tuple[int, int, int]],
    ) -> Dict:
        return {
            "tilt": True,
            "goal_tilt": self.adapter.current_tilt.goal_tilt,
            "veh_veh_tilt": self.adapter.current_tilt.veh_veh_tilt,
            "veh_edge_tilt": self.adapter.current_tilt.veh_edge_tilt,
            "per_vehicle": mapping,
        }

    def _build_policy_kwargs(self, model, tilt_dict: Dict) -> Dict:
        return {
            "cfg": self.adapter.cfg,
            "model_path": self.adapter.checkpoint_path,
            "model": model,
            "use_rtg": True,
            "predict_rtgs": True,
            "discretize_rtgs": True,
            "real_time_rewards": True,
            "privileged_return": False,
            "max_return": False,
            "min_return": False,
            "key_dict": {
                "next_acceleration": "next_acceleration",
                "next_steering": "next_steering",
                "rtgs": "rtgs",
            },
            "tilt_dict": tilt_dict,
            "name": "ctrl_sim",
            "action_temperature": self.adapter.action_temperature,
            "nucleus_sampling": self.adapter.nucleus_sampling,
            "nucleus_threshold": self.adapter.nucleus_threshold,
            "device": self.adapter.device,
        }

    def _sync_policy_tilt(
        self,
        tilt_dict: Dict,
        goal_tilt: int,
        veh_veh_tilt: int,
        veh_edge_tilt: int,
    ) -> None:
        if self.adapter._policy is None:
            return
        self.adapter._policy.tilt_dict = tilt_dict
        self.adapter._policy.goal_tilt = goal_tilt
        self.adapter._policy.veh_veh_tilt = veh_veh_tilt
        self.adapter._policy.veh_edge_tilt = veh_edge_tilt

    def _load_checkpoint_cfg(self):
        """
        从 checkpoint 中只读取 cfg，不加载模型权重到 GPU。
        Read only cfg from the checkpoint without loading model weights onto the GPU.
        """
        if self.adapter._checkpoint_cfg is not None:
            return
        tmp = CtRLSim.load_from_checkpoint(self.adapter.checkpoint_path, map_location="cpu")
        self.adapter._checkpoint_cfg = tmp.cfg
        del tmp

    def _validate_external_cfg_compatibility(self):
        """
        校验 checkpoint cfg 与 adapter cfg 的关键字段一致性。
        Validate that key fields stay consistent between checkpoint cfg and adapter cfg.
        """
        if self.adapter._checkpoint_cfg is None:
            return
        ckpt_ds = self.adapter._checkpoint_cfg.dataset.waymo
        ckpt_model = self.adapter._checkpoint_cfg.model
        adapter_ds = self.adapter.cfg.dataset.waymo
        checks = [
            (
                "train_context_length",
                getattr(ckpt_ds, "train_context_length", None),
                getattr(adapter_ds, "train_context_length", None),
            ),
            (
                "rtg_discretization",
                getattr(ckpt_ds, "rtg_discretization", None),
                getattr(adapter_ds, "rtg_discretization", None),
            ),
            (
                "accel_discretization",
                getattr(ckpt_ds, "accel_discretization", None),
                getattr(adapter_ds, "accel_discretization", None),
            ),
            (
                "steer_discretization",
                getattr(ckpt_ds, "steer_discretization", None),
                getattr(adapter_ds, "steer_discretization", None),
            ),
            (
                "max_num_agents",
                getattr(ckpt_model, "max_num_agents", None),
                getattr(self.adapter.cfg.model, "max_num_agents", None),
            ),
            (
                "num_reward_components",
                getattr(ckpt_model, "num_reward_components", None),
                getattr(self.adapter.cfg.model, "num_reward_components", None),
            ),
        ]
        for name, ckpt_val, adapter_val in checks:
            if (
                ckpt_val is not None
                and adapter_val is not None
                and ckpt_val != adapter_val
            ):
                raise ValueError(
                    f"External teacher cfg mismatch: {name} "
                    f"checkpoint={ckpt_val} vs adapter={adapter_val}"
                )

    def _load_dataset_only(self):
        """
        仅加载 dataset，不加载模型。
        Load only the dataset without loading the model.
        """
        self.adapter.dataset = RLWaymoDatasetCtRLSim(
            self.adapter.cfg,
            split_name="test",
            mode="eval",
        )

    def _load_model_and_dataset(self):
        self._load_checkpoint_cfg()
        self._validate_external_cfg_compatibility()
        self._load_dataset_only()

    def _create_policy(self) -> AutoregressivePolicy:
        """
        创建 AutoregressivePolicy 实例
        Create an AutoregressivePolicy instance.

        参考: eval_sim.py 第 42-66 行
        See eval_sim.py lines 42-66.
        """
        if self.adapter._checkpoint_cfg is None:
            self._load_checkpoint_cfg()
            self._validate_external_cfg_compatibility()
        model = _DummyModel(self.adapter._checkpoint_cfg)
        action_dim = (
            int(self.adapter.cfg.dataset.waymo.accel_discretization)
            * int(self.adapter.cfg.dataset.waymo.steer_discretization)
        )

        if self.adapter.per_vehicle_tilting is not None:
            tilt_dict = self._build_per_vehicle_tilt_dict(self.adapter.per_vehicle_tilting)
            policy = PerVehicleAutoregressivePolicy(
                **self._build_policy_kwargs(model=model, tilt_dict=tilt_dict),
            )
            policy.action_dim = action_dim
            return policy

        policy = AutoregressivePolicy(
            **self._build_policy_kwargs(
                model=model,
                tilt_dict=self.adapter.current_tilt.to_dict(),
            ),
        )
        policy.action_dim = action_dim
        return policy

    def set_tilting(
        self,
        goal_tilt: int,
        veh_veh_tilt: int,
        veh_edge_tilt: int,
    ):
        """
        设置 domain tilting 参数
        Set domain-level tilting parameters.

        注：tilting 通过修改 RTG 预测的 logits 来实现
        Note: tilting is implemented by modifying the logits of RTG predictions.
        参考: datasets/rl_waymo/dataset.py 第 347-352 行 get_tilt_logits()
        See datasets/rl_waymo/dataset.py lines 347-352 in get_tilt_logits().

        Args:
        goal_tilt: 目标导向程度 [-25, 25]
        goal_tilt: goal-seeking strength in [-25, 25].
        veh_veh_tilt: 车-车交互激进度 [-25, 25]
        veh_veh_tilt: aggressiveness of vehicle-to-vehicle interaction in [-25, 25].
        veh_edge_tilt: 车-边界交互激进度 [-25, 25]
        veh_edge_tilt: aggressiveness of vehicle-to-edge interaction in [-25, 25].
        """
        self.adapter.current_tilt = TiltConfig(
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
        )
        self._sync_policy_tilt(
            tilt_dict=self.adapter.current_tilt.to_dict(),
            goal_tilt=goal_tilt,
            veh_veh_tilt=veh_veh_tilt,
            veh_edge_tilt=veh_edge_tilt,
        )

    def set_tilting_from_tuple(self, tilt: Tuple[int, int, int]):
        """
        从元组设置 tilting（便捷接口）
        Set tilting from a tuple as a convenience helper.
        """
        self.set_tilting(tilt[0], tilt[1], tilt[2])

    def set_per_vehicle_tilting(self, mapping: Dict[int, Tuple[int, int, int]]):
        """
        设置 per-vehicle tilting
        Set per-vehicle tilting.

        Args:
        mapping: {veh_id: (goal_tilt, veh_veh_tilt, veh_edge_tilt)} 映射字典
        mapping: mapping dict of {veh_id: (goal_tilt, veh_veh_tilt, veh_edge_tilt)}.
        """
        self.adapter.per_vehicle_tilting = mapping
        self._sync_policy_tilt(
            tilt_dict=self._build_per_vehicle_tilt_dict(mapping),
            goal_tilt=self.adapter.current_tilt.goal_tilt,
            veh_veh_tilt=self.adapter.current_tilt.veh_veh_tilt,
            veh_edge_tilt=self.adapter.current_tilt.veh_edge_tilt,
        )
