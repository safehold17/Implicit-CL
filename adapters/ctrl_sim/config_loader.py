"""
ctrl-sim config loader

Load and manage Hydra configs for ctrl-sim

Design reference:
- get CONFIG_PATH from cfgs/dcd_config.py
- configs are combined through Hydra's defaults mechanism

References:
- third_party/ctrl-sim/eval_sim.py: config loading example
- third_party/ctrl-sim/cfgs/config.py: CONFIG_PATH
- cfgs/dcd_config.py: DCD project's CONFIG_PATH
- cfgs/data/ctrl_sim.yaml: local path overrides
"""
import os
import sys
from functools import lru_cache
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

# import CONFIG_PATH from DCD config module (same as ctrl-sim's mode)
from cfgs.dcd_config import CONFIG_PATH as DCD_CONFIG_PATH

# ctrl-sim config path
CTRL_SIM_ROOT = os.path.join(os.path.dirname(__file__), '../../third_party/ctrl-sim')
CTRL_SIM_CONFIG_PATH = os.path.join(CTRL_SIM_ROOT, 'cfgs')


@lru_cache(maxsize=1)
def _load_dcd_config() -> DictConfig:
    """
    加载 DCD 项目配置（带缓存）
    
    使用 DCD_CONFIG_PATH（从 cfgs/dcd_config.py 导入）加载配置，
    
    Returns:
        DCD 项目配置对象（包含 dcd_config_path 等）
    """
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=DCD_CONFIG_PATH, version_base=None)
    return compose(config_name="config")


def _apply_local_path_overrides(cfg: DictConfig, local_cfg: DictConfig) -> DictConfig:
    """
    应用本地路径覆盖配置
    
    将 cfgs/data/ctrl_sim.yaml 中的路径映射到 ctrl-sim 配置结构
    
    Args:
        cfg: ctrl-sim 基础配置
        local_cfg: 本地路径配置（从 cfgs/data/ctrl_sim.yaml 加载）
    
    Returns:
        更新后的配置对象
    """
    if 'ctrl_sim' not in local_cfg:
        return cfg
    
    local = local_cfg.ctrl_sim
    
    # 路径映射：(源路径, 目标路径)
    mappings = [
        # 顶层路径
        ('dataset_root', 'dataset_root'),
        ('project_root', 'project_root'),
        ('nocturne_waymo_data_folder', 'nocturne_waymo_data_folder'),
        ('nocturne_waymo_train_folder', 'nocturne_waymo_train_folder'),
        ('nocturne_waymo_val_folder', 'nocturne_waymo_val_folder'),
        ('nocturne_waymo_val_interactive_folder', 'nocturne_waymo_val_interactive_folder'),
        ('preprocess_dir', 'dataset.waymo.preprocess_dir'),
        ('simulated_dataset', 'dataset.waymo.simulated_dataset'),
        ('simulated_dataset_preprocessed_dir', 'dataset.waymo.simulated_dataset_preprocessed_dir'),
        # offline_rl 路径
        ('offline_rl.dataset_path', 'dataset.waymo.dataset_path'),
        ('offline_rl.output_data_folder_train', 'offline_rl.output_data_folder_train'),
        ('offline_rl.output_data_folder_val', 'offline_rl.output_data_folder_val'),
        ('offline_rl.output_data_folder_val_interactive', 'offline_rl.output_data_folder_val_interactive'),
    ]
    
    for src, dest in mappings:
        value = OmegaConf.select(local, src)
        if value is not None:
            OmegaConf.update(cfg, dest, value)
    
    return cfg


def load_ctrl_sim_config(
    checkpoint_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    use_local_paths: bool = True,
) -> DictConfig:

    # 加载 ctrl-sim 基础配置
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CTRL_SIM_CONFIG_PATH, version_base=None)
    cfg = compose(config_name="config")
    
    # 应用本地路径覆盖
    if use_local_paths:
        try:
            # 从 DCD 配置获取 dcd_config_path（参考 ctrl-sim 的模式）
            dcd_cfg = _load_dcd_config()
            local_cfg_path = os.path.join(dcd_cfg.dcd_config_path, "ctrl_sim.yaml")
            
            if os.path.exists(local_cfg_path):
                local_cfg = OmegaConf.load(local_cfg_path)
                cfg = _apply_local_path_overrides(cfg, local_cfg)
        except Exception as e:
            print(f"Warning: Failed to load local path config: {e}")
    
    # 设置 checkpoint 路径
    if checkpoint_path:
        OmegaConf.update(cfg, "eval.policy.model_path", checkpoint_path)
    
    # 应用用户自定义覆盖项
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)
    
    return cfg


def load_ctrl_sim_config_from_yaml(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    从 YAML 文件直接加载配置
    
    Args:
        config_path: YAML 配置文件路径
        overrides: 配置覆盖项
    
    Returns:
        cfg: OmegaConf 配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)
    
    return cfg


@lru_cache(maxsize=1)
def _load_ctrl_sim_base_config() -> DictConfig:
    """
    加载 ctrl-sim 基础配置（带缓存）
    
    使用 CTRL_SIM_CONFIG_PATH 加载配置，与 ctrl-sim 的模式一致。
    
    Returns:
        ctrl-sim 基础配置对象
    """
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CTRL_SIM_CONFIG_PATH, version_base=None)
    return compose(config_name="config")


def get_default_opponent_config() -> Dict[str, Any]:
    """
    获取对手策略的默认配置
    
    从 ctrl-sim 的 YAML 配置文件加载，model_path 从 DCD 配置获取
    
    Returns:
        config: 默认配置字典
    """
    policy_cfg_path = os.path.join(CTRL_SIM_CONFIG_PATH, 'policy/ctrl_sim.yaml')
    if os.path.exists(policy_cfg_path):
        policy_cfg = OmegaConf.load(policy_cfg_path)
        # 使用 resolve=False 避免解析插值（如 model_path 中的 ${eval.policy.run_name}）
        config = OmegaConf.to_container(policy_cfg, resolve=False)
        
        # 移除 Hydra 元数据和 ctrl-sim 的 model_path（包含无法解析的插值）
        config.pop('defaults', None)
        config.pop('model_path', None)
        
        # 从 DCD 配置获取 model_path（在 cfgs/config.yaml 中定义）
        try:
            dcd_cfg = _load_dcd_config()
            config['model_path'] = dcd_cfg.model_path
        except Exception:
            # 回退到默认路径
            config['model_path'] = os.path.join(
                os.path.dirname(__file__), '../../checkpoints/model.ckpt'
            )
        
        # 确保 tilting 参数存在（默认为 0）
        config.setdefault('goal_tilt', 0)
        config.setdefault('veh_veh_tilt', 0)
        config.setdefault('veh_edge_tilt', 0)
        
        return config
    
    # 回退到默认值
    return {
        'use_rtg': True,
        'predict_rtgs': True,
        'discretize_rtgs': True,
        'real_time_rewards': True,
        'action_temperature': 1.0,
        'nucleus_sampling': False,
        'nucleus_threshold': 0.8,
        'goal_tilt': 0,
        'veh_veh_tilt': 0,
        'veh_edge_tilt': 0,
        'model_path': os.path.join(
            os.path.dirname(__file__), '../../checkpoints/model.ckpt'
        ),
    }


def get_default_nocturne_config() -> Dict[str, Any]:
    """
    获取 Nocturne 仿真的默认配置
    
    从 ctrl-sim 配置文件加载
    
    Returns:
        config: 默认配置字典
    """
    cfg = _load_ctrl_sim_base_config()
    if 'nocturne' in cfg:
        return OmegaConf.to_container(cfg.nocturne, resolve=True)
    
    return {'steps': 90, 'dt': 0.1, 'history_steps': 10, 'collision_fix': True}


def get_default_dataset_config() -> Dict[str, Any]:
    """
    获取数据集的默认配置
    
    从 ctrl-sim 配置文件加载
    
    Returns:
        config: 默认配置字典
    """
    cfg = _load_ctrl_sim_base_config()
    if 'dataset' in cfg:
        return OmegaConf.to_container(cfg.dataset, resolve=True)
    
    return {'waymo': {'train_context_length': 32, 'max_num_agents': 24}}


def get_default_model_config() -> Dict[str, Any]:
    """
    获取模型的默认配置
    
    从 ctrl-sim 配置文件加载
    
    Returns:
        config: 默认配置字典
    """
    cfg = _load_ctrl_sim_base_config()
    if 'model' in cfg:
        return OmegaConf.to_container(cfg.model, resolve=True)
    
    # 回退到基础模型配置
    return {
        'hidden_dim': 256,
        'map_attr': 3,
        'num_road_types': 8,
        'no_actions': False,
        'num_heads': 8,
        'num_reward_components': 3,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'state_dim': 12,
        'use_map': True,
        'goal_dropout': 0.1,
        'max_pool_map': True,
        'supervise_moving': True,
        'predict_rtg': True,
        'attend_own_return_action': False,
        'trajeglish': False,
        'il': False,
        'ctg_plus_plus': False,
        'decision_transformer': False,
        'num_transformer_encoder_layers': 2,
        'num_decoder_layers': 4,
        'predict_future_states': True,
        'local_frame_predictions': False,
        'loss_action_coef': 1.0,
        'encode_initial_state': True,
    }


def create_minimal_config(
    checkpoint_path: str,
    scenario_dir: str,
    preprocess_dir: Optional[str] = None,
) -> DictConfig:
    """
    创建最小化配置（用于快速测试）
    
    Args:
        checkpoint_path: 模型 checkpoint 路径
        scenario_dir: 场景文件目录
        preprocess_dir: 预处理数据目录
    
    Returns:
        cfg: 最小化配置对象
    """
    # 获取默认对手配置，并用传入的 checkpoint_path 覆盖
    opponent_config = get_default_opponent_config()
    opponent_config['model_path'] = checkpoint_path  # 确保使用传入的路径
    
    # 获取数据集配置
    dataset_config = get_default_dataset_config()
    
    # 如果没有提供 preprocess_dir，使用临时目录
    if preprocess_dir is None:
        import tempfile
        preprocess_dir = os.path.join(tempfile.gettempdir(), 'dcd_ctrlsim_preprocess')
        os.makedirs(preprocess_dir, exist_ok=True)
    
    # 确保 waymo 子配置存在并设置 preprocess_dir
    if 'waymo' not in dataset_config:
        dataset_config['waymo'] = {}
    dataset_config['waymo']['preprocess_dir'] = preprocess_dir
    
    config = {
        'nocturne': get_default_nocturne_config(),
        'dataset': dataset_config,
        'model': get_default_model_config(),
        'eval': {
            'policy': opponent_config
        },
        'nocturne_waymo_val_folder': scenario_dir,
        'dataset_root': os.path.dirname(scenario_dir),
    }
    
    return OmegaConf.create(config)


class ConfigManager:
    """
    配置管理器：管理 ctrl-sim 配置的生命周期
    
    支持配置的加载、修改和验证
    """
    
    def __init__(self, base_config: Optional[DictConfig] = None):
        """
        Args:
            base_config: 基础配置（如果为 None，使用默认配置）
        """
        if base_config is not None:
            self.cfg = base_config
        else:
            self.cfg = None
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_local_paths: bool = True,
    ) -> DictConfig:
        """加载配置"""
        self.cfg = load_ctrl_sim_config(
            checkpoint_path=checkpoint_path,
            overrides=overrides,
            use_local_paths=use_local_paths,
        )
        return self.cfg
    
    def update(self, key: str, value: Any):
        """更新配置项"""
        if self.cfg is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        OmegaConf.update(self.cfg, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if self.cfg is None:
            return default
        try:
            return OmegaConf.select(self.cfg, key, default=default)
        except Exception:
            return default
    
    def validate(self) -> bool:
        """验证配置完整性"""
        if self.cfg is None:
            return False
        
        required_keys = [
            'nocturne.steps',
            'nocturne.dt',
            'dataset.waymo.train_context_length',
            'dataset.waymo.max_num_agents',
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required config key: {key}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        if self.cfg is None:
            return {}
        return OmegaConf.to_container(self.cfg, resolve=True)
