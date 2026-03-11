"""
负责加载 ctrl-sim 的 Hydra 配置，并生成适配器运行所需的默认参数集合。
该模块封装配置目录定位、override 合并与最小可运行配置构造逻辑。
Loads ctrl-sim Hydra configs and assembles the default settings required by the adapter runtime.
Encapsulates config-path resolution, override merging, and minimal runnable config construction.
"""
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

from ctrlsim_adapter.ctrlsim_path import get_ctrlsim_root

# ctrl-sim config path
CTRL_SIM_ROOT = str(get_ctrlsim_root())
CTRL_SIM_CONFIG_PATH = os.path.join(CTRL_SIM_ROOT, 'cfgs')


def _compose_config_from_dir(
    config_dir: str,
    config_name: str = "config",
) -> DictConfig:
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base=None)
    return compose(config_name=config_name)


def _apply_overrides(
    cfg: DictConfig,
    overrides: Optional[Dict[str, Any]],
) -> DictConfig:
    if not overrides:
        return cfg
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def _default_model_checkpoint_path() -> str:
    from arguments import NOCTURNE_CTRLSIM_DEFAULTS

    return NOCTURNE_CTRLSIM_DEFAULTS['opponent_checkpoint']


def load_ctrl_sim_config(
    checkpoint_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    use_local_paths: bool = False,
) -> DictConfig:
    del use_local_paths

    cfg = _compose_config_from_dir(CTRL_SIM_CONFIG_PATH)

    # 设置 checkpoint 路径
    # Set the checkpoint path.
    if checkpoint_path:
        OmegaConf.update(cfg, "eval.policy.model_path", checkpoint_path)
    return _apply_overrides(cfg, overrides)


def load_ctrl_sim_config_from_yaml(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    从 YAML 文件直接加载配置
    Load config directly from a YAML file.

    Args:
    config_path: YAML 配置文件路径
    config_path: path to the YAML config file.
    overrides: 配置覆盖项
    overrides: config overrides.

    Returns:
    cfg: OmegaConf 配置对象
    cfg: OmegaConf config object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return _apply_overrides(cfg, overrides)


@lru_cache(maxsize=1)
def _load_ctrl_sim_base_config() -> DictConfig:
    """
    加载 ctrl-sim 基础配置（带缓存）
    Load the cached base ctrl-sim config.

    使用 CTRL_SIM_CONFIG_PATH 加载配置，与 ctrl-sim 的模式一致。
    Load the config from CTRL_SIM_CONFIG_PATH using the same pattern as ctrl-sim.

    Returns:
    ctrl-sim 基础配置对象
    Base ctrl-sim config object.
    """
    return _compose_config_from_dir(CTRL_SIM_CONFIG_PATH)


def get_default_opponent_config() -> Dict[str, Any]:
    """
    获取对手策略的默认配置
    Get the default opponent-policy config.

    从 ctrl-sim 的 YAML 配置文件加载，model_path 使用共享默认值。
    Load the config from ctrl-sim YAML and source model_path from the shared defaults.

    Returns:
    config: 默认配置字典
    config: default config dictionary.
    """
    policy_cfg_path = os.path.join(CTRL_SIM_CONFIG_PATH, 'policy/ctrl_sim.yaml')
    if os.path.exists(policy_cfg_path):
        policy_cfg = OmegaConf.load(policy_cfg_path)
        # 使用 resolve=False 避免解析插值（如 model_path 中的 ${eval.policy.run_name}）
        # Use resolve=False to avoid resolving interpolations such as ${eval.policy.run_name} in model_path.
        config = OmegaConf.to_container(policy_cfg, resolve=False)
        
        # 移除 Hydra 元数据和 ctrl-sim 的 model_path（包含无法解析的插值）
        # Remove Hydra metadata and ctrl-sim's model_path, which contains unresolved interpolations.
        config.pop('defaults', None)
        config.pop('model_path', None)
        
        config['model_path'] = _default_model_checkpoint_path()
        
        # 确保 tilting 参数存在（默认为 0）
        # Ensure tilting parameters exist with a default value of 0.
        config.setdefault('goal_tilt', 0)
        config.setdefault('veh_veh_tilt', 0)
        config.setdefault('veh_edge_tilt', 0)
        
        return config
    
    # 回退到默认值
    # Fall back to built-in defaults.
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
        'model_path': _default_model_checkpoint_path(),
    }


def get_default_nocturne_config() -> Dict[str, Any]:
    """
    获取 Nocturne 仿真的默认配置
    Get the default Nocturne simulation config.

    从 ctrl-sim 配置文件加载
    Load it from the ctrl-sim config files.

    Returns:
    config: 默认配置字典
    config: default config dictionary.
    """
    cfg = _load_ctrl_sim_base_config()
    if 'nocturne' in cfg:
        return OmegaConf.to_container(cfg.nocturne, resolve=True)
    
    return {'steps': 90, 'dt': 0.1, 'history_steps': 10, 'collision_fix': True}


def get_default_dataset_config() -> Dict[str, Any]:
    """
    获取数据集的默认配置
    Get the default dataset config.

    从 ctrl-sim 配置文件加载
    Load it from the ctrl-sim config files.

    Returns:
    config: 默认配置字典
    config: default config dictionary.
    """
    cfg = _load_ctrl_sim_base_config()
    if 'dataset' in cfg:
        return OmegaConf.to_container(cfg.dataset, resolve=True)
    
    return {'waymo': {'train_context_length': 32, 'max_num_agents': 24}}


def get_default_model_config() -> Dict[str, Any]:
    """
    获取模型的默认配置
    Get the default model config.

    从 ctrl-sim 配置文件加载
    Load it from the ctrl-sim config files.

    Returns:
    config: 默认配置字典
    config: default config dictionary.
    """
    cfg = _load_ctrl_sim_base_config()
    if 'model' in cfg:
        return OmegaConf.to_container(cfg.model, resolve=True)
    
    # 回退到基础模型配置
    # Fall back to the base model config.
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
    Create a minimal config for quick testing.

    Args:
    checkpoint_path: 模型 checkpoint 路径
    checkpoint_path: model checkpoint path.
    scenario_dir: 场景文件目录
    scenario_dir: scenario file directory.
    preprocess_dir: 预处理数据目录
    preprocess_dir: preprocessed data directory.

    Returns:
    cfg: 最小化配置对象
    cfg: minimal config object.
    """
    # 获取默认对手配置，并用传入的 checkpoint_path 覆盖
    # Get the default opponent config and override it with the provided checkpoint_path.
    opponent_config = get_default_opponent_config()
    # 确保使用传入的路径
    # Ensure the provided path is used.
    opponent_config['model_path'] = checkpoint_path
    
    # 获取数据集配置
    # Get the dataset config.
    dataset_config = get_default_dataset_config()
    
    # 如果没有提供 preprocess_dir，使用临时目录
    # Use a temporary directory when preprocess_dir is not provided.
    if preprocess_dir is None:
        import tempfile
        preprocess_dir = os.path.join(tempfile.gettempdir(), 'dcd_ctrlsim_preprocess')
        os.makedirs(preprocess_dir, exist_ok=True)
    
    # 确保 waymo 子配置存在并设置 preprocess_dir
    # Ensure the waymo sub-config exists and set preprocess_dir.
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
    Config manager for the ctrl-sim config lifecycle.

    支持配置的加载、修改和验证
    Support loading, updating, and validating configs.
    """
    
    def __init__(self, base_config: Optional[DictConfig] = None):
        """
        Args:
        base_config: 基础配置（如果为 None，使用默认配置）
        base_config: base config; if None, use the default config.
        """
        if base_config is not None:
            self.cfg = base_config
        else:
            self.cfg = None
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_local_paths: bool = False,
    ) -> DictConfig:
        """
        加载配置
        Load config.
        """
        self.cfg = load_ctrl_sim_config(
            checkpoint_path=checkpoint_path,
            overrides=overrides,
            use_local_paths=use_local_paths,
        )
        return self.cfg
    
    def update(self, key: str, value: Any):
        """
        更新配置项
        Update a config entry.
        """
        if self.cfg is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        OmegaConf.update(self.cfg, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        Get a config entry.
        """
        if self.cfg is None:
            return default
        try:
            return OmegaConf.select(self.cfg, key, default=default)
        except Exception:
            return default
    
    def validate(self) -> bool:
        """
        验证配置完整性
        Validate config completeness.
        """
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
        """
        转换为普通字典
        Convert to a plain dictionary.
        """
        if self.cfg is None:
            return {}
        return OmegaConf.to_container(self.cfg, resolve=True)
