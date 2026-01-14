"""
Nocturne + CtRL-Sim 对抗环境

实现 DCD 框架要求的环境接口，支持：
- PLR (Prioritized Level Replay) 机制
- PAIRED / ACCEL 等 UED 算法（通过 Adversary 接口）
- 动态场景池大小
- Level 变异和编辑
"""
import gym
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

from .level import ScenarioLevel
from .video_recorder import NocturneVideoRecorder
from util.build_scenario_index import ScenarioIndex
from adapters.ctrl_sim import (
    CtrlSimOpponentAdapter,
    DataBridge,
    load_ctrl_sim_config,
    create_minimal_config,
)


# ========== Level 参数范围定义 ==========
# 参考 BipedalWalker 的 PARAM_RANGES_FULL
# 这些是默认值，可通过配置或构造函数参数覆盖
DEFAULT_TILT_RANGE = [-25.0, 25.0]  # tilting 参数范围
DEFAULT_TILT_MUTATION_STD = 5.0  # 变异时的扰动幅度（与 config.yaml 一致）
DEFAULT_OBS_DIM = 128  # 观测维度
DEFAULT_ACTION_DIM = 2  # 动作维度（accel, steer）

# 默认 level 参数向量：[scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt]
DEFAULT_LEVEL_PARAMS = [0, 0.0, 0.0, 0.0]


def rand_int_seed():
    """生成随机种子（32 位无符号整数）"""
    import os
    # 生成 4 字节（32 位）的随机数
    return int.from_bytes(os.urandom(4), byteorder="little")


class NocturneCtrlSimAdversarial(gym.Env):
    """
    DCD 对抗环境：Nocturne 场景 + CtRL-Sim 对手
    
    支持两种使用模式：
    
    1. **PAIRED/ACCEL 模式**（环境 adversary 构建）：
       - 调用 reset() 初始化 adversary 构建流程
       - 调用 step_adversary() 逐步构建 level
       - 构建完成后调用 reset_agent() 让 student 开始训练
    
    2. **DR/PLR 模式**（直接采样）：
       - 调用 reset_random() 随机生成 level
       - 或调用 reset_to_level() 加载指定 level
    
    Adversary 动作空间（4步构建）：
    - Step 0: 选择 scenario_id（离散：映射到场景池索引）
    - Step 1: 设置 goal_tilt（连续：[-1, 1] -> [-25, 25]）
    - Step 2: 设置 veh_veh_tilt（连续：[-1, 1] -> [-25, 25]）
    - Step 3: 设置 veh_edge_tilt（连续：[-1, 1] -> [-25, 25]）
    
    与 Milestone 1/2 的集成：
    - 使用 ScenarioLevel 数据结构（Milestone 1）
    - 使用 CtrlSimOpponentAdapter 和 DataBridge（Milestone 2）
    """
    
    def __init__(
        self,
        scenario_index_path: str,
        opponent_checkpoint: str,
        scenario_data_dir: str,
        preprocess_dir: str,
        opponent_k: int = 7,
        max_episode_steps: int = 90,
        device: str = 'cuda',
        cfg: Any = None,
        seed: int = 0,
        fixed_environment: bool = False,
        # Adversary 配置
        random_z_dim: int = 50,
        # 动态场景池配置
        dynamic_scenario_pool: bool = False,
        max_scenario_pool_size: int = 10000,
        # 可配置的空间和 tilting 参数（从 config.yaml 读取）
        obs_dim: int = DEFAULT_OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        tilt_range: List[float] = None,
        tilt_mutation_std: float = DEFAULT_TILT_MUTATION_STD,
        **kwargs
    ):
        """
        Args:
            scenario_index_path: 场景索引 JSON 文件路径
            opponent_checkpoint: ctrl-sim 模型 checkpoint 路径
            scenario_data_dir: Nocturne 场景数据目录
            preprocess_dir: ctrl-sim 预处理数据目录
            opponent_k: 对手车辆数量（选择距离 ego 最近的 K 个）
            max_episode_steps: 最大步数（默认 90，与 ctrl-sim 一致）
            device: 推理设备
            cfg: Hydra 配置对象（可选，若为 None 则自动创建）
            seed: 随机种子
            fixed_environment: 是否固定环境（用于评估）
            random_z_dim: 随机向量维度（用于条件生成）
            dynamic_scenario_pool: 是否启用动态场景池
            max_scenario_pool_size: 动态场景池最大大小
        """
        super().__init__()
        
        self.seed_value = seed
        self.fixed_environment = fixed_environment
        np.random.seed(seed)
        
        # ========== 场景索引（支持动态扩展）==========
        self.scenario_index_path = scenario_index_path
        self.scenario_index = ScenarioIndex(scenario_index_path)
        self.scenario_ids = list(self.scenario_index.scenario_ids)
        self.scenario_id_to_index = dict(self.scenario_index.scenario_id_to_index)
        self.index_to_scenario_id = dict(self.scenario_index.index_to_scenario_id)
        
        # 动态场景池配置
        self.dynamic_scenario_pool = dynamic_scenario_pool
        self.max_scenario_pool_size = max_scenario_pool_size
        self._scenario_pool_dirty = False
        
        # ========== 配置加载 ==========
        if cfg is None:
            cfg = create_minimal_config(
                checkpoint_path=opponent_checkpoint,
                scenario_dir=scenario_data_dir,
                preprocess_dir=preprocess_dir,
            )
        self.cfg = cfg
        self.scenario_data_dir = scenario_data_dir
        self.preprocess_dir = preprocess_dir
        
        # ========== 数据桥接器（Milestone 2）==========
        self.data_bridge = DataBridge(cfg, preprocess_dir)
        
        # ========== 对手策略适配器（Milestone 2）==========
        self.opponent = CtrlSimOpponentAdapter(
            cfg=cfg,
            checkpoint_path=opponent_checkpoint,
            device=device,
        )
        
        # ========== 环境配置 ==========
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.opponent_k = opponent_k
        self.dt = cfg.nocturne.dt
        
        # ========== 状态变量 ==========
        self.current_level: Optional[ScenarioLevel] = None
        self.current_step = 0
        self.adversary_step_count = 0  # Adversary 构建步数
        self.level_seed = seed
        
        # Nocturne 仿真对象
        self.sim = None
        self.scenario = None
        self.vehicles: List = []
        self.ego_vehicle = None
        self.opponent_vehicles: List = []
        self.opponent_vehicle_ids: List[int] = []
        
        # Ground truth 和预处理数据
        self._gt_data_dict: Dict = {}
        self._preproc_data: Optional[Dict] = None
        
        # Ego 车辆的目标和奖励相关状态（用于 _compute_reward）
        self._ego_goal_dict: Optional[Dict] = None
        self._ego_goal_dist_normalizer: float = 1.0
        self._ego_vehicle_data_dict: Dict = {}  # 跟踪 ego 的历史数据
        
        # 终止条件状态
        self._collision_occurred: bool = False
        self._goal_reached: bool = False
        self._offroad_occurred: bool = False
        
        # Episode 统计（用于训练监控）
        self._episode_collision_occurred: bool = False
        self._episode_goal_reached: bool = False
        self._episode_offroad_occurred: bool = False
        self._episode_steps: int = 0
        self._episode_progress: float = 0.0  # 目标进度 [0, 1]
        
        # Level 参数向量（用于 adversary 构建）
        # [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt]
        self.level_params_vec = list(DEFAULT_LEVEL_PARAMS)
        
        # ========== Student 观测配置（从 args 传入）==========
        # 这些参数在 make_agent 时会用到，这里设置默认值
        self._max_observable_agents = kwargs.get('student_num_neighbors', 16)
        self._top_k_road_points = kwargs.get('student_top_k_road', 64)
        
        # 缓存道路数据（在 _initialize_simulation 后填充）
        self._road_graph_cache: Optional[List[Dict]] = None
        
        # ========== 观测和动作空间（Student）==========
        # 计算 Late Fusion 观测维度: ego(6) + partners(K×6) + road_graph(R×13)
        late_fusion_obs_dim = 6 + self._max_observable_agents * 6 + self._top_k_road_points * 13
        
        # 使用配置的 obs_dim 或计算的维度（取较大者以兼容）
        self._obs_dim = max(obs_dim, late_fusion_obs_dim)
        self._action_dim = action_dim
        self.tilt_range = tilt_range if tilt_range is not None else list(DEFAULT_TILT_RANGE)
        self.tilt_mutation_std = tilt_mutation_std
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._obs_dim,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self._action_dim,), 
            dtype=np.float32
        )
        
        # ========== Adversary 空间定义 ==========
        # Adversary 构建环境需要 4 步：scenario_id + 3 个 tilt 参数
        self.adversary_max_steps = 4
        self.random_z_dim = random_z_dim
        self.passable = True  # 驾驶场景默认可通过
        
        # Adversary 动作空间：连续动作 [-1, 1]
        # Step 0: 映射到 scenario 索引
        # Step 1-3: 映射到 tilt 参数
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Adversary 观测空间
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, 
            high=self.adversary_max_steps, 
            shape=(1,), 
            dtype='uint8'
        )
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, 
            high=1.0, 
            shape=(random_z_dim,), 
            dtype=np.float32
        )
        # image: 当前 level 参数向量
        self.adversary_image_obs_space = gym.spaces.Box(
            low=-25.0, 
            high=max(len(self.scenario_ids), 25.0),
            shape=(len(self.level_params_vec),), 
            dtype=np.float32
        )
        self.adversary_observation_space = gym.spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space
        })
        
        # ========== Encoding 格式 ==========
        # 使用字符串数组，与 BipedalWalker 兼容
        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))
        
        # ========== 指标追踪 ==========
        self.reset_metrics()
        
        # ========== 视频录制 ==========
        self.video_recorder: Optional[NocturneVideoRecorder] = None
        self.recording_video = False
        
        # ========== 初始化随机种子 ==========
        self.seed_value = seed
    
    # ========== 基础环境接口 ==========
    
    def seed(self, seed=None):
        """设置环境的随机种子"""
        if seed is not None:
            self.level_seed = seed
            self.seed_value = seed
        return [self.level_seed]
    
    # ========== Adversary 接口（PAIRED/ACCEL）==========
    
    def reset(self) -> Dict:
        """
        重置环境，准备 adversary 构建流程
        
        这是 PAIRED/Minimax 等 UED 算法的入口点。
        返回 adversary 观测，而非 student 观测。
        
        Returns:
            adversary 观测字典: {'image', 'time_step', 'random_z'}
        """
        self.adversary_step_count = 0
        
        # 重置 level 参数为默认值
        self.level_params_vec = list(DEFAULT_LEVEL_PARAMS)
        
        # 生成新的 level seed
        self.level_seed = rand_int_seed()
        
        # 返回 adversary 观测
        obs = {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z()
        }
        
        return obs
    
    def step_adversary(self, action) -> Tuple[Dict, float, bool, Dict]:
        """
        Adversary 构建环境的一步
        
        动作映射：
        - Step 0: action -> scenario_index (离散化到场景池大小)
        - Step 1: action -> goal_tilt ([-1,1] -> [-25,25])
        - Step 2: action -> veh_veh_tilt ([-1,1] -> [-25,25])
        - Step 3: action -> veh_edge_tilt ([-1,1] -> [-25,25])
        
        Args:
            action: 连续动作 [-1, 1]
        
        Returns:
            (obs, reward, done, info)
            - obs: adversary 观测
            - reward: 始终为 0（adversary 奖励在 rollout 后计算）
            - done: 是否完成构建
            - info: 额外信息
        """
        import torch
        if torch.is_tensor(action):
            action = action.item()
        
        # 根据当前步骤设置参数
        if self.adversary_step_count == 0:
            # Step 0: 选择 scenario
            # 将 [-1, 1] 映射到 [0, num_scenarios-1]
            num_scenarios = len(self.scenario_ids)
            scenario_idx = int((action + 1) / 2 * num_scenarios)
            scenario_idx = np.clip(scenario_idx, 0, num_scenarios - 1)
            self.level_params_vec[0] = scenario_idx
        else:
            # Step 1-3: 设置 tilt 参数
            # 将 [-1, 1] 映射到 tilt_range
            tilt_scale = (self.tilt_range[1] - self.tilt_range[0]) / 2.0
            tilt_value = action * tilt_scale
            tilt_value = np.clip(tilt_value, self.tilt_range[0], self.tilt_range[1])
            self.level_params_vec[self.adversary_step_count] = round(float(tilt_value), 1)
        
        self.adversary_step_count += 1
        
        # 检查是否完成构建
        done = self.adversary_step_count >= self.adversary_max_steps
        
        if done:
            # 构建完成，创建 ScenarioLevel 并初始化环境
            self._build_level_from_params()
        
        # 返回 adversary 观测
        obs = {
            'image': np.array(self.level_params_vec, dtype=np.float32),
            'time_step': np.array([self.adversary_step_count], dtype=np.uint8),
            'random_z': self.generate_random_z()
        }
        
        return obs, 0, done, {}
    
    def _build_level_from_params(self):
        """根据 level_params_vec 构建 ScenarioLevel 并初始化环境"""
        # 检查场景池映射是否需要重建
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        
        scenario_idx = int(self.level_params_vec[0])
        
        # 检查场景 ID 是否存在，如果不存在则记录警告
        if scenario_idx not in self.index_to_scenario_id:
            import warnings
            warnings.warn(
                f"Scenario index {scenario_idx} not found in mapping. "
                f"Falling back to first scenario: {self.scenario_ids[0]}"
            )
        scenario_id = self.index_to_scenario_id.get(scenario_idx, self.scenario_ids[0])
        
        self.current_level = ScenarioLevel(
            scenario_id=scenario_id,
            seed=self.level_seed,
            goal_tilt=self.level_params_vec[1],
            veh_veh_tilt=self.level_params_vec[2],
            veh_edge_tilt=self.level_params_vec[3],
        )
        
        # 初始化仿真环境（但不返回观测，等 reset_agent 调用）
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """初始化 Nocturne 仿真（内部方法）"""
        if self.current_level is None:
            return
            
        level = self.current_level
        self.current_step = 0
        self.reset_metrics()
        
        # 重置终止条件状态
        self._collision_occurred = False
        self._goal_reached = False
        self._offroad_occurred = False
        
        # 重置 episode 统计
        self._episode_collision_occurred = False
        self._episode_goal_reached = False
        self._episode_offroad_occurred = False
        self._episode_steps = 0
        self._episode_progress = 0.0
        
        # 设置随机种子
        np.random.seed(level.seed)
        
        # ⚠️ 重要：必须先获取 GT 数据，再加载主场景
        # 原因：get_ground_truth() 内部会创建临时 Simulation 并执行步进，
        # 这会破坏 Nocturne 的全局状态，导致之后创建的 Simulation 中
        # 的车辆对象变得无效（设置属性时会发生段错误）。
        # 解决方案：先获取 GT 数据（让临时 Simulation 完成并销毁），
        # 然后再加载主场景。
        
        # 获取 ground truth 数据（需要添加 .json 后缀）
        self._gt_data_dict = self.data_bridge.get_ground_truth(
            self.scenario_data_dir, 
            f"{level.scenario_id}.json"
        )
        
        # 加载 Nocturne 场景（必须在获取 GT 数据之后）
        self._load_scenario(level.scenario_id)
        
        # 选择 ego 车辆（需要 GT 数据来选择 interesting pair）
        self.ego_vehicle = self._select_ego_vehicle()
        
        # 加载预处理数据（带检查）
        self._preproc_data, file_exists = self.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        if not file_exists:
            raise FileNotFoundError(
                f"Preprocessed data not found for scenario '{level.scenario_id}'. "
                f"Check preprocess_dir: {self.data_bridge.preprocess_dir}"
            )
        
        # 选择对手控制的车辆（从 moving vehicles 中选择最近的 k 辆）
        self._select_opponent_vehicles(k=self.opponent_k)
        
        # 初始化 ego 车辆的目标和奖励相关状态
        self._initialize_ego_goal_state()
        
        # 设置对手 tilting
        self.opponent.set_tilting(
            level.goal_tilt, 
            level.veh_veh_tilt, 
            level.veh_edge_tilt
        )
        self.opponent.reset(
            self.scenario,
            self.vehicles,
            self._gt_data_dict,
            self._preproc_data,
            self.opponent_vehicle_ids,
        )
        
        # 缓存道路数据（用于 Student 观测）
        self._road_graph_cache = self.data_bridge.get_road_data(self.scenario)
    
    def generate_random_z(self) -> np.ndarray:
        """生成随机条件向量（用于 adversary 观测）"""
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
    
    @property
    def processed_action_dim(self) -> int:
        """处理后的动作维度（兼容 AdversarialRunner）"""
        return 1
    
    # ========== PLR/DR 接口 ==========
    
    def reset_random(self) -> np.ndarray:
        """
        随机生成新 level 并 reset
        
        这是 DCD Domain Randomization 的入口点。
        直接跳过 adversary 构建，随机采样所有参数。
        
        Returns:
            student 初始观测
        """
        level = self._sample_random_level()
        return self.reset_to_level(level)
    
    def reset_to_level(self, level: Union[ScenarioLevel, str, np.ndarray]) -> np.ndarray:
        """
        加载指定 level
        
        支持三种输入格式（兼容 DCD LevelStore）：
        1. ScenarioLevel 对象
        2. 字符串（from to_level_string()）
        3. numpy 数组（from encoding）
        
        Args:
            level: Level 对象、字符串或编码数组
        
        Returns:
            student 初始观测
        """
        # 统一转换为 ScenarioLevel 对象
        if isinstance(level, str):
            level = ScenarioLevel.from_level_string(level)
        elif isinstance(level, np.ndarray):
            # 处理字符串数组格式
            if level.dtype.kind == 'U':
                level = self._decode_string_encoding(level)
            else:
                level = ScenarioLevel.from_encoding(level, self.index_to_scenario_id)
        
        self.current_level = level
        
        # 更新 level_params_vec 以保持一致
        scenario_idx = self.scenario_id_to_index.get(level.scenario_id, 0)
        self.level_params_vec = [
            scenario_idx,
            level.goal_tilt,
            level.veh_veh_tilt,
            level.veh_edge_tilt,
        ]
        self.level_seed = level.seed
        
        # 初始化仿真
        self._initialize_simulation()
        
        # 返回 student 观测
        return self._get_student_observation()
    
    def _decode_string_encoding(self, encoding: np.ndarray) -> ScenarioLevel:
        """从字符串数组编码恢复 ScenarioLevel"""
        # 检查场景池映射是否需要重建
        if self._scenario_pool_dirty:
            self.rebuild_index_mappings()
        
        # 格式: [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt, seed]
        scenario_idx = int(float(encoding[0]))
        
        # 检查场景 ID 是否存在，如果不存在则记录警告
        if scenario_idx not in self.index_to_scenario_id:
            import warnings
            warnings.warn(
                f"Scenario index {scenario_idx} not found in mapping. "
                f"Falling back to first scenario: {self.scenario_ids[0]}"
            )
        scenario_id = self.index_to_scenario_id.get(scenario_idx, self.scenario_ids[0])
        
        return ScenarioLevel(
            scenario_id=scenario_id,
            seed=int(float(encoding[4])),
            goal_tilt=float(encoding[1]),
            veh_veh_tilt=float(encoding[2]),
            veh_edge_tilt=float(encoding[3]),
        )
    
    def reset_agent(self) -> np.ndarray:
        """
        在当前 level 重新 reset（不改变 level 配置）
        
        用于：
        1. PAIRED 中 adversary 构建完成后启动 student
        2. PLR 中同一 level 的多次评估
        
        Returns:
            student 初始观测
        """
        if self.current_level is None:
            raise ValueError("Must call reset_to_level or complete step_adversary first")
        
        # 重新初始化仿真
        self._initialize_simulation()
        
        return self._get_student_observation()
    
    def mutate_level(self, num_edits: int = 1) -> np.ndarray:
        """
        变异当前 level 并 reset
        
        这是 DCD ACCEL 算法的核心接口，用于 level 编辑。
        
        变异策略（参考 BipedalWalker）：
        - 随机选择参数进行扰动
        - 扰动方向：-1, 0, +1
        - 扰动幅度：高斯分布
        
        Args:
            num_edits: 变异次数
        
        Returns:
            变异后 level 的 student 初始观测
        """
        if self.current_level is None:
            raise ValueError("Must call reset_to_level first")
        
        if num_edits > 0:
            mutated = self._mutate_level_internal(self.current_level, num_edits)
            self.level_seed = rand_int_seed()  # 新种子
            return self.reset_to_level(mutated)
        
        return self.reset_agent()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步仿真（Student 动作）
        
        数据流：
        1. 对手策略推理（使用 CtrlSimOpponentAdapter）
        2. 应用所有动作到 Nocturne 仿真
        3. 仿真步进
        4. 计算奖励和终止条件
        5. 如果启用录制，捕获当前帧
        """
        self.current_step += 1
        
        # 1. 对手推理动作
        opponent_actions = self.opponent.step(self.current_step - 1, self.vehicles)
        
        # 2. 应用学生动作到 ego 车辆
        self._apply_student_action(action)
        
        # 3. 应用对手动作
        for veh_id, (accel, steer) in opponent_actions.items():
            veh = self._get_vehicle_by_id(veh_id)
            if veh is not None:
                self.opponent.apply_action(veh, (accel, steer))
        
        # 4. 对未控制车辆应用 GT 动作（参考 policy_evaluator.py line 536-538）
        ego_id = self.ego_vehicle.getID() if self.ego_vehicle else None
        controlled_ids = set(opponent_actions.keys())
        if ego_id is not None:
            controlled_ids.add(ego_id)
        
        for veh in self.vehicles:
            veh_id = veh.getID()
            if veh_id not in controlled_ids:
                gt_action = self._get_gt_action(veh_id, self.current_step - 1)
                if gt_action is not None:
                    self.opponent.apply_action(veh, gt_action)
        
        # 5. 记录所有车辆的动作（用于下一步的 update_state）
        self.opponent.record_all_actions(
            self.current_step - 1, 
            self.vehicles, 
            opponent_actions
        )
        
        # 6. 仿真步进
        self.sim.step(self.dt)
        
        # 7. 如果启用录制，捕获当前帧
        if self.recording_video and self.video_recorder is not None:
            self.video_recorder.capture_frame(
                self.scenario,
                self.vehicles,
                highlight_vehicle_ids=[self.ego_vehicle.getID()] if self.ego_vehicle else None
            )
        
        # 8. 计算奖励和终止条件
        obs = self._get_student_observation()
        reward = self._compute_reward()
        
        # 更新 episode 统计
        self._episode_steps += 1
        if self._collision_occurred:
            self._episode_collision_occurred = True
        if self._goal_reached:
            self._episode_goal_reached = True
        if self._offroad_occurred:
            self._episode_offroad_occurred = True
        
        # 计算目标进度（当前距离 vs 初始距离）
        if self.ego_vehicle and self._ego_goal_dict and self._ego_goal_dist_normalizer > 0:
            ego_pos = self.ego_vehicle.getPosition()
            goal_pos = self._ego_goal_dict['pos']
            current_dist = np.linalg.norm(goal_pos - np.array([ego_pos.x, ego_pos.y]))
            self._episode_progress = max(0.0, 1.0 - current_dist / self._ego_goal_dist_normalizer)
        
        done = self._check_done()
        info = self._get_info()
        
        return obs, reward, done, info
    
    # ========== Level 属性和编码 ==========
    
    @property
    def level(self) -> str:
        """
        返回当前 level 的字符串表示
        
        用于 LevelStore 存储（字符串模式）
        """
        if self.current_level is None:
            return ""
        return self.current_level.to_level_string()
    
    def get_level(self) -> str:
        """level 属性的方法形式"""
        return self.level
    
    @property
    def encoding(self) -> np.ndarray:
        """
        返回当前 level 的编码
        
        格式与 BipedalWalker 兼容：字符串数组
        [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt, seed]
        
        用于 PLR buffer 存储（字节模式）
        """
        if self.current_level is None:
            enc = DEFAULT_LEVEL_PARAMS + [self.level_seed]
        else:
            scenario_idx = self.scenario_id_to_index.get(
                self.current_level.scenario_id, 0
            )
            enc = [
                scenario_idx,
                self.current_level.goal_tilt,
                self.current_level.veh_veh_tilt,
                self.current_level.veh_edge_tilt,
                self.current_level.seed,
            ]
        
        # 转为字符串数组（与 BipedalWalker 兼容）
        enc_str = [str(x) for x in enc]
        return np.array(enc_str, dtype=self.encoding_u_chars)
    
    def get_encodings(self) -> List[np.ndarray]:
        """返回编码列表（兼容 vectorized env 接口）"""
        return [self.encoding]
    
    # ========== 动态场景池支持 ==========
    
    def add_scenario(self, scenario_id: str) -> bool:
        """添加新场景到场景池"""
        if not self.dynamic_scenario_pool:
            return False
        
        if scenario_id in self.scenario_id_to_index:
            return False
        
        if len(self.scenario_ids) >= self.max_scenario_pool_size:
            old_id = self.scenario_ids.pop(0)
            old_idx = self.scenario_id_to_index.pop(old_id)
            del self.index_to_scenario_id[old_idx]
        
        new_idx = len(self.scenario_ids)
        self.scenario_ids.append(scenario_id)
        self.scenario_id_to_index[scenario_id] = new_idx
        self.index_to_scenario_id[new_idx] = scenario_id
        
        self._scenario_pool_dirty = True
        return True
    
    def get_scenario_pool_size(self) -> int:
        """返回当前场景池大小"""
        return len(self.scenario_ids)
    
    def rebuild_index_mappings(self):
        """重建索引映射"""
        self.scenario_id_to_index = {
            sid: i for i, sid in enumerate(self.scenario_ids)
        }
        self.index_to_scenario_id = {
            i: sid for i, sid in enumerate(self.scenario_ids)
        }
        self._scenario_pool_dirty = False
    
    # ========== 指标和信息 ==========
    
    def reset_metrics(self):
        """重置指标追踪"""
        self.episode_reward = 0.0
        self.collision_count = 0
        self.goal_reached = False
    
    def get_complexity_info(self) -> Dict[str, Any]:
        """
        返回当前 level 的复杂度信息和 episode 统计（用于日志和分析）
        
        Returns:
            包含 level 参数和 episode 统计的字典
        """
        if self.current_level is None:
            return {}
        
        info = {
            # Level 参数
            'scenario_id': self.current_level.scenario_id,
            'seed': self.current_level.seed,
            'opponent_k': self.opponent_k,
            'goal_tilt': self.current_level.goal_tilt,
            'veh_veh_tilt': self.current_level.veh_veh_tilt,
            'veh_edge_tilt': self.current_level.veh_edge_tilt,
            'scenario_pool_size': len(self.scenario_ids),
            
            # Episode 统计（用于训练监控）
            'collision_rate': 1.0 if self._episode_collision_occurred else 0.0,
            'goal_reached_rate': 1.0 if self._episode_goal_reached else 0.0,
            'offroad_rate': 1.0 if self._episode_offroad_occurred else 0.0,
            'avg_progress': self._episode_progress,
            'episode_steps': self._episode_steps,
            'episode_reward': self.episode_reward,
        }
        
        return info
    
    # ========== 内部辅助方法 ==========
    
    def _sample_random_level(self) -> ScenarioLevel:
        """随机生成 level"""
        return ScenarioLevel(
            scenario_id=np.random.choice(self.scenario_ids),
            seed=rand_int_seed(),
            goal_tilt=round(float(np.random.uniform(*self.tilt_range)), 1),
            veh_veh_tilt=round(float(np.random.uniform(*self.tilt_range)), 1),
            veh_edge_tilt=round(float(np.random.uniform(*self.tilt_range)), 1),
        )
    
    def _mutate_level_internal(
        self,
        level: ScenarioLevel,
        num_edits: int
    ) -> ScenarioLevel:
        """
        执行 level 变异
        
        变异策略（参考 BipedalWalker）：
        1. 随机选择参数
        2. 随机选择方向：-1, 0, +1
        3. 应用高斯扰动
        
        注意：只变异 tilt 参数，不改变 scenario_id
        """
        from dataclasses import replace
        
        mutations = {}
        params = ['goal_tilt', 'veh_veh_tilt', 'veh_edge_tilt']
        
        for _ in range(num_edits):
            # 只变异 tilt 参数，不改变 scenario_id
            param = np.random.choice(params)
            current_val = mutations.get(param, getattr(level, param))
            direction = np.random.randint(-1, 2)  # -1, 0, 1
            mutation = direction * np.random.uniform(0, self.tilt_mutation_std)
            new_val = np.clip(current_val + mutation, *self.tilt_range)
            mutations[param] = round(float(new_val), 1)
        
        return replace(level, **mutations)
    
    def _load_scenario(self, scenario_id: str):
        """
        加载 Nocturne 场景
        
        参考: third_party/ctrl-sim/utils/sim.py 的 get_sim() 函数
        """
        import os
        from nocturne import Simulation
        from omegaconf import OmegaConf
        
        scenario_path = os.path.join(self.scenario_data_dir, f"{scenario_id}.json")
        
        # Nocturne Simulation 只需要 scenario 配置字典（扁平格式）
        # 参考 cfgs/config.py 的 get_scenario_dict() 函数
        if 'scenario' in self.cfg.nocturne:
            scenario_dict = OmegaConf.to_container(
                self.cfg.nocturne.scenario, resolve=True
            )
        else:
            # 回退到基本配置
            scenario_dict = {
                'start_time': 0,
                'allow_non_vehicles': False,
            }
        
        self.sim = Simulation(scenario_path, scenario_dict)
        self.scenario = self.sim.getScenario()
        self.vehicles = list(self.scenario.vehicles())
        
        # 设置车辆控制标志（参考 ctrl-sim evaluator.py line 37-39）
        for veh in self.vehicles:
            veh.expert_control = False
            veh.physics_simulated = True
        
        # 注意: ego 选择移到 _initialize_simulation() 中，
        # 因为需要 GT 数据来选择 interesting pair
    
    def _get_moving_vehicle_ids(self) -> List[int]:
        """
        获取场景中所有 moving vehicles 的 ID
        
        参考: ctrl-sim utils/sim.py get_moving_vehicles()
        """
        return [v.getID() for v in self.scenario.getObjectsThatMoved()]
    
    def _find_interesting_pair(self, moving_veh_ids: List[int]) -> Optional[Tuple[int, int]]:
        """
        找到一对 interesting 车辆（参考 ctrl-sim policy_evaluator.py line 362-412）
        
        筛选条件:
        - 目标位置接近（<10米）
        - 目标时间步接近（<20步）
        - 轨迹足够长（>=60步）
        
        Returns:
            (veh_id_1, veh_id_2) 元组，如果找不到则返回 None
        """
        # 配置阈值（参考 ctrl-sim cfg.eval）
        goal_dist_threshold = 10.0  # 米
        timestep_diff_threshold = 20  # 步
        traj_len_threshold = 60  # 步
        history_steps = getattr(self.cfg.nocturne, 'history_steps', 10)
        
        goals = []
        goal_timesteps = []
        valid_traj_mask = []
        veh_ids = []
        
        for veh_id in moving_veh_ids:
            if veh_id not in self._gt_data_dict:
                continue
                
            gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
            existence_mask = gt_traj[:, 4]
            
            # 计算目标位置和时间步
            idx_goal = self.max_episode_steps - 1
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
            
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
                
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj[idx_goal, :2]
            
            # 检查轨迹长度
            has_valid_traj = existence_mask[history_steps:].sum() >= traj_len_threshold
            
            goals.append(goal_pos)
            goal_timesteps.append(idx_goal - history_steps)
            valid_traj_mask.append(1 if has_valid_traj else 0)
            veh_ids.append(veh_id)
        
        if len(goals) < 2:
            return None
        
        goals = np.array(goals)
        goal_timesteps = np.array(goal_timesteps)
        valid_traj_mask = np.array(valid_traj_mask)
        
        # 计算目标距离矩阵
        dists = np.linalg.norm(goals[:, np.newaxis] - goals[np.newaxis, :], axis=-1)
        
        # 构建 mask
        nearby_mask = dists < goal_dist_threshold
        not_same_mask = dists > 0
        valid_traj_both = np.outer(valid_traj_mask, valid_traj_mask)
        timestep_diff = np.abs(goal_timesteps[:, np.newaxis] - goal_timesteps[np.newaxis, :])
        within_time_mask = timestep_diff < timestep_diff_threshold
        
        goal_mask = nearby_mask & not_same_mask & valid_traj_both.astype(bool) & within_time_mask
        
        indices = np.where(goal_mask)
        valid_pairs = list(zip(indices[0], indices[1]))
        
        if len(valid_pairs) == 0:
            return None
        
        # 确定性选择: 选择第一对（按索引排序，保证一致性）
        pair_idx = valid_pairs[0]
        return (veh_ids[pair_idx[0]], veh_ids[pair_idx[1]])
    
    def _select_ego_vehicle(self):
        """
        选择 ego 车辆（学生控制）
        
        使用 find_interesting_pair 逻辑选择两个 interesting 车辆，
        然后确定性地选择 veh_id 较小的作为 ego。
        
        如果找不到 interesting pair，抛出异常。
        """
        # 1. 获取 moving vehicles
        moving_veh_ids = self._get_moving_vehicle_ids()
        
        if len(moving_veh_ids) == 0:
            raise ValueError(
                f"No moving vehicles found in scenario {self.current_level.scenario_id}. "
                "Scenario will be skipped."
            )
        
        # 2. 找到 interesting pair
        interesting_pair = self._find_interesting_pair(moving_veh_ids)
        
        if interesting_pair is None:
            # 如果没有找到 interesting pair，降级选择：选择第一个 moving vehicle
            print(
                f"Warning: No interesting vehicle pair found in scenario {self.current_level.scenario_id}. "
                f"Using first moving vehicle as ego."
            )
            if len(moving_veh_ids) > 0:
                ego_veh_id = moving_veh_ids[0]
            else:
                raise ValueError(
                    f"No moving vehicles found in scenario {self.current_level.scenario_id}."
                )
        else:
            # 3. 确定性选择: veh_id 较小的作为 ego
            ego_veh_id = min(interesting_pair)
        
        return self._get_vehicle_by_id(ego_veh_id)
    
    def _select_opponent_vehicles(self, k: int = 7):
        """
        选择对手控制的车辆（距离 ego 最近的 k 个 moving vehicles）
        
        参考 ctrl-sim 的距离计算方式
        """
        if self.ego_vehicle is None:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # 1. 获取 moving vehicles（排除 ego）
        moving_veh_ids = self._get_moving_vehicle_ids()
        ego_id = self.ego_vehicle.getID()
        candidate_ids = [vid for vid in moving_veh_ids if vid != ego_id]
        
        if len(candidate_ids) == 0:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        # 2. 计算到 ego 的距离
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        
        distances = []
        for veh_id in candidate_ids:
            veh = self._get_vehicle_by_id(veh_id)
            if veh is None:
                continue
            pos = veh.getPosition()
            dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
            distances.append((dist, veh_id, veh))
        
        # 3. 按距离排序，选择最近的 k 辆
        distances.sort(key=lambda x: x[0])
        selected = distances[:k]
        
        self.opponent_vehicles = [item[2] for item in selected]
        self.opponent_vehicle_ids = [item[1] for item in selected]
    
    def _get_vehicle_by_id(self, veh_id: int):
        """根据 ID 获取车辆对象"""
        for veh in self.vehicles:
            if veh.getID() == veh_id:
                return veh
        return None
    
    def _initialize_ego_goal_state(self):
        """
        初始化 ego 车辆的目标和奖励相关状态
        
        参考: ctrl-sim evaluator.py initialize_goal_dict() 和 compute_goal_dist_normalizer()
        """
        if self.ego_vehicle is None:
            return
        
        ego_id = self.ego_vehicle.getID()
        
        # 获取 GT 轨迹数据
        if ego_id not in self._gt_data_dict:
            return
        
        gt_traj_data = np.array(self._gt_data_dict[ego_id]['traj'])
        
        # 计算目标位置（参考 evaluator.py initialize_goal_dict）
        goal_pos = np.array([
            self.ego_vehicle.target_position.x,
            self.ego_vehicle.target_position.y
        ])
        goal_heading = self.ego_vehicle.target_heading
        goal_speed = self.ego_vehicle.target_speed
        
        # 检查车辆是否在轨迹结束前消失，如果是则使用最后有效位置作为目标
        existence_mask = gt_traj_data[:, 4]
        idx_disappear = np.where(existence_mask == 0)[0]
        if len(idx_disappear) > 0:
            idx_goal = idx_disappear[0] - 1
            if idx_goal >= 0 and np.linalg.norm(gt_traj_data[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj_data[idx_goal, :2]
                goal_heading = gt_traj_data[idx_goal, 2]
                goal_speed = gt_traj_data[idx_goal, 3]
        
        self._ego_goal_dict = {
            'pos': goal_pos,
            'heading': goal_heading,
            'speed': goal_speed
        }
        
        # 计算目标距离归一化因子
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        dist = np.linalg.norm(ego_pos - goal_pos)
        self._ego_goal_dist_normalizer = dist if dist > 0 else 1.0
        
        # 初始化 ego 的 vehicle_data_dict（用于奖励计算）
        self._ego_vehicle_data_dict = {
            ego_id: {
                'reward': [],
                'position': [],
                'heading': [],
                'speed': [],
            }
        }
    
    def _get_gt_action(self, veh_id: int, t: int) -> Optional[Tuple[float, float]]:
        """
        从 GT 轨迹数据中获取车辆在时间步 t 的动作
        
        参考: ctrl-sim policy_evaluator.py apply_gt_action()
        
        Args:
            veh_id: 车辆 ID
            t: 时间步
        
        Returns:
            (acceleration, steering) 元组，如果数据不存在则返回 None
        """
        if veh_id not in self._gt_data_dict:
            return None
        
        gt_traj = np.array(self._gt_data_dict[veh_id]['traj'])
        
        # 检查时间步是否有效
        if t < 0 or t >= len(gt_traj) - 1:
            return (0.0, 0.0)
        
        # 检查车辆在当前和下一时间步是否存在
        veh_exists = gt_traj[t, 4] and gt_traj[t + 1, 4]
        if not veh_exists:
            return (0.0, 0.0)
        
        # 计算加速度（速度差分）
        accel = (gt_traj[t + 1, 3] - gt_traj[t, 3]) / self.dt
        
        # 计算转向率（航向差分）
        steer = (gt_traj[t + 1, 2] - gt_traj[t, 2]) / self.dt
        
        return (float(accel), float(steer))
    
    def _apply_student_action(self, action: np.ndarray):
        """
        将学生动作应用到 ego 车辆
        
        Args:
            action: [acceleration, steering] 归一化到 [-1, 1]
        """
        if self.ego_vehicle is None:
            return
        
        # 将归一化动作转换为实际值
        # TODO: 根据实际动作范围调整
        accel = action[0] * 5.0  # 假设最大加速度 5 m/s²
        steer = action[1] * 0.5  # 假设最大转向角 0.5 rad
        
        if accel > 0:
            self.ego_vehicle.acceleration = accel
        else:
            self.ego_vehicle.brake(abs(accel))
        self.ego_vehicle.steering = steer
    
    def _build_road_graph_obs(
        self, 
        ego_pos, 
        ego_heading: float
    ) -> List[np.ndarray]:
        """
        构建 Road Graph 观测（符合 gpudrive 格式）
        
        Road Graph 特征 (13 维):
        - pos_x, pos_y (2): 道路点相对于 ego 的位置
        - length (1): 道路段长度
        - scale_x, scale_y (2): 道路点尺度
        - orientation (1): 道路方向
        - type_onehot (7): 道路类型 one-hot编码
        
        Args:
            ego_pos: ego 车辆位置
            ego_heading: ego 车辆朝向
        
        Returns:
            road_graph_states: List of road point features (R 个 13 维向量)
        """
        if self._road_graph_cache is None or len(self._road_graph_cache) == 0:
            # 没有道路数据，返回空的 road graph
            return [np.zeros(13, dtype=np.float32) for _ in range(self._top_k_road_points)]
        
        # 提取道路点特征
        road_points = []
        
        for road_item in self._road_graph_cache:
            road_type = road_item['type']
            geometry = road_item['geometry']
            
            # 处理不同类型的几何数据
            if isinstance(geometry, list) and len(geometry) > 0:
                # 道路线（多个点）
                for i, pt in enumerate(geometry):
                    # 相对位置
                    rel_x = pt['x'] - ego_pos.x
                    rel_y = pt['y'] - ego_pos.y
                    
                    # 计算道路段长度
                    if i < len(geometry) - 1:
                        next_pt = geometry[i + 1]
                        seg_length = np.sqrt(
                            (next_pt['x'] - pt['x'])**2 + 
                            (next_pt['y'] - pt['y'])**2
                        )
                        # 方向：指向下一个点
                        orientation = np.arctan2(
                            next_pt['y'] - pt['y'],
                            next_pt['x'] - pt['x']
                        )
                    else:
                        seg_length = 1.0  # 默认值
                        orientation = 0.0
                    
                    # 道路点尺度（默认值）
                    scale_x = 1.0
                    scale_y = 1.0
                    
                    # 道路类型 one-hot (7 维)
                    # ctrl-sim: {none:0, lane:1, road_line:2, road_edge:3, stop_sign:4, crosswalk:5, speed_bump:6, other:7}
                    # gpudrive: {None:0, RoadLine:1, RoadEdge:2, RoadLane:3, CrossWalk:4, SpeedBump:5, StopSign:6}
                    # 映射到 gpudrive 顺序
                    type_mapping = {
                        'none': 0,
                        'road_line': 1,
                        'road_edge': 2,
                        'lane': 3,
                        'crosswalk': 4,
                        'speed_bump': 5,
                        'stop_sign': 6,
                        'other': 0,  # 映射到 None
                    }
                    type_idx = type_mapping.get(road_type, 0)
                    type_onehot = np.zeros(7, dtype=np.float32)
                    type_onehot[type_idx] = 1.0
                    
                    # 拼接特征 (13 维)
                    road_feat = np.array([
                        rel_x,
                        rel_y,
                        seg_length,
                        scale_x,
                        scale_y,
                        orientation,
                        *type_onehot
                    ], dtype=np.float32)
                    
                    road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))
            
            elif isinstance(geometry, dict):
                # 静态目标（如 stop_sign）
                rel_x = geometry['x'] - ego_pos.x
                rel_y = geometry['y'] - ego_pos.y
                
                type_mapping = {
                    'stop_sign': 6,
                    'crosswalk': 4,
                    'speed_bump': 5,
                }
                type_idx = type_mapping.get(road_type, 0)
                type_onehot = np.zeros(7, dtype=np.float32)
                type_onehot[type_idx] = 1.0
                
                road_feat = np.array([
                    rel_x,
                    rel_y,
                    0.0,  # length
                    1.0,  # scale_x
                    1.0,  # scale_y
                    0.0,  # orientation
                    *type_onehot
                ], dtype=np.float32)
                
                road_points.append((np.sqrt(rel_x**2 + rel_y**2), road_feat))
        
        # 按距离排序，选择最近的 top_k 个点
        road_points.sort(key=lambda x: x[0])
        
        road_graph_states = []
        num_valid_points = min(len(road_points), self._top_k_road_points)
        
        for i in range(num_valid_points):
            road_graph_states.append(road_points[i][1])
        
        # 填充不足的道路点
        for _ in range(self._top_k_road_points - num_valid_points):
            road_graph_states.append(np.zeros(13, dtype=np.float32))
        
        return road_graph_states

    def _get_student_observation(self) -> np.ndarray:
        """
        获取学生策略的观测（Late Fusion 格式，与 gpudrive 一致）
        
        观测向量结构：
        - Ego 状态: [speed, length, width, rel_goal_x, rel_goal_y, collision_state] (6 维)
        - Partners: K 辆车 × [speed, rel_pos_x, rel_pos_y, rel_orientation, length, width] (K×6 维)
        - Road graph: R 个点 × [pos_x, pos_y, length, scale_x, scale_y, orientation, type_onehot(7)] (R×13 维)
        
        Returns:
            观测向量，形状为 (obs_dim,)
        """
        if self.ego_vehicle is None or self._ego_goal_dict is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        
        # ========== Ego 状态 (6 维) ==========
        ego_pos = self.ego_vehicle.getPosition()
        ego_heading = self.ego_vehicle.getHeading()
        ego_speed = self.ego_vehicle.getSpeed()
        
        # 相对目标位置（ego 坐标系）
        goal_pos = self._ego_goal_dict['pos']
        rel_goal_x = goal_pos[0] - ego_pos.x
        rel_goal_y = goal_pos[1] - ego_pos.y
        
        # 碰撞状态
        collision_state = 1.0 if self._collision_occurred else 0.0
        
        ego_state = np.array([
            ego_speed,
            self.ego_vehicle.getLength(),
            self.ego_vehicle.getWidth(),
            rel_goal_x,
            rel_goal_y,
            collision_state,
        ], dtype=np.float32)
        
        # ========== Partner 状态 (K×6 维) ==========
        # 使用 args 配置的 num_neighbors，从 __init__ 获取
        max_neighbors = getattr(self, '_max_observable_agents', 16)
        partner_states = []
        
        # 选择最近的 K 辆邻车
        num_neighbors = min(len(self.opponent_vehicles), max_neighbors)
        
        for i in range(num_neighbors):
            veh = self.opponent_vehicles[i]
            veh_pos = veh.getPosition()
            
            # 相对于 ego 的位置
            rel_pos_x = veh_pos.x - ego_pos.x
            rel_pos_y = veh_pos.y - ego_pos.y
            
            # 相对朝向
            rel_orientation = veh.getHeading() - ego_heading
            # 归一化到 [-pi, pi]
            while rel_orientation > np.pi:
                rel_orientation -= 2 * np.pi
            while rel_orientation < -np.pi:
                rel_orientation += 2 * np.pi
            
            partner_state = np.array([
                veh.getSpeed(),
                rel_pos_x,
                rel_pos_y,
                rel_orientation,
                veh.getLength(),
                veh.getWidth(),
            ], dtype=np.float32)
            partner_states.append(partner_state)
        
        # 填充不足的邻车为零向量
        for _ in range(max_neighbors - num_neighbors):
            partner_states.append(np.zeros(6, dtype=np.float32))
        
        # ========== Road Graph (R×13 维) ==========
        road_graph_states = self._build_road_graph_obs(ego_pos, ego_heading)
        
        # ========== 拼接所有观测 ==========
        obs_parts = [ego_state]
        obs_parts.extend(partner_states)
        obs_parts.extend(road_graph_states)
        
        obs_concat = np.concatenate(obs_parts)
        
        # 填充或截断到 obs_dim
        if len(obs_concat) < self._obs_dim:
            obs_final = np.zeros(self._obs_dim, dtype=np.float32)
            obs_final[:len(obs_concat)] = obs_concat
        else:
            obs_final = obs_concat[:self._obs_dim]
        
        return obs_final
    
    def _compute_reward(self) -> float:
        """
        计算学生奖励
        
        奖励组成（参考 ctrl-sim compute_reward）：
        - 到达目标奖励（shaped_goal_distance）
        - 碰撞惩罚
        - 出界惩罚
        
        Returns:
            标量奖励值
        """
        import nocturne
        
        if self.ego_vehicle is None or self._ego_goal_dict is None:
            return 0.0
        
        reward = 0.0
        ego_id = self.ego_vehicle.getID()
        
        # ========== 获取当前状态 ==========
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        ego_speed = self.ego_vehicle.getSpeed()
        ego_heading = self.ego_vehicle.getHeading()
        
        goal_pos = self._ego_goal_dict['pos']
        goal_speed = self._ego_goal_dict['speed']
        goal_heading = self._ego_goal_dict['heading']
        
        # ========== 目标达成检测 ==========
        dist_to_goal = np.linalg.norm(goal_pos - ego_pos)
        position_tolerance = 1.0  # 米
        speed_tolerance = 1.0  # m/s
        heading_tolerance = 0.3  # rad
        
        position_achieved = dist_to_goal < position_tolerance
        speed_achieved = abs(ego_speed - goal_speed) < speed_tolerance
        heading_achieved = abs(self._angle_diff(ego_heading, goal_heading)) < heading_tolerance
        
        # 如果之前已经达到目标，保持达成状态
        if self._goal_reached:
            position_achieved = True
        elif position_achieved and speed_achieved and heading_achieved:
            self._goal_reached = True
        
        # ========== Shaped Goal Distance Reward ==========
        # 越接近目标，奖励越高
        goal_dist_scaling = 0.2
        reward_scaling = 1.0
        
        if self._ego_goal_dist_normalizer > 0:
            # 归一化距离奖励：[0, 1]，越近越高
            if self._goal_reached:
                pos_goal_rew = goal_dist_scaling / reward_scaling
            else:
                pos_goal_rew = goal_dist_scaling * (1 - dist_to_goal / self._ego_goal_dist_normalizer) / reward_scaling
                pos_goal_rew = max(0.0, pos_goal_rew)  # 确保非负
        else:
            pos_goal_rew = 0.0
        
        reward += pos_goal_rew
        
        # ========== 碰撞惩罚 ==========
        try:
            veh_veh_collision = self.ego_vehicle.collision_type_veh == nocturne.CollisionType.VEHICLE_VEHICLE
            veh_edge_collision = self.ego_vehicle.collision_type_edge == nocturne.CollisionType.VEHICLE_ROAD
        except AttributeError:
            # 如果 nocturne 版本不支持，使用旧版 API
            try:
                veh_veh_collision = self.ego_vehicle.collision_type == nocturne.CollisionType.VEHICLE_VEHICLE
                veh_edge_collision = self.ego_vehicle.collision_type == nocturne.CollisionType.VEHICLE_ROAD
            except:
                veh_veh_collision = False
                veh_edge_collision = False
        
        collision_penalty = -1.0
        if veh_veh_collision:
            reward += collision_penalty
            self._collision_occurred = True
        
        if veh_edge_collision:
            reward += collision_penalty * 0.5  # 出界惩罚稍轻
            self._offroad_occurred = True
        
        # ========== 更新 vehicle_data_dict（用于持续跟踪）==========
        if ego_id in self._ego_vehicle_data_dict:
            self._ego_vehicle_data_dict[ego_id]['reward'].append([
                float(position_achieved),
                float(heading_achieved),
                float(speed_achieved),
                pos_goal_rew,
                0.0,  # speed_goal_rew
                0.0,  # heading_goal_rew
                float(veh_veh_collision),
                float(veh_edge_collision),
            ])
        
        self.episode_reward += reward
        return reward
    
    def _angle_diff(self, a: float, b: float) -> float:
        """计算两个角度之间的差值（处理 wraparound）"""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _check_done(self) -> bool:
        """
        检查 episode 是否结束
        
        终止条件：
        1. 达到最大步数（超时）
        2. 发生碰撞
        3. 到达目标
        4. 出界（offroad）
        
        Returns:
            是否终止
        """
        # 达到最大步数
        if self.current_step >= self.max_episode_steps:
            return True
        
        # 发生碰撞（vehicle-vehicle）
        if self._collision_occurred:
            return True
        
        # 到达目标
        if self._goal_reached:
            return True
        
        # 出界（offroad）- 可选，某些场景可能不需要立即终止
        # if self._offroad_occurred:
        #     return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """返回额外信息"""
        # 计算 progress（到目标的距离进度）
        progress = 0.0
        if self.ego_vehicle is not None and self._ego_goal_dict is not None:
            ego_pos = self.ego_vehicle.getPosition()
            dist_to_goal = np.linalg.norm(
                self._ego_goal_dict['pos'] - np.array([ego_pos.x, ego_pos.y])
            )
            if self._ego_goal_dist_normalizer > 0:
                progress = 1.0 - dist_to_goal / self._ego_goal_dist_normalizer
                progress = max(0.0, min(1.0, progress))
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            # 诊断信息（参考 ctrl-sim metrics）
            'collision': self._collision_occurred,
            'goal_reached': self._goal_reached,
            'offroad': self._offroad_occurred,
            'progress': progress,
        }
        
        # 在 episode 结束时添加统计信息
        if self._check_done():
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step,
            }
            info.update(self.get_complexity_info())
        
        return info
    
    def render(self, mode='human'):
        """渲染环境（静态截图）"""
        if mode not in ['human', 'rgb_array', 'level']:
            raise NotImplementedError

        if self.scenario is None or not self.vehicles:
            return None

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.patches as mpatches
        import matplotlib.transforms as transforms

        vehicle_data = []
        positions = []
        for veh in self.vehicles:
            pos = veh.getPosition()
            if pos.x == -10000 and pos.y == -10000:
                continue
            vehicle_data.append({
                'id': veh.getID(),
                'x': pos.x,
                'y': pos.y,
                'heading': veh.getHeading(),
                'length': veh.getLength(),
                'width': veh.getWidth(),
            })
            positions.append([pos.x, pos.y])

        if not vehicle_data:
            return None

        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        def _draw_road(geometry, color, linewidth):
            if isinstance(geometry, dict):
                ax.scatter(geometry['x'], geometry['y'], color='red', s=20, zorder=1)
            elif isinstance(geometry, list) and len(geometry) > 0:
                xs = [p['x'] for p in geometry]
                ys = [p['y'] for p in geometry]
                ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=1)

        roads_data = self._road_graph_cache
        if roads_data is None and self.scenario is not None:
            roads_data = self.data_bridge.get_road_data(self.scenario)

        if roads_data:
            for road in roads_data:
                if road.get('type') == 'road_edge':
                    _draw_road(road.get('geometry', []), color='grey', linewidth=0.5)
            for road in roads_data:
                if road.get('type') != 'road_edge':
                    _draw_road(road.get('geometry', []), color='lightgray', linewidth=0.3)

        positions = np.array(positions)
        x_min = np.min(positions[:, 0]) - 25
        x_max = np.max(positions[:, 0]) + 25
        y_min = np.min(positions[:, 1]) - 25
        y_max = np.max(positions[:, 1]) + 25

        if (x_max - x_min) > (y_max - y_min):
            diff = (x_max - x_min) - (y_max - y_min)
            y_min -= diff / 2
            y_max += diff / 2
        else:
            diff = (y_max - y_min) - (x_max - x_min)
            x_min -= diff / 2
            x_max += diff / 2

        line_scale = (x_max - x_min) / 140 if x_max > x_min else 1.0
        lw = 0.35 / line_scale
        heading_lw = 0.25 / line_scale

        highlight_ids = set()
        if self.ego_vehicle is not None:
            highlight_ids.add(self.ego_vehicle.getID())
        opponent_ids = set(self.opponent_vehicle_ids) if self.opponent_vehicle_ids else set()

        for veh in vehicle_data:
            is_highlight = veh['id'] in highlight_ids
            is_opponent = (not is_highlight) and veh['id'] in opponent_ids
            if is_highlight:
                color = '#ff6b6b'
                alpha = 0.8
            elif is_opponent:
                color = '#4aa3ff'
                alpha = 0.8
            else:
                color = '#ffde8b'
                alpha = 0.5

            length = veh['length'] * 0.8
            width = veh['width'] * 0.8
            bbox_x_min = veh['x'] - width / 2
            bbox_y_min = veh['y'] - length / 2

            rectangle = mpatches.FancyBboxPatch(
                (bbox_x_min, bbox_y_min),
                width, length,
                ec='black', fc=color, linewidth=lw, alpha=alpha,
                boxstyle=mpatches.BoxStyle("Round", pad=0.3),
                zorder=4
            )

            tr = transforms.Affine2D().rotate_deg_around(
                veh['x'], veh['y'], math.degrees(veh['heading']) - 90
            ) + ax.transData
            rectangle.set_transform(tr)
            ax.add_patch(rectangle)

            heading_length = length / 2 + 1.5
            line_end_x = veh['x'] + heading_length * math.cos(veh['heading'])
            line_end_y = veh['y'] + heading_length * math.sin(veh['heading'])
            ax.plot(
                [veh['x'], line_end_x], [veh['y'], line_end_y],
                color='black', zorder=6, alpha=0.25, linewidth=heading_lw
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        fig.tight_layout()
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        fig.clear()

        return image
    
    def start_recording(self, output_dir: str, video_name: str, fps: int = 10, dpi: int = 100):
        """
        开始录制视频
        
        Args:
            output_dir: 输出目录
            video_name: 视频文件名（不含扩展名）
            fps: 帧率
            dpi: 分辨率
        """
        if self.video_recorder is None:
            self.video_recorder = NocturneVideoRecorder(
                output_dir=output_dir,
                fps=fps,
                dpi=dpi,
                delete_images=True
            )
        
        self.video_recorder.start_recording(video_name)
        self.recording_video = True
        
        # 捕获第一帧（初始状态）
        if self.scenario is not None and self.vehicles:
            self.video_recorder.capture_frame(
                self.scenario,
                self.vehicles,
                highlight_vehicle_ids=[self.ego_vehicle.getID()] if self.ego_vehicle else None
            )
    
    def stop_recording(self, video_name: Optional[str] = None) -> Optional[str]:
        """
        停止录制并保存视频
        
        Args:
            video_name: 视频文件名（如果与 start_recording 不同）
        
        Returns:
            视频文件路径，如果没有录制则返回 None
        """
        if not self.recording_video or self.video_recorder is None:
            return None
        
        self.recording_video = False
        
        try:
            if video_name is None:
                # 使用默认名称
                if self.current_level:
                    video_name = f"scenario_{self.current_level.scenario_id}"
                else:
                    video_name = "episode"
            
            video_path = self.video_recorder.save_video(video_name)
            return video_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None
    
    def close(self):
        """关闭环境"""
        # 如果正在录制，先停止
        if self.recording_video:
            self.stop_recording()
        
        # 清理录制器
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        
        # 清理 Nocturne 资源
        if self.sim is not None:
            # TODO: 清理 Nocturne 资源
            pass
