"""
Nocturne + CtRL-Sim 对抗环境

实现 DCD 框架要求的环境接口，支持：
- PLR (Prioritized Level Replay) 机制
- PAIRED / ACCEL 等 UED 算法（通过 Adversary 接口）
- 动态场景池大小
- Level 变异和编辑
"""
import gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

from .level import ScenarioLevel
from util.build_scenario_index import ScenarioIndex
from adapters.ctrl_sim import (
    CtrlSimOpponentAdapter,
    DataBridge,
    load_ctrl_sim_config,
    create_minimal_config,
)


# ========== Level 参数范围定义 ==========
# 参考 BipedalWalker 的 PARAM_RANGES_FULL
TILT_RANGE = [-25.0, 25.0]  # tilting 参数范围

# 变异时的扰动幅度
TILT_MUTATION_STD = 1.0

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
        
        # Level 参数向量（用于 adversary 构建）
        # [scenario_index, goal_tilt, veh_veh_tilt, veh_edge_tilt]
        self.level_params_vec = list(DEFAULT_LEVEL_PARAMS)
        
        # ========== 观测和动作空间（Student）==========
        self._obs_dim = kwargs.get('obs_dim', 128)
        self._action_dim = kwargs.get('action_dim', 2)
        
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
    
    # ========== Adversary 接口（PAIRED/ACCEL）==========
    
    def reset(self) -> Dict:
        """
        重置环境，准备 adversary 构建流程
        
        这是 PAIRED/Minimax 等 UED 算法的入口点。
        返回 adversary 观测，而非 student 观测。
        
        Returns:
            adversary 观测字典: {'image', 'time_step', 'random_z'}
        """
        self.step_count = 0
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
            # 将 [-1, 1] 映射到 [-25, 25]
            tilt_value = action * 25.0
            tilt_value = np.clip(tilt_value, TILT_RANGE[0], TILT_RANGE[1])
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
        scenario_idx = int(self.level_params_vec[0])
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
        
        # 设置随机种子
        np.random.seed(level.seed)
        
        # 加载 Nocturne 场景
        self._load_scenario(level.scenario_id)
        
        # 获取 ground truth 数据（需要添加 .json 后缀）
        self._gt_data_dict = self.data_bridge.get_ground_truth(
            self.scenario_data_dir, 
            f"{level.scenario_id}.json"
        )
        
        # 加载预处理数据
        self._preproc_data, _ = self.data_bridge.load_preprocessed_data(
            level.scenario_id
        )
        
        # 选择对手控制的车辆
        self._select_opponent_vehicles(k=self.opponent_k)
        
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
        # 格式: [scenario_idx, goal_tilt, veh_veh_tilt, veh_edge_tilt, seed]
        scenario_idx = int(float(encoding[0]))
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
        
        # 4. 仿真步进
        self.sim.step(self.dt)
        
        # 5. 计算奖励和终止条件
        obs = self._get_student_observation()
        reward = self._compute_reward()
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
        返回当前 level 的复杂度信息（用于日志和分析）
        """
        if self.current_level is None:
            return {}
        
        return {
            'scenario_id': self.current_level.scenario_id,
            'seed': self.current_level.seed,
            'opponent_k': self.opponent_k,
            'goal_tilt': self.current_level.goal_tilt,
            'veh_veh_tilt': self.current_level.veh_veh_tilt,
            'veh_edge_tilt': self.current_level.veh_edge_tilt,
            'scenario_pool_size': len(self.scenario_ids),
        }
    
    # ========== 内部辅助方法 ==========
    
    def _sample_random_level(self) -> ScenarioLevel:
        """随机生成 level"""
        return ScenarioLevel(
            scenario_id=np.random.choice(self.scenario_ids),
            seed=rand_int_seed(),
            goal_tilt=round(float(np.random.uniform(*TILT_RANGE)), 1),
            veh_veh_tilt=round(float(np.random.uniform(*TILT_RANGE)), 1),
            veh_edge_tilt=round(float(np.random.uniform(*TILT_RANGE)), 1),
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
        """
        from dataclasses import replace
        
        mutations = {}
        params = ['goal_tilt', 'veh_veh_tilt', 'veh_edge_tilt']
        
        for _ in range(num_edits):
            # 决定是否变异 scenario_id（30% 概率）
            if len(self.scenario_ids) > 1 and np.random.rand() < 0.3:
                other_scenarios = [s for s in self.scenario_ids if s != level.scenario_id]
                if other_scenarios:
                    mutations['scenario_id'] = np.random.choice(other_scenarios)
                    mutations['seed'] = rand_int_seed()
            else:
                # 变异 tilt 参数
                param = np.random.choice(params)
                current_val = mutations.get(param, getattr(level, param))
                direction = np.random.randint(-1, 2)  # -1, 0, 1
                mutation = direction * np.random.uniform(0, TILT_MUTATION_STD)
                new_val = np.clip(current_val + mutation, *TILT_RANGE)
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
        
        # 设置 ego 车辆（通常是第一个可控车辆）
        self.ego_vehicle = self._select_ego_vehicle()
    
    def _select_ego_vehicle(self):
        """选择 ego 车辆（学生控制）"""
        # TODO: 实现 ego 车辆选择逻辑
        # 通常选择场景中标记为 ego 的车辆
        for veh in self.vehicles:
            if hasattr(veh, 'is_ego') and veh.is_ego:
                return veh
        # 默认选择第一个车辆
        return self.vehicles[0] if self.vehicles else None
    
    def _select_opponent_vehicles(self, k: int):
        """
        选择对手控制的车辆（距离 ego 最近的 k 个）
        """
        if self.ego_vehicle is None:
            self.opponent_vehicles = []
            self.opponent_vehicle_ids = []
            return
        
        ego_pos = self.ego_vehicle.getPosition()
        ego_pos = np.array([ego_pos.x, ego_pos.y])
        
        # 计算所有车辆到 ego 的距离
        distances = []
        for veh in self.vehicles:
            if veh.getID() == self.ego_vehicle.getID():
                continue
            pos = veh.getPosition()
            dist = np.linalg.norm(np.array([pos.x, pos.y]) - ego_pos)
            distances.append((dist, veh))
        
        # 选择最近的 k 个
        distances.sort(key=lambda x: x[0])
        self.opponent_vehicles = [veh for _, veh in distances[:k]]
        self.opponent_vehicle_ids = [veh.getID() for veh in self.opponent_vehicles]
    
    def _get_vehicle_by_id(self, veh_id: int):
        """根据 ID 获取车辆对象"""
        for veh in self.vehicles:
            if veh.getID() == veh_id:
                return veh
        return None
    
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
    
    def _get_student_observation(self) -> np.ndarray:
        """
        获取学生策略的观测
        
        TODO: 实现实际的观测提取逻辑
        """
        # 占位符实现
        return np.zeros(self._obs_dim, dtype=np.float32)
    
    def _compute_reward(self) -> float:
        """
        计算学生奖励
        
        奖励组成：
        - 到达目标奖励
        - 碰撞惩罚
        - 进度奖励
        """
        reward = 0.0
        
        # TODO: 实现实际的奖励计算逻辑
        # 可以复用 ctrl-sim 的奖励计算（通过 DataBridge）
        
        self.episode_reward += reward
        return reward
    
    def _check_done(self) -> bool:
        """检查 episode 是否结束"""
        # 达到最大步数
        if self.current_step >= self.max_episode_steps:
            return True
        
        # TODO: 添加其他终止条件
        # - 碰撞
        # - 到达目标
        # - 超出边界
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """返回额外信息"""
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
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
        """渲染环境（可选）"""
        # TODO: 实现 Nocturne 渲染
        pass
    
    def close(self):
        """关闭环境"""
        if self.sim is not None:
            # TODO: 清理 Nocturne 资源
            pass
