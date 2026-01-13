# Find Interesting Pairs 工具使用说明

## 功能

这个工具用于扫描 Nocturne 场景 JSON 文件，识别哪些场景包含"interesting pairs"（有趣的车辆对）。

使用与 `envs/nocturne_ctrlsim/adversarial.py` 完全相同的筛选逻辑。

## 快速开始

### 方法1：使用运行脚本（推荐）

运行脚本会自动激活正确的conda环境：

```bash
# 基本使用（使用默认参数）
./run_find_interesting_pairs.sh --data_dir data/nocturne_waymo/formatted_json_v2_no_tl_valid

# 自定义筛选条件
./run_find_interesting_pairs.sh \
    --data_dir data/nocturne_waymo/formatted_json_v2_no_tl_valid \
    --goal_dist_threshold 15.0 \
    --timestep_diff_threshold 30 \
    --traj_len_threshold 50 \
    --save_results my_results.json

# 只处理前N个场景（快速测试）
./run_find_interesting_pairs.sh \
    --data_dir data/nocturne_waymo/formatted_json_v2_no_tl_valid \
    --max_scenarios 10
```

### 方法2：直接使用Python（需要先激活环境）

```bash
# 激活conda环境
conda activate dcd-ctrlsim

# 运行脚本
python find_interesting_pairs.py --data_dir data/nocturne_waymo/formatted_json_v2_no_tl_valid
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data/nocturne_waymo/formatted_json_v2_no_tl_valid` | JSON场景文件目录 |
| `--goal_dist_threshold` | `10.0` | 目标位置距离阈值（米） |
| `--timestep_diff_threshold` | `20` | 目标时间步差异阈值 |
| `--traj_len_threshold` | `60` | 最小轨迹长度阈值 |
| `--max_scenarios` | `None` | 最多处理的场景数量（None=全部） |
| `--save_results` | `results_interesting_pairs.json` | 结果保存文件 |

## 筛选条件说明

Interesting pair 的判断标准（与 adversarial.py 一致）：

1. **目标位置接近**：两辆车的目标位置距离 < `goal_dist_threshold` 米
2. **目标时间接近**：两辆车到达目标的时间步差异 < `timestep_diff_threshold` 步
3. **轨迹足够长**：两辆车的有效轨迹长度都 >= `traj_len_threshold` 步

## 输出说明

### 终端输出

```
================================================================================
Finding Interesting Pairs in Nocturne Scenarios
================================================================================
Data directory: data/nocturne_waymo/formatted_json_v2_no_tl_valid
Goal distance threshold: 10.0 meters
Timestep diff threshold: 20 steps
Trajectory length threshold: 60 steps
================================================================================

Found 150 scenario files
Scanning scenarios: 100%|████████████████████| 150/150 [02:30<00:00,  1.00s/it]

================================================================================
RESULTS
================================================================================
Total scenarios: 150
Scenarios WITH interesting pair: 45 (30.0%)
Scenarios WITHOUT interesting pair: 105 (70.0%)

Examples of scenarios WITH interesting pairs:
  - tfrecord-00001-of-00150_42: pair=(12, 35), num_moving=8
  - tfrecord-00002-of-00150_15: pair=(7, 23), num_moving=12
  ...

Examples of scenarios WITHOUT interesting pairs:
  - tfrecord-00009-of-00150_63
  - tfrecord-00010-of-00150_88
  ...

Results saved to: results_interesting_pairs.json
================================================================================
```

### JSON 结果文件

保存的 JSON 文件包含：

```json
{
  "scenarios_with_pair": [
    "tfrecord-00001-of-00150_42",
    "tfrecord-00002-of-00150_15",
    ...
  ],
  "scenarios_without_pair": [
    "tfrecord-00009-of-00150_63",
    ...
  ],
  "pair_details": {
    "tfrecord-00001-of-00150_42": {
      "pair": [12, 35],
      "num_moving_vehicles": 8
    },
    ...
  },
  "statistics": {
    "total": 150,
    "with_pair": 45,
    "without_pair": 105,
    "error": 0
  }
}
```

## 使用场景

1. **验证数据集质量**：检查有多少场景包含有意义的交互场景
2. **调整筛选参数**：通过调整阈值找到合适的筛选条件
3. **场景预筛选**：为训练/评估选择合适的场景子集
4. **问题排查**：定位为什么某些场景没有 interesting pair

## 注意事项

1. 脚本需要加载和仿真每个场景，处理时间较长（约1秒/场景）
2. 使用 `--max_scenarios` 参数可以快速测试
3. 筛选条件参数可以灵活调整以适应不同需求
4. 结果会自动保存到 JSON 文件，方便后续分析

## 故障排除

### 问题：找不到数据目录

**错误信息**：`Error: Data directory not found`

**解决方案**：检查 `--data_dir` 路径是否正确

### 问题：No JSON files found

**解决方案**：确认目录下有 `.json` 文件

### 问题：导入错误

**解决方案**：确保在项目根目录运行脚本，且已安装所有依赖
