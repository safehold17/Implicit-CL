#!/usr/bin/env python3
"""
CtrlSim 推理诊断脚本

按顺序检查以下问题：
1. 模型输入输出日志 - 验证模型推理是否合理
2. GT动作对比 - 确定问题在模型还是环境
3. 预处理数据完整性 - 确保数据正确加载
4. 坐标系一致性 - 排除坐标转换问题

运行方式:
    source /home/chen/miniconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate dcd-ctrlsim
    python tools/diagnose_ctrlsim_inference.py \
        --scenario_index_path data/scenarios_index_valid.json \
        --scenario_data_dir /path/to/nocturne_waymo \
        --preprocess_dir /path/to/preprocess \
        --checkpoint_path checkpoints/model.ckpt
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ctrlsim_adapter.config_loader import create_minimal_config
from ctrlsim_adapter.data_bridge import DataBridge
from ctrlsim_adapter.opponent_vehicle import CtrlSimOpponentAdapter
from batch_inference import ExternalTeacher
from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial


class DiagnosticResults:
    """诊断结果收集器"""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.warnings = []
    
    def add_result(self, test_name: str, passed: bool, details: Dict = None):
        self.results[test_name] = {
            'passed': passed,
            'details': details or {}
        }
        
    def add_issue(self, issue: str):
        self.issues.append(issue)
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("诊断结果总结")
        print("=" * 80)
        
        for name, result in self.results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n[{status}] {name}")
            if result['details']:
                for key, value in result['details'].items():
                    print(f"    {key}: {value}")
        
        if self.issues:
            print("\n" + "-" * 40)
            print("发现的问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings:
            print("\n" + "-" * 40)
            print("警告:")
            for w in self.warnings:
                print(f"  ⚠ {w}")
        
        print("\n" + "=" * 80)


def _predict_opponent_actions_batch(
    env: NocturneCtrlSimAdversarial,
    teacher: ExternalTeacher,
) -> Dict[int, Tuple[float, float]]:
    prepared = env.opponent.prepare_step(
        env.current_step,
        env.vehicles,
    )
    if prepared is None:
        return {}
    outputs = teacher.batched_forward([prepared])[0]
    return env.opponent.apply_predictions(outputs)


def test_1_model_input_output(
    env: NocturneCtrlSimAdversarial,
    teacher: ExternalTeacher,
    results: DiagnosticResults,
    num_steps: int = 15  # 需要超过 history_steps (默认10) 才能看到模型动作
) -> Dict[str, Any]:
    """
    测试1: 检查模型输入输出
    
    验证:
    - 输入数据格式是否正确
    - 输出动作范围是否合理
    - RTG值是否正常
    
    注意：前 history_steps-1 步使用 GT 动作（warm-up），之后才是模型动作
    """
    print("\n" + "=" * 60)
    print("测试 1: 模型输入输出检查")
    print("=" * 60)
    
    collected_data = {
        'vehicle_data': [],
        'actions': [],
        'rtgs': [],
        'positions': [],
        'goals': [],
    }
    
    # Reset environment
    env.reset_random()
    
    # 获取对手适配器的内部状态
    opponent = env.opponent
    
    print(f"\n=== 环境初始化信息 ===")
    print(f"场景 ID: {env.current_level.scenario_id}")
    print(f"Ego vehicle ID: {env.ego_vehicle.getID() if env.ego_vehicle else 'None'}")
    print(f"Opponent vehicle IDs: {env.opponent_vehicle_ids}")
    print(f"总车辆数: {len(env.vehicles)}")
    
    # 检查初始化后的 vehicle_data_dict
    if hasattr(opponent, '_vehicle_data_dict') and opponent._vehicle_data_dict:
        print(f"\n=== Vehicle Data Dict 初始化检查 ===")
        for veh_id in list(opponent._vehicle_data_dict.keys())[:3]:  # 只打印前3个
            vd = opponent._vehicle_data_dict[veh_id]
            print(f"\nVehicle {veh_id}:")
            print(f"  Goal position: {vd.get('goal_position', 'N/A')}")
            print(f"  Goal heading: {vd.get('goal_heading', 'N/A')}")
            print(f"  Goal speed: {vd.get('goal_speed', 'N/A')}")
            print(f"  Vehicle width: {vd.get('width', 'N/A')}")
            print(f"  Vehicle length: {vd.get('length', 'N/A')}")
            print(f"  Vehicle type: {vd.get('type', 'N/A')}")
            
            # 检查 RTG 初始值
            if 'rtgs' in vd and len(vd['rtgs']) > 0:
                print(f"  Initial RTG: {vd['rtgs'][0]}")
            else:
                print(f"  Initial RTG: NOT SET")
                results.add_issue(f"Vehicle {veh_id} 没有初始化 RTG 值")
    else:
        results.add_issue("opponent._vehicle_data_dict 为空或未初始化")
    
    # 获取 history_steps 参数
    history_steps = getattr(opponent, 'history_steps', 10)
    print(f"\n=== 关键参数 ===")
    print(f"history_steps: {history_steps}")
    print(f"注意: 前 {history_steps-1} 步使用 GT 动作（warm-up），之后才是模型预测动作")
    
    # 执行几步并记录
    print(f"\n=== 执行 {num_steps} 步推理 ===")
    
    warmup_actions = []  # GT 动作
    model_actions = []   # 模型动作
    for step in range(num_steps):
        t = env.current_step
        is_warmup = t < history_steps - 1
        phase = "WARMUP (GT)" if is_warmup else "MODEL"
        
        print(f"\n--- Step {step} (t={t}, {phase}) ---")
        
        # 随机动作（会被忽略，因为我们关注的是对手）
        action = np.zeros(2, dtype=np.float32)
        
        opponent_actions = _predict_opponent_actions_batch(env, teacher)
        
        # ✅ 记录动作到 vehicle_data_dict（必须在下一次 step 之前调用）
        if hasattr(opponent, 'record_all_actions'):
            opponent.record_all_actions(t, env.vehicles, opponent_actions)
        
        print(f"对手动作数: {len(opponent_actions)}")
        
        action_values = []
        for veh_id, (accel, steer) in opponent_actions.items():
            print(f"  Vehicle {veh_id}: accel={accel:.4f}, steer={steer:.4f}")
            action_values.append((accel, steer))
            
            # 分类收集
            if is_warmup:
                warmup_actions.append((accel, steer))
            else:
                model_actions.append((accel, steer))
            
            # 检查动作范围
            if abs(accel) > 15.0:
                results.add_warning(f"Step {step}: Vehicle {veh_id} 加速度异常大: {accel}")
            if abs(steer) > 1.5:
                results.add_warning(f"Step {step}: Vehicle {veh_id} 转向角异常大: {steer}")
            if accel == 0.0 and steer == 0.0 and not is_warmup:
                results.add_warning(f"Step {step}: Vehicle {veh_id} 模型动作为 (0, 0)，可能表示推理失败")
        
        collected_data['actions'].append(action_values)
        
        # 检查 RTG 更新
        if hasattr(opponent, '_vehicle_data_dict'):
            for veh_id in env.opponent_vehicle_ids[:2]:
                if veh_id in opponent._vehicle_data_dict:
                    vd = opponent._vehicle_data_dict[veh_id]
                    if 'rtgs' in vd and len(vd['rtgs']) > 0:
                        latest_rtg = vd['rtgs'][-1]
                        if step < 3 or step == history_steps - 1 or step == history_steps:
                            print(f"  Vehicle {veh_id} latest RTG: {latest_rtg}")
        
        # 执行环境步
        # 注意：这里不用 env.step 因为我们只想诊断对手
        env.sim.step(env.dt)
        env.current_step += 1
    
    # 分析收集的数据
    all_actions = [a for step_actions in collected_data['actions'] for a in step_actions]
    if all_actions:
        accels = [a[0] for a in all_actions]
        steers = [a[1] for a in all_actions]
        
        details = {
            'total_steps': num_steps,
            'warmup_steps': history_steps - 1,
            'accel_mean': np.mean(accels),
            'accel_std': np.std(accels),
            'accel_range': (min(accels), max(accels)),
            'steer_mean': np.mean(steers),
            'steer_std': np.std(steers),
            'steer_range': (min(steers), max(steers)),
        }
        
        print(f"\n=== 全部动作统计 ===")
        print(f"加速度: mean={details['accel_mean']:.4f}, std={details['accel_std']:.4f}, "
              f"range={details['accel_range']}")
        print(f"转向角: mean={details['steer_mean']:.4f}, std={details['steer_std']:.4f}, "
              f"range={details['steer_range']}")
        
        # 单独分析模型动作（非 warmup）
        if model_actions:
            model_accels = [a[0] for a in model_actions]
            model_steers = [a[1] for a in model_actions]
            
            print(f"\n=== 模型动作统计（排除 warm-up）===")
            print(f"模型动作数: {len(model_actions)}")
            print(f"加速度: mean={np.mean(model_accels):.4f}, std={np.std(model_accels):.4f}, "
                  f"range=({min(model_accels):.4f}, {max(model_accels):.4f})")
            print(f"转向角: mean={np.mean(model_steers):.4f}, std={np.std(model_steers):.4f}, "
                  f"range=({min(model_steers):.4f}, {max(model_steers):.4f})")
            
            details['model_accel_mean'] = np.mean(model_accels)
            details['model_accel_std'] = np.std(model_accels)
            details['model_steer_mean'] = np.mean(model_steers)
            details['model_steer_std'] = np.std(model_steers)
            
            # 判断模型动作是否正常
            passed = (
                details['model_accel_std'] > 0.01 and  # 动作有变化
                details['model_steer_std'] > 0.001 and
                abs(details['model_accel_mean']) < 10.0 and
                abs(details['model_steer_mean']) < 1.0
            )
            
            if not passed:
                if details['model_accel_std'] < 0.01:
                    results.add_issue("模型加速度几乎没有变化，模型可能没有正确推理")
                if details['model_steer_std'] < 0.001:
                    results.add_issue("模型转向角几乎没有变化，模型可能没有正确推理")
        else:
            passed = False
            results.add_issue(f"没有收集到模型动作（需要运行超过 {history_steps-1} 步）")
        
        results.add_result("模型输入输出检查", passed, details)
    else:
        results.add_result("模型输入输出检查", False, {'error': '没有收集到任何动作'})
        results.add_issue("模型推理没有返回任何动作")
    
    return collected_data


def test_2_gt_action_comparison(
    env: NocturneCtrlSimAdversarial,
    teacher: ExternalTeacher,
    results: DiagnosticResults,
    num_steps: int = 10
) -> Dict[str, Any]:
    """
    测试2: GT动作对比
    
    验证:
    - 使用GT动作时车辆是否正常行驶
    - GT动作与模型动作的差异
    """
    print("\n" + "=" * 60)
    print("测试 2: GT动作对比")
    print("=" * 60)
    
    # Reset environment
    env.reset_random()
    
    opponent = env.opponent
    scenario_id = env.current_level.scenario_id
    
    print(f"\n场景: {scenario_id}")
    print(f"Opponent IDs: {env.opponent_vehicle_ids}")
    
    gt_actions_collected = []
    model_actions_collected = []
    position_errors = []
    
    for step in range(num_steps):
        t = env.current_step
        
        model_actions = _predict_opponent_actions_batch(env, teacher)
        
        # ✅ 记录动作到 vehicle_data_dict（必须在下一次 step 之前调用）
        if hasattr(opponent, 'record_all_actions'):
            opponent.record_all_actions(t, env.vehicles, model_actions)
        
        # 获取GT动作
        gt_actions = {}
        for veh_id in env.opponent_vehicle_ids:
            veh = env._get_vehicle_by_id(veh_id)
            if veh is not None:
                gt_action = opponent._get_gt_action(veh_id, t, veh)
                gt_actions[veh_id] = gt_action
        
        # 记录并比较
        if step < 3:  # 只打印前3步
            print(f"\n--- Step {step} (t={t}) ---")
            
        for veh_id in env.opponent_vehicle_ids[:3]:
            if veh_id in model_actions and veh_id in gt_actions:
                m_accel, m_steer = model_actions[veh_id]
                g_accel, g_steer = gt_actions[veh_id]
                
                model_actions_collected.append((m_accel, m_steer))
                gt_actions_collected.append((g_accel, g_steer))
                
                accel_diff = abs(m_accel - g_accel)
                steer_diff = abs(m_steer - g_steer)
                
                if step < 3:
                    print(f"  Vehicle {veh_id}:")
                    print(f"    Model: accel={m_accel:.4f}, steer={m_steer:.4f}")
                    print(f"    GT:    accel={g_accel:.4f}, steer={g_steer:.4f}")
                    print(f"    Diff:  accel={accel_diff:.4f}, steer={steer_diff:.4f}")
        
        # 应用GT动作并步进（测试GT模式）
        for veh_id, (g_accel, g_steer) in gt_actions.items():
            veh = env._get_vehicle_by_id(veh_id)
            if veh is not None:
                opponent.apply_action(veh, (g_accel, g_steer))
        
        env.sim.step(env.dt)
        env.current_step += 1
    
    # 分析差异
    if model_actions_collected and gt_actions_collected:
        model_accels = [a[0] for a in model_actions_collected]
        model_steers = [a[1] for a in model_actions_collected]
        gt_accels = [a[0] for a in gt_actions_collected]
        gt_steers = [a[1] for a in gt_actions_collected]
        
        accel_mse = np.mean((np.array(model_accels) - np.array(gt_accels))**2)
        steer_mse = np.mean((np.array(model_steers) - np.array(gt_steers))**2)
        
        details = {
            'accel_mse': accel_mse,
            'steer_mse': steer_mse,
            'model_accel_mean': np.mean(model_accels),
            'gt_accel_mean': np.mean(gt_accels),
            'model_steer_mean': np.mean(model_steers),
            'gt_steer_mean': np.mean(gt_steers),
        }
        
        print(f"\n=== GT vs Model 统计 ===")
        print(f"加速度 MSE: {accel_mse:.6f}")
        print(f"转向角 MSE: {steer_mse:.6f}")
        print(f"Model 加速度均值: {np.mean(model_accels):.4f}, GT 均值: {np.mean(gt_accels):.4f}")
        print(f"Model 转向角均值: {np.mean(model_steers):.4f}, GT 均值: {np.mean(gt_steers):.4f}")
        
        # 判断差异是否过大
        # 注意：一定程度的差异是预期的，因为模型是学习而非复制GT
        passed = True
        if accel_mse > 50.0:
            results.add_warning(f"加速度 MSE 较大 ({accel_mse:.2f})，模型动作与GT差异显著")
        if steer_mse > 0.5:
            results.add_warning(f"转向角 MSE 较大 ({steer_mse:.4f})，模型动作与GT差异显著")
        
        # 如果GT动作本身正常但模型动作异常，说明问题在模型
        if abs(np.mean(gt_accels)) < 5.0 and abs(np.mean(model_accels)) > 10.0:
            results.add_issue("GT动作正常但模型动作异常大，问题可能在模型推理")
            passed = False
        
        results.add_result("GT动作对比", passed, details)
    else:
        results.add_result("GT动作对比", False, {'error': '无法收集动作数据'})
    
    return {}


def test_3_preprocessed_data(
    env: NocturneCtrlSimAdversarial,
    results: DiagnosticResults
) -> Dict[str, Any]:
    """
    测试3: 预处理数据检查
    
    验证:
    - RTG数据是否正确加载
    - road_points 是否存在
    - 数据维度是否匹配
    - RTG 初始化逻辑是否正确
    """
    print("\n" + "=" * 60)
    print("测试 3: 预处理数据检查")
    print("=" * 60)
    
    # Reset to get fresh data
    env.reset_random()
    
    preproc = env._preproc_data
    scenario_id = env.current_level.scenario_id
    opponent = env.opponent
    
    print(f"\n场景: {scenario_id}")
    
    issues = []
    details = {}
    
    if preproc is None:
        results.add_issue(f"场景 {scenario_id} 的预处理数据为 None")
        results.add_result("预处理数据检查", False, {'error': 'preproc_data is None'})
        return {}
    
    print(f"\n=== 预处理数据结构 ===")
    print(f"Type: {type(preproc)}")
    
    # 处理不同类型的预处理数据
    if isinstance(preproc, dict):
        print(f"Keys: {list(preproc.keys())}")
        rtgs = preproc.get('rtgs')
        road_points = preproc.get('road_points')
        road_types = preproc.get('road_types')
    else:
        # 可能是 MotionData 对象或其他格式
        print(f"Attributes: {dir(preproc)}")
        rtgs = getattr(preproc, 'rtgs', None)
        road_points = getattr(preproc, 'road_points', None)
        road_types = getattr(preproc, 'road_types', None)
        
        # 尝试从 agent 属性获取
        if rtgs is None and hasattr(preproc, 'agent'):
            agent_data = preproc.agent
            rtgs = getattr(agent_data, 'rtgs', None)
    
    # RTG 检查
    print(f"\n=== RTG 数据详细检查 ===")
    if rtgs is not None:
        if isinstance(rtgs, torch.Tensor):
            rtgs_np = rtgs.cpu().numpy()
        else:
            rtgs_np = rtgs
        
        print(f"RTG shape: {rtgs_np.shape}")
        print(f"RTG dtype: {rtgs_np.dtype}")
        
        # 检查RTG形状
        # 预期形状: (num_agents, steps, num_reward_components) 
        # 或 (batch, num_agents, steps, num_reward_components)
        if len(rtgs_np.shape) == 3:
            num_agents, num_steps, num_components = rtgs_np.shape
            print(f"  num_agents: {num_agents}")
            print(f"  num_steps: {num_steps}")
            print(f"  num_components: {num_components}")
            
            # 检查 ctrl-sim 期望的 reward 维度
            # ctrl-sim 使用 5 个 reward component，但只取 [0, 3, 4] -> goal, veh, edge
            if num_components != 5:
                results.add_warning(f"RTG components = {num_components}，预期为 5")
            
            # 打印第一个 agent 的 RTG 样本
            print(f"\n  第一个 agent 的 RTG (前5个时间步):")
            for t in range(min(5, num_steps)):
                print(f"    t={t}: {rtgs_np[0, t]}")
            
        elif len(rtgs_np.shape) == 4:
            batch, num_agents, num_steps, num_components = rtgs_np.shape
            print(f"  batch: {batch}")
            print(f"  num_agents: {num_agents}")
            print(f"  num_steps: {num_steps}")
            print(f"  num_components: {num_components}")
        else:
            results.add_issue(f"RTG shape 异常: {rtgs_np.shape}")
        
        # 检查 RTG 值范围
        rtg_min = rtgs_np.min()
        rtg_max = rtgs_np.max()
        rtg_mean = rtgs_np.mean()
        
        print(f"\n  RTG 统计: min={rtg_min:.2f}, max={rtg_max:.2f}, mean={rtg_mean:.2f}")
        
        details['rtg_shape'] = str(rtgs_np.shape)
        details['rtg_range'] = f"[{rtg_min:.2f}, {rtg_max:.2f}]"
        details['rtg_mean'] = rtg_mean
        
        if np.isnan(rtgs_np).any():
            results.add_issue("RTG 数据包含 NaN")
            issues.append("RTG contains NaN")
    else:
        results.add_issue("RTG 数据未找到")
        issues.append("RTG not found")
        print("RTG: NOT FOUND")
    
    # 检查 opponent adapter 中 RTG 的使用方式
    print(f"\n=== Opponent Adapter RTG 使用检查 ===")
    if hasattr(opponent, '_preproc_data') and opponent._preproc_data is not None:
        adapter_preproc = opponent._preproc_data
        print(f"Adapter preproc type: {type(adapter_preproc)}")
        
        if isinstance(adapter_preproc, dict):
            adapter_rtgs = adapter_preproc.get('rtgs')
        else:
            adapter_rtgs = getattr(adapter_preproc, 'rtgs', None)
            if adapter_rtgs is None and hasattr(adapter_preproc, 'agent'):
                adapter_rtgs = getattr(adapter_preproc.agent, 'rtgs', None)
        
        if adapter_rtgs is not None:
            if isinstance(adapter_rtgs, torch.Tensor):
                adapter_rtgs = adapter_rtgs.cpu().numpy()
            print(f"Adapter RTG shape: {adapter_rtgs.shape}")
            
            # 关键检查：RTG 索引与车辆顺序是否匹配
            print(f"\n=== 关键检查：RTG 索引与车辆 ID 映射 ===")
            print(f"环境车辆顺序 (前5个):")
            for idx, veh in enumerate(env.vehicles[:5]):
                veh_id = veh.getID()
                print(f"  veh_idx={idx}, veh_id={veh_id}")
            
            # 检查预处理数据中是否有 filtered_ag_ids
            filtered_ids = None
            if isinstance(adapter_preproc, dict):
                filtered_ids = adapter_preproc.get('filtered_ag_ids')
            else:
                filtered_ids = getattr(adapter_preproc, 'filtered_ag_ids', None)
            
            if filtered_ids is not None:
                print(f"\n预处理数据中的 filtered_ag_ids: {filtered_ids[:5] if len(filtered_ids) > 5 else filtered_ids}")
                
                # 检查是否匹配
                env_veh_ids = [v.getID() for v in env.vehicles]
                if len(filtered_ids) > 0:
                    if list(filtered_ids[:len(env_veh_ids)]) != env_veh_ids:
                        results.add_issue(
                            "车辆 ID 顺序与预处理数据中的 agent 顺序不匹配！"
                            "这会导致 RTG 被错误分配给错误的车辆。"
                        )
                        print(f"  ⚠ 环境车辆 IDs: {env_veh_ids[:5]}")
                        print(f"  ⚠ 预处理 agent IDs: {list(filtered_ids[:5])}")
            else:
                results.add_warning(
                    "预处理数据中没有 filtered_ag_ids，无法验证车辆顺序是否正确"
                )
        else:
            print("Adapter RTG: NOT FOUND in preproc")
            results.add_warning("Opponent adapter 的预处理数据中没有找到 rtgs")
    
    # 检查 _update_vehicle_data_dict 中 RTG 初始化逻辑
    print(f"\n=== Vehicle Data Dict RTG 初始化检查 ===")
    if hasattr(opponent, '_vehicle_data_dict') and opponent._vehicle_data_dict:
        for veh_id in list(opponent._vehicle_data_dict.keys())[:3]:
            vd = opponent._vehicle_data_dict[veh_id]
            rtg_list = vd.get('rtgs', [])
            if rtg_list:
                print(f"Vehicle {veh_id} RTG history length: {len(rtg_list)}")
                print(f"  First RTG: {rtg_list[0]}")
                if len(rtg_list) > 1:
                    print(f"  Last RTG: {rtg_list[-1]}")
            else:
                print(f"Vehicle {veh_id}: NO RTG history")
                results.add_warning(f"Vehicle {veh_id} 的 RTG 历史为空")
    
    # Road points 检查
    print(f"\n=== Road Data ===")
    if road_points is not None:
        if isinstance(road_points, torch.Tensor):
            road_points = road_points.cpu().numpy()
        
        print(f"road_points shape: {road_points.shape}")
        details['road_points_shape'] = str(road_points.shape)
        
        if road_points.shape[0] == 0:
            results.add_warning("road_points 为空")
    else:
        results.add_warning("road_points 未找到")
        print("road_points: NOT FOUND")
    
    if road_types is not None:
        if isinstance(road_types, torch.Tensor):
            road_types = road_types.cpu().numpy()
        print(f"road_types shape: {road_types.shape}")
        details['road_types_shape'] = str(road_types.shape)
    else:
        print("road_types: NOT FOUND")
    
    # 检查车辆数量匹配
    print(f"\n=== 数据一致性检查 ===")
    num_vehicles = len(env.vehicles)
    print(f"环境中车辆数: {num_vehicles}")
    print(f"GT data dict 中车辆数: {len(env._gt_data_dict)}")
    
    if rtgs is not None:
        if isinstance(rtgs, torch.Tensor):
            rtgs = rtgs.cpu().numpy()
        if len(rtgs.shape) == 4:
            rtg_agents = rtgs.shape[1]
        else:
            rtg_agents = rtgs.shape[0]
        print(f"RTG 中 agent 数: {rtg_agents}")
        
        if rtg_agents < num_vehicles:
            results.add_issue(
                f"关键问题：RTG agents ({rtg_agents}) < 环境车辆数 ({num_vehicles})！\n"
                f"  这会导致索引越界错误。当 veh_idx >= {rtg_agents} 时，\n"
                f"  访问 preproc_data['rtgs'][veh_idx, t] 会失败。\n"
                f"  解决方案：需要为所有车辆提供 RTG 数据，或在代码中添加边界检查。"
            )
        elif rtg_agents > num_vehicles:
            results.add_warning(
                f"RTG agents ({rtg_agents}) > 环境车辆数 ({num_vehicles})，有冗余数据"
            )
    
    passed = len(issues) == 0
    results.add_result("预处理数据检查", passed, details)
    
    return {'preproc': preproc}


def test_4_coordinate_system(
    env: NocturneCtrlSimAdversarial,
    results: DiagnosticResults
) -> Dict[str, Any]:
    """
    测试4: 坐标系一致性检查
    
    验证:
    - Nocturne 车辆位置与 GT 数据位置匹配
    - 目标位置正确性
    - 朝向一致性
    """
    print("\n" + "=" * 60)
    print("测试 4: 坐标系一致性检查")
    print("=" * 60)
    
    env.reset_random()
    
    gt_data = env._gt_data_dict
    opponent = env.opponent
    
    print(f"\n场景: {env.current_level.scenario_id}")
    
    position_errors = []
    heading_errors = []
    goal_errors = []
    
    print(f"\n=== 初始位置对比 (t=0) ===")
    
    for veh in env.vehicles[:5]:  # 只检查前5个车辆
        veh_id = veh.getID()
        
        # Nocturne 位置
        nocturne_pos = veh.getPosition()
        nocturne_heading = veh.getHeading()
        nocturne_speed = veh.getSpeed()
        
        # GT 位置
        if veh_id in gt_data:
            gt_traj = np.array(gt_data[veh_id]['traj'])
            gt_pos_x, gt_pos_y = gt_traj[0, 0], gt_traj[0, 1]
            gt_heading = gt_traj[0, 2]
            gt_speed = gt_traj[0, 3]
            gt_existence = gt_traj[0, 4]
            
            # 计算误差
            pos_error = np.sqrt((nocturne_pos.x - gt_pos_x)**2 + (nocturne_pos.y - gt_pos_y)**2)
            heading_error = abs(nocturne_heading - gt_heading)
            # 处理角度循环
            if heading_error > np.pi:
                heading_error = 2 * np.pi - heading_error
            
            position_errors.append(pos_error)
            heading_errors.append(heading_error)
            
            print(f"\nVehicle {veh_id} (existence={gt_existence}):")
            print(f"  Nocturne pos: ({nocturne_pos.x:.4f}, {nocturne_pos.y:.4f})")
            print(f"  GT pos:       ({gt_pos_x:.4f}, {gt_pos_y:.4f})")
            print(f"  Position error: {pos_error:.6f}")
            print(f"  Nocturne heading: {nocturne_heading:.4f}, GT heading: {gt_heading:.4f}")
            print(f"  Heading error: {heading_error:.6f} rad ({np.degrees(heading_error):.2f} deg)")
            print(f"  Nocturne speed: {nocturne_speed:.4f}, GT speed: {gt_speed:.4f}")
            
            # 检查目标
            target_pos = veh.target_position
            gt_goal = gt_traj[-1, :2] if gt_traj[-1, 4] else gt_traj[0, 5:7]  # 使用最后有效位置或原始目标
            
            goal_error = np.sqrt((target_pos.x - gt_goal[0])**2 + (target_pos.y - gt_goal[1])**2)
            goal_errors.append(goal_error)
            
            print(f"  Target pos: ({target_pos.x:.4f}, {target_pos.y:.4f})")
            
            # 警告大误差
            if pos_error > 0.1:
                results.add_warning(f"Vehicle {veh_id} 初始位置误差较大: {pos_error:.4f}")
            if heading_error > 0.1:
                results.add_warning(f"Vehicle {veh_id} 初始朝向误差较大: {np.degrees(heading_error):.2f} deg")
    
    # 统计
    details = {
        'mean_position_error': np.mean(position_errors) if position_errors else float('nan'),
        'max_position_error': max(position_errors) if position_errors else float('nan'),
        'mean_heading_error_deg': np.degrees(np.mean(heading_errors)) if heading_errors else float('nan'),
        'max_heading_error_deg': np.degrees(max(heading_errors)) if heading_errors else float('nan'),
    }
    
    print(f"\n=== 统计 ===")
    print(f"平均位置误差: {details['mean_position_error']:.6f}")
    print(f"最大位置误差: {details['max_position_error']:.6f}")
    print(f"平均朝向误差: {details['mean_heading_error_deg']:.4f} deg")
    print(f"最大朝向误差: {details['max_heading_error_deg']:.4f} deg")
    
    # 判断是否通过
    passed = True
    if details['max_position_error'] > 1.0:
        results.add_issue("位置误差过大，坐标系可能不一致")
        passed = False
    if details['max_heading_error_deg'] > 10.0:
        results.add_issue("朝向误差过大，坐标系可能不一致")
        passed = False
    
    results.add_result("坐标系一致性检查", passed, details)
    
    return {}


def test_5_model_checkpoint(
    checkpoint_path: str,
    device: str,
    results: DiagnosticResults
) -> Dict[str, Any]:
    """
    测试5: 模型 checkpoint 检查
    
    验证:
    - Checkpoint 文件是否存在
    - 模型是否正确加载
    - 模型参数是否正常
    """
    print("\n" + "=" * 60)
    print("测试 5: 模型 Checkpoint 检查")
    print("=" * 60)
    
    from models.ctrl_sim import CtRLSim
    
    print(f"\nCheckpoint 路径: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        results.add_issue(f"Checkpoint 文件不存在: {checkpoint_path}")
        results.add_result("模型Checkpoint检查", False, {'error': 'file not found'})
        return {}
    
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    details = {'file_size_mb': file_size}
    
    try:
        # 加载模型
        print("\n加载模型...")
        model = CtRLSim.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        details['total_params'] = total_params
        details['trainable_params'] = trainable_params
        
        # 检查参数值
        print("\n=== 模型参数统计 ===")
        nan_count = 0
        inf_count = 0
        zero_count = 0
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_count += 1
                print(f"  WARNING: {name} contains NaN")
            if torch.isinf(param).any():
                inf_count += 1
                print(f"  WARNING: {name} contains Inf")
            if (param == 0).all():
                zero_count += 1
                # 不是所有全零参数都是问题，比如 bias
        
        print(f"\n参数层含 NaN: {nan_count}")
        print(f"参数层含 Inf: {inf_count}")
        print(f"全零参数层: {zero_count}")
        
        if nan_count > 0:
            results.add_issue(f"模型包含 {nan_count} 个 NaN 参数层")
        if inf_count > 0:
            results.add_issue(f"模型包含 {inf_count} 个 Inf 参数层")
        
        # 检查模型配置
        if hasattr(model, 'cfg'):
            print(f"\n=== 模型配置 ===")
            model_cfg = model.cfg
            if hasattr(model_cfg, 'model'):
                print(f"Hidden dim: {getattr(model_cfg.model, 'hidden_dim', 'N/A')}")
                print(f"Num reward components: {getattr(model_cfg.model, 'num_reward_components', 'N/A')}")
            if hasattr(model_cfg, 'dataset') and hasattr(model_cfg.dataset, 'waymo'):
                waymo_cfg = model_cfg.dataset.waymo
                print(f"Train context length: {getattr(waymo_cfg, 'train_context_length', 'N/A')}")
                print(f"Max num agents: {getattr(waymo_cfg, 'max_num_agents', 'N/A')}")
        
        passed = nan_count == 0 and inf_count == 0
        results.add_result("模型Checkpoint检查", passed, details)
        
    except Exception as e:
        results.add_issue(f"加载模型失败: {str(e)}")
        results.add_result("模型Checkpoint检查", False, {'error': str(e)})
        import traceback
        traceback.print_exc()
    
    return {}


def test_6_config_consistency(
    cfg,
    results: DiagnosticResults
) -> Dict[str, Any]:
    """
    测试6: 配置一致性检查
    
    验证关键配置参数是否正确
    """
    print("\n" + "=" * 60)
    print("测试 6: 配置一致性检查")
    print("=" * 60)
    
    from omegaconf import OmegaConf
    
    details = {}
    issues = []
    
    # 关键配置项
    key_configs = [
        ('nocturne.steps', 90, 'episode 步数'),
        ('nocturne.dt', 0.1, '时间步长'),
        ('nocturne.history_steps', 10, '历史步数'),
        ('dataset.waymo.train_context_length', 32, '上下文长度'),
        ('dataset.waymo.max_num_agents', 24, '最大 agent 数'),
        ('model.num_reward_components', 3, 'reward 维度'),
    ]
    
    print("\n=== 关键配置 ===")
    
    for key, expected, desc in key_configs:
        value = OmegaConf.select(cfg, key, default=None)
        status = "✓" if value == expected else "?"
        print(f"{status} {desc} ({key}): {value} (expected: {expected})")
        
        details[key] = value
        
        if value is None:
            issues.append(f"配置 {key} 未设置")
        elif value != expected:
            results.add_warning(f"配置 {key}={value} 与预期 {expected} 不同")
    
    # 检查动作空间配置
    action_configs = [
        ('dataset.waymo.max_accel', 10.0, '最大加速度'),
        ('dataset.waymo.min_accel', -10.0, '最小加速度'),
        ('dataset.waymo.max_steer', 0.7, '最大转向'),
        ('dataset.waymo.min_steer', -0.7, '最小转向'),
        ('dataset.waymo.accel_discretization', None, '加速度离散化数'),
        ('dataset.waymo.steer_discretization', None, '转向离散化数'),
    ]
    
    print("\n=== 动作空间配置 ===")
    for key, expected, desc in action_configs:
        value = OmegaConf.select(cfg, key, default=None)
        if expected is not None:
            status = "✓" if value == expected else "?"
            print(f"{status} {desc} ({key}): {value} (expected: {expected})")
        else:
            print(f"  {desc} ({key}): {value}")
        details[key] = value
    
    passed = len(issues) == 0
    results.add_result("配置一致性检查", passed, details)
    
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="CtrlSim 推理诊断工具")
    parser.add_argument("--scenario_index_path", type=str, required=True)
    parser.add_argument("--scenario_data_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vehicle_map_path", type=str, default="data/vehicle_map_valid.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=20, 
                        help="每个测试的步数（需要 > history_steps=10 才能看到模型动作）")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("CtrlSim 推理诊断工具")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  Scenario index: {args.scenario_index_path}")
    print(f"  Scenario data dir: {args.scenario_data_dir}")
    print(f"  Preprocess dir: {args.preprocess_dir}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Device: {args.device}")
    
    # 初始化结果收集器
    results = DiagnosticResults()
    
    # 测试5: 先检查模型 checkpoint（不依赖环境）
    test_5_model_checkpoint(args.checkpoint_path, args.device, results)
    
    # 创建配置
    print("\n创建配置...")
    cfg = create_minimal_config(
        checkpoint_path=args.checkpoint_path,
        scenario_dir=args.scenario_data_dir,
        preprocess_dir=args.preprocess_dir,
    )
    
    # 测试6: 配置一致性检查
    test_6_config_consistency(cfg, results)
    
    # 创建环境
    print("\n创建环境...")
    try:
        env = NocturneCtrlSimAdversarial(
            scenario_index_path=args.scenario_index_path,
            opponent_checkpoint=args.checkpoint_path,
            scenario_data_dir=args.scenario_data_dir,
            preprocess_dir=args.preprocess_dir,
            vehicle_map_path=args.vehicle_map_path,
            opponent_k=7,
            max_episode_steps=90,
            device=args.device,
            seed=args.seed,
            tilting_mode='global',  # 使用 global 模式简化测试
        )
        print("环境创建成功")
    except Exception as e:
        results.add_issue(f"创建环境失败: {str(e)}")
        import traceback
        traceback.print_exc()
        results.print_summary()
        return

    teacher = ExternalTeacher(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    
    # 运行测试
    try:
        # 测试3: 预处理数据（先检查，因为影响后续测试）
        test_3_preprocessed_data(env, results)
        
        # 测试4: 坐标系一致性
        test_4_coordinate_system(env, results)
        
        # 测试1: 模型输入输出
        try:
            test_1_model_input_output(env, teacher, results, num_steps=args.num_steps)
        except Exception as e:
            results.add_issue(f"测试1失败: {str(e)}")
            import traceback
            print("\n测试1异常详情:")
            traceback.print_exc()
        
        # 测试2: GT动作对比
        try:
            test_2_gt_action_comparison(env, teacher, results, num_steps=args.num_steps)
        except Exception as e:
            results.add_issue(f"测试2失败: {str(e)}")
            import traceback
            print("\n测试2异常详情:")
            traceback.print_exc()
        
    except Exception as e:
        results.add_issue(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    # 打印总结
    results.print_summary()
    
    # 返回退出码
    if results.issues:
        print("\n诊断发现问题，请根据上述信息排查。")
        sys.exit(1)
    else:
        print("\n诊断完成，未发现严重问题。")
        sys.exit(0)


if __name__ == "__main__":
    main()
