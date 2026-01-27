#!/usr/bin/env python3
"""
Debug vehicle selection for a specific scenario.
"""
import argparse
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from adapters.ctrl_sim import DataBridge, create_minimal_config


def analyze_scenario(scenario_id: str, scenario_dir: str):
    """Analyze vehicle selection for a specific scenario."""
    print(f"\n{'='*80}")
    print(f"分析场景: {scenario_id}")
    print(f"{'='*80}\n")
    
    cfg = create_minimal_config(
        checkpoint_path="",
        scenario_dir=scenario_dir,
        preprocess_dir=None,
    )
    data_bridge = DataBridge(cfg, preprocess_dir="")
    history_steps = int(getattr(cfg.nocturne, "history_steps", 10))
    max_episode_steps = 90
    
    scenario_filename = f"{scenario_id}.json"
    scenario_path = os.path.join(scenario_dir, scenario_filename)
    
    if not os.path.exists(scenario_path):
        print(f"错误: 场景文件不存在: {scenario_path}")
        return
    
    gt_data_dict = data_bridge.get_ground_truth(scenario_dir, scenario_filename)
    sim = data_bridge.create_simulation(scenario_dir, scenario_filename)
    scenario = sim.getScenario()
    vehicles = list(scenario.vehicles())
    moving_veh_ids = [v.getID() for v in scenario.getObjectsThatMoved()]
    
    print(f"移动车辆总数: {len(moving_veh_ids)}")
    print(f"移动车辆ID列表: {moving_veh_ids}\n")
    
    # 分析每个车辆的位置和轨迹长度
    veh_dict = {v.getID(): v for v in vehicles}
    print(f"{'车辆ID':<10} {'初始位置(x, y)':<25} {'轨迹长度':<12} {'有效轨迹(>=60)'}")
    print('-' * 80)
    
    vehicle_info = []
    for veh_id in moving_veh_ids:
        veh = veh_dict.get(veh_id)
        if veh is None:
            continue
        
        pos = veh.getPosition()
        pos_str = f"({pos.x:.2f}, {pos.y:.2f})"
        
        if veh_id in gt_data_dict:
            gt_traj = np.array(gt_data_dict[veh_id]["traj"])
            existence_mask = gt_traj[:, 4]
            traj_len = existence_mask[history_steps:].sum()
            has_valid_traj = traj_len >= 60
            
            idx_goal = max_episode_steps - 1
            idx_disappear = np.where(existence_mask == 0)[0]
            if len(idx_disappear) > 0:
                idx_goal = idx_disappear[0] - 1
            
            goal_pos = np.array([veh.target_position.x, veh.target_position.y])
            if idx_goal >= 0 and np.linalg.norm(gt_traj[idx_goal, :2] - goal_pos) > 0.0:
                goal_pos = gt_traj[idx_goal, :2]
            
            goal_timestep = idx_goal - history_steps
            
            vehicle_info.append({
                'id': veh_id,
                'pos': np.array([pos.x, pos.y]),
                'goal_pos': goal_pos,
                'goal_timestep': goal_timestep,
                'traj_len': int(traj_len),
                'has_valid_traj': has_valid_traj
            })
            
            print(f"{veh_id:<10} {pos_str:<25} {int(traj_len):<12} {'是' if has_valid_traj else '否'}")
        else:
            print(f"{veh_id:<10} {pos_str:<25} {'N/A':<12} {'N/A'}")
    
    # 分析interesting pairs
    print(f"\n{'='*80}")
    print("分析Interesting Pairs (目标距离 < 10m, 时间步差 < 20, 轨迹长度 >= 60)")
    print(f"{'='*80}\n")
    
    valid_vehicles = [v for v in vehicle_info if v['has_valid_traj']]
    
    if len(valid_vehicles) < 2:
        print("没有足够的有效车辆来形成interesting pair\n")
    else:
        print(f"{'车辆对':<15} {'目标距离(m)':<15} {'时间步差':<12} {'是否满足条件'}")
        print('-' * 80)
        
        interesting_pairs = []
        for i, v1 in enumerate(valid_vehicles):
            for j, v2 in enumerate(valid_vehicles):
                if i >= j:
                    continue
                
                goal_dist = np.linalg.norm(v1['goal_pos'] - v2['goal_pos'])
                timestep_diff = abs(v1['goal_timestep'] - v2['goal_timestep'])
                
                is_interesting = (goal_dist < 10.0 and timestep_diff < 20)
                
                pair_str = f"({v1['id']}, {v2['id']})"
                status = "✓" if is_interesting else "✗"
                
                print(f"{pair_str:<15} {goal_dist:<15.2f} {timestep_diff:<12} {status}")
                
                if is_interesting:
                    interesting_pairs.append((v1['id'], v2['id'], goal_dist, timestep_diff))
        
        if interesting_pairs:
            print(f"\n找到 {len(interesting_pairs)} 个interesting pairs")
            first_pair = interesting_pairs[0]
            selected_ego = min(first_pair[0], first_pair[1])
            print(f"第一个pair: ({first_pair[0]}, {first_pair[1]})")
            print(f"选择较小的ID作为ego: {selected_ego}")
        else:
            print("\n没有找到interesting pair，将使用dense vehicle选择方法")
    
    # 分析Dense Vehicle Selection (平均距离最小)
    print(f"\n{'='*80}")
    print("分析Dense Vehicle Selection (k=7个最近邻居的平均距离)")
    print(f"{'='*80}\n")
    
    positions = {}
    for veh_id in moving_veh_ids:
        veh = veh_dict.get(veh_id)
        if veh is None:
            continue
        
        # 检查轨迹长度约束
        if veh_id not in gt_data_dict:
            continue
        gt_traj = np.array(gt_data_dict[veh_id]["traj"])
        existence_mask = gt_traj[:, 4]
        has_valid_traj = existence_mask[history_steps:].sum() >= 30  # dense使用的阈值是30
        if not has_valid_traj:
            continue
        
        pos = veh.getPosition()
        positions[veh_id] = np.array([pos.x, pos.y], dtype=np.float32)
    
    if len(positions) == 0:
        print("没有满足条件的车辆")
    else:
        print(f"满足条件的车辆数: {len(positions)}")
        print(f"\n{'车辆ID':<10} {'7个最近邻居的平均距离(m)'}")
        print('-' * 50)
        
        avg_distances = []
        for vid, pos in positions.items():
            dists = []
            for other_id, other_pos in positions.items():
                if other_id == vid:
                    continue
                dists.append(np.linalg.norm(pos - other_pos))
            
            if len(dists) > 0:
                dists.sort()
                k = min(7, len(dists))
                avg_dist = float(np.mean(dists[:k]))
                avg_distances.append((vid, avg_dist))
                print(f"{vid:<10} {avg_dist:.2f}")
        
        if avg_distances:
            avg_distances.sort(key=lambda x: (x[1], x[0]))  # 按平均距离排序，相同距离按ID排序
            print(f"\n按平均距离排序:")
            for vid, avg_dist in avg_distances:
                marker = " ← 最小平均距离" if vid == avg_distances[0][0] else ""
                print(f"  车辆 {vid}: {avg_dist:.2f}m{marker}")
            
            if len(interesting_pairs) == 0:
                print(f"\n如果使用dense方法，会选择车辆 {avg_distances[0][0]} 作为ego")
    
    # 显示实际的vehicle_map配置
    print(f"\n{'='*80}")
    print("实际的vehicle_map.json配置")
    print(f"{'='*80}\n")
    
    vehicle_map_path = "/home/chen/workspace/dcd-ctrlsim/data/vehicle_map_valid.json"
    if os.path.exists(vehicle_map_path):
        with open(vehicle_map_path, 'r') as f:
            vehicle_map = json.load(f)
        
        if scenario_id in vehicle_map:
            config = vehicle_map[scenario_id]
            print(f"ego_vehicle_id: {config.get('ego_vehicle_id')}")
            print(f"opponent_vehicle_ids: {config.get('opponent_vehicle_ids')}")
        else:
            print(f"场景 {scenario_id} 不在vehicle_map中")
    else:
        print(f"vehicle_map文件不存在: {vehicle_map_path}")
    
    sim.reset()


def main():
    parser = argparse.ArgumentParser(description="Debug vehicle selection for a scenario")
    parser.add_argument(
        "--scenario_id",
        type=str,
        default="tfrecord-00033-of-00150_279",
        help="Scenario ID to analyze",
    )
    parser.add_argument(
        "--scenario_dir",
        type=str,
        default="/home/chen/workspace/dcd-ctrlsim/data/nocturne_waymo/formatted_json_v2_no_tl_valid",
        help="Scenario directory",
    )
    args = parser.parse_args()
    
    analyze_scenario(args.scenario_id, args.scenario_dir)


if __name__ == "__main__":
    main()
