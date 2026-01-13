#!/usr/bin/env python
"""简单的环境测试脚本"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.registration import make as gym_make

# 环境配置
env_kwargs = {
    'seed': 0,
    'scenario_index_path': 'data/scenarios_index_valid.json',
    'opponent_checkpoint': 'checkpoints/model.ckpt',
    'scenario_data_dir': 'data/nocturne_waymo/formatted_json_v2_no_tl_valid',
    'preprocess_dir': 'data/preprocess/test',
    'max_episode_steps': 90,
    'device': 'cpu',
}

print("Creating environment...")
try:
    print("[1] About to call gym_make...")
    env = gym_make('Nocturne-CtrlSim-Adversarial-v0', **env_kwargs)
    print("[2] ✓ Environment created successfully")
    
    print("\n[3] Testing reset...")
    obs = env.reset()
    print(f"[4] ✓ Reset successful, obs type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"    Keys: {list(obs.keys())}")
        for key, val in obs.items():
            print(f"    {key}: shape={val.shape if hasattr(val, 'shape') else type(val)}")
    
    print("\n[5] Testing step_adversary (building level)...")
    for i in range(4):
        print(f"    [{5+i}] Calling step_adversary for step {i}...")
        action = 0.0  # 简单的零动作
        obs, reward, done, info = env.step_adversary(action)
        print(f"    ✓ Step {i}: done={done}")
    
    print("\n[9] Testing reset_agent...")
    obs = env.reset_agent()
    print(f"[10] ✓ Agent reset successful, obs shape: {obs.shape}")
    
    print("\n[11] Testing step...")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"[12] ✓ Step successful: reward={reward:.3f}, done={done}")
    
    print("\n✅ All tests passed!")
    env.close()
    
except Exception as e:
    import traceback
    print(f"\n❌ Error occurred:")
    traceback.print_exc()
    sys.exit(1)
