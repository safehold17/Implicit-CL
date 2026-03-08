#!/usr/bin/env python
"""简单的环境测试脚本。"""

from pathlib import Path
import sys
import traceback


__test__ = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registration import make as gym_make


ENV_KWARGS = {
    "seed": 0,
    "scenario_index_path": "data/scenarios_index_valid.json",
    "opponent_checkpoint": "checkpoints/model.ckpt",
    "scenario_data_dir": "data/nocturne_waymo/formatted_json_v2_no_tl_valid",
    "preprocess_dir": "data/preprocess/test",
    "max_episode_steps": 90,
    "device": "cpu",
}


def main() -> int:
    print("Creating environment...")
    env = None
    try:
        print("[1] About to call gym_make...")
        env = gym_make("Nocturne-CtrlSim-v0", **ENV_KWARGS)
        print("[2] ✓ Environment created successfully")

        print("\n[3] Testing reset...")
        obs = env.reset()
        print(f"[4] ✓ Reset successful, obs type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"    Keys: {list(obs.keys())}")
            for key, val in obs.items():
                shape = val.shape if hasattr(val, "shape") else type(val)
                print(f"    {key}: shape={shape}")

        print("\n[5] Testing step_adversary (building level)...")
        for i in range(4):
            print(f"    [{5 + i}] Calling step_adversary for step {i}...")
            action = 0.0
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
        return 0
    except Exception:
        print("\n❌ Error occurred:")
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
