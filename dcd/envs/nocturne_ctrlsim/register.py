"""
Register Nocturne + CtRL-Sim environment to DCD framework

"""
import gym
from dcd.envs.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95, max_episode_steps=None):
    """Register environment to DCD framework"""
    assert env_id.startswith("Nocturne-")
    if env_id in env_list:
        del gym.envs.registry.env_specs[env_id]
    else:
        env_list.append(env_id)

    kwargs = dict(
        id=env_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold
    )

    if max_episode_steps:
        kwargs.update({'max_episode_steps': max_episode_steps})

    gym_register(**kwargs)


# Register default environment configuration
register(
    env_id="Nocturne-CtrlSim-Adversarial-v0",
    entry_point="envs.nocturne_ctrlsim.adversarial:NocturneCtrlSimAdversarial",
    max_episode_steps=90,  # Same as ctrl-sim default steps
)
