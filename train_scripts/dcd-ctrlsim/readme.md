# Training Parameters

## Steps

- num process = 32
- PPO rollout length = 512
- batch size = 16384
- num mini batch = 32
- mini batch size = 512
- update = 2000
- num env steps = 32768000
- warm up ratio = 0.1  ->  first 200 updates

## PPO parameters

based on these two papers

[Human-compatible driving partners through data-regularized self-play reinforcement learning](https://arxiv.org/html/2403.19648v2)

[Building reliable sim driving agents by scaling self-play](https://arxiv.org/html/2502.14706v3)


## PLR

based on dcd car-racing parameters

- total level number = 32 * (2000 - 200) = 57600
- PLR buffer size = 57600 / 5 = 11520

## Reward setup

### RTG in ctrl-sim

- goal_pos : consistent goal reaching reward until rollout ends
- veh_veh : veh_veh_shaped - veh_veh collision_penalty
- veh_edge : veh_edge_shaped - veh_edge collision_penalty

### Reward for student vehicle

Could be useful for training:
- position_reward_term : consistent goal reaching reward until rollout ends
- approaching goal : reward ego vehicle if it gets closer to the goal point
- heading_shaped term : reward ego vehicle for correct heading
- pos / speed / heading shaped reward : not good for the early training stage, also not using in ctrlsim
- speed / heading target reward : one-time reward if speed / heading of ego vehicle match the ground truth when reaching the goal point

To align with RTG in ctrl-sim for regret enhancement
- use_persistent_position_reward = True
- use_approaching_goal = False
- use_speed_heading_target = False
- use_veh_veh_shaped = True
- use_veh_edge_shaped = True
