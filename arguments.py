# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from util import str2bool


parser = argparse.ArgumentParser(description='RL')


# PPO & other optimization arguments. 
parser.add_argument(
    '--algo',
    type=str,
    default='ppo',
    choices=['ppo', 'a2c', 'acktr', 'ucb', 'mixreg'],
    help='Which RL algorithm to use.')
parser.add_argument(
    '--lr', 
    type=float, 
    default=3e-4,
    help='Learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon.')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer alpha.')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    help='Discount factor for rewards.')
parser.add_argument(
    '--use_gae',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use generalized advantage estimator.')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='GAE lambda parameter.')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.03,
    help='Entropy bonus coefficient for student.')
parser.add_argument(
    '--adv_entropy_coef',
    type=float,
    default=0.0,
    help='Entropy bonus coefficient for teacher.')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='Value loss coefficient.')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of student gradients.')
parser.add_argument(
    '--adv_max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of teacher gradients.')
parser.add_argument(
    '--normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student returns.')
parser.add_argument(
    '--adv_normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize teacher returns.')
parser.add_argument(
    '--use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student values via PopArt.')
parser.add_argument(
    '--adv_use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize teacher values using PopArt.')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='Experiment random seed.')
parser.add_argument(
    '--num_processes',
    type=int,
    default=32,
    help='How many training CPU processes to use for experience collection.')
parser.add_argument(
    '--num_steps',
    type=int,
    default=90,
    help='Rollout horizon for A2C-style algorithms.')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=4,
    help='Number of PPO epochs.')
parser.add_argument(
    '--adv_ppo_epoch',
    type=int,
    default=5,
    help='Number of PPO epochs used by teacher.')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=1,
    help='Number of batches for PPO for student.')
parser.add_argument(
    '--adv_num_mini_batch',
    type=int,
    default=1,
    help='Number of batches for PPO for teacher.')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='PPO advantage clipping.')
parser.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=False,
    help='PPO value loss clipping.')
parser.add_argument(
    '--clip_reward',
    type=float,
    default=None,
    help="Amount to clip student rewards. By default no clipping.")
parser.add_argument(
    '--adv_clip_reward',
    type=float,
    default=None,
    help="Amount to clip teacher rewards. By default no clipping.")
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=500000,
    help='Number of environment steps for training.')

# Architecture arguments.
parser.add_argument(
    '--recurrent_arch',
    type=str,
    default='lstm',
    choices=['gru', 'lstm'],
    help='RNN architecture for student and teacher.')
parser.add_argument(
    '--recurrent_agent',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use a RNN architecture for student.')
parser.add_argument(
    '--recurrent_adversary_env',
    type=str2bool, nargs='?', const=True, default=False,
    help='Use a RNN architecture for teacher.')
parser.add_argument(
    '--recurrent_hidden_size',
    type=int,
    default=256,
    help='Recurrent hidden state size.')


# === UED arguments ===
parser.add_argument(
    '--ued_algo',
    type=str,
    default='domain_randomization',
    choices=['domain_randomization', 'minimax', 
             'paired', 'flexible_paired', 
             'alp_gmm'],
    help='UED algorithm')
parser.add_argument(
    '--protagonist_plr',
    type=str2bool, nargs='?', const=True, default=False,
    help="PLR via protagonist's trajectories.")
parser.add_argument(
    '--antagonist_plr',
    type=str2bool, nargs='?', const=True, default=False,
    help="PLR via antagonist's lotrajectoriesss. If protagonist_plr is True, each agent trains using their own.")
parser.add_argument(
    '--use_reset_random_dr',
    type=str2bool, nargs='?', const=True, default=False,
    help='''
         Domain randomization (DR) resets using reset random. 
         If False, DR resets using a uniformly random adversary policy.
         Defaults to False for legacy reasons.''')


# PLR arguments.
parser.add_argument(
    "--use_plr",
    type=str2bool, nargs='?', const=True, default=True,
    help='Whether to use PLR.'
)
parser.add_argument(
    "--level_replay_strategy", 
    type=str,
    default='value_l1',
    choices=['off', 'random', 'uniform', 'sequential',
            'policy_entropy', 'least_confidence', 'min_margin', 
            'gae', 'value_l1', 'signed_value_loss', 'positive_value_loss',
            'grounded_signed_value_loss', 'grounded_positive_value_loss',
            'one_step_td_error', 'alt_advantage_abs',
            'tscl_window'],
    help="PLR score function.")
parser.add_argument(
    "--level_replay_eps", 
    type=float,
    default=0.05,
    help="PLR epsilon for eps-greedy sampling. (Not typically used.)")
parser.add_argument(
    "--level_replay_score_transform",
    type=str, 
    default='rank', 
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax', 'match', 'match_rank'], 
    help="PLR score transform.")
parser.add_argument(
    "--level_replay_temperature", 
    type=float,
    default=0.3,
    help="PLR replay distribution temperature.")
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default='proportionate',
    help="PLR schedule for annealing the replay rate.")
parser.add_argument(
    "--level_replay_rho",
    type=float, 
    default=1.0,
    help="Minimum fill ratio for PLR buffer before sampling replays.")
parser.add_argument(
    "--level_replay_prob", 
    type=float,
    default=0.9,
    help="Probability of sampling a replay level instead of a new level.")
parser.add_argument(
    "--level_replay_alpha",
    type=float, 
    default=1.0,
    help="PLR level score EWA smoothing factor.")
parser.add_argument(
    "--staleness_coef",
    type=float, 
    default=0.3,
    help="Staleness-sampling weighting.")
parser.add_argument(
    "--staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness score transform.")
parser.add_argument(
    "--staleness_temperature",
    type=float, 
    default=0.3,
    help="Staleness distribution temperature.")
parser.add_argument(
    "--train_full_distribution",
    type=str2bool, nargs='?', const=True, default=True,
    help='Train on the full distribution of levels.')
parser.add_argument(
    "--level_replay_seed_buffer_size",
    type=int, 
    default=4000,
    help="Size of PLR level buffer.")
parser.add_argument(
    "--level_replay_seed_buffer_priority",
    type=str, 
    default='replay_support',
    choices=['score', 'replay_support'], 
    help="How to prioritize level buffer members when capacity is reached.")
parser.add_argument(
    "--reject_unsolvable_seeds",
    type=str2bool, nargs='?', const=True, default=False,
    help='Do not add unsolvable seeds to the PLR buffer.')
parser.add_argument(
    "--no_exploratory_grad_updates",
    type=str2bool, nargs='?', const=True, default=False,
    help='Turns on Robust PLR: Only perform gradient updates for episodes on replay levels.'
)

# ACCEL arguments.
parser.add_argument(
    "--use_editor",
    type=str2bool, nargs='?', const=True, default=True,
    help='Turns on ACCEL: Evaluate mutated replay levels for entry in PLR buffer.')
parser.add_argument(
    "--level_editor_prob",
    type=float,
    default=1.0,
    help="Probability of mutating a replayed level under PLR.")
parser.add_argument(
    "--level_editor_method",
    type=str,
    default='random',
    choices=['random'],
    help="Method for mutating levels. ACCEL simply uses random mutations.")
parser.add_argument(
    "--base_levels",
    type=str,
    default='batch',
    choices=['batch', 'easy', 'hard'],
    help="What kind of replayed level under PLR do we edit?")
parser.add_argument(
    "--num_edits",
    type=int,
    default=0.,
    help="Number of edits to make each time a level is mutated.")
parser.add_argument(
    "--use_accel_paired",
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use paired regret estimate for level editor.')
parser.add_argument(
    "--accel_paired_score_function",
    type=str, default="paired", choices=["paired", "flex_paired"],
    help="Type of regret estimate for level editor"
)
parser.add_argument(
    "--use_lstm",
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use LSTM architecture for BipedalWalker models.')

# BC arguments
parser.add_argument(
    '--use_behavioural_cloning',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use behavioural cloning')
parser.add_argument(
    '--kl_update_step',
    type=float,
    default=1,
    help='Number of steps after which KL loss should be used')
parser.add_argument(
    '--kl_loss_coef',
    type=float,
    default=0.0,
    help='KL divergence loss coefficient for behavioural cloning (default: 0.1)')
parser.add_argument(
    '--use_kl_only_agent',
    type=str2bool, nargs='?', const=True, default=False,
    help='Use behavioural cloning loss in agent only. Default behaviour is bc in both')

# Fine-tuning arguments.
parser.add_argument(
    '--xpid_finetune',
    default=None,
    help='Checkpoint directory containing model for fine-tuning.')
parser.add_argument(
    '--model_finetune',
    default='model',
    help='Name of .tar to load for fine-tuning.')

# Hardware arguments.
parser.add_argument(
    '--no_cuda',
    type=str2bool, nargs='?', const=True, default=False,
    help='Disables CUDA training.')

# Logging arguments.
parser.add_argument(
    '--xpid',
    default='latest',
    help='Name for the training run. Used for the name of the output results directory.')
parser.add_argument(
    '--log_dir',
    default='~/logs/dcd/',
    help='Directory in which to save experimental outputs.')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='Log training stats every this many updates.')
parser.add_argument(
    "--checkpoint_interval", 
    type=int, 
    default=100,
    help="Save model every this many updates.")
parser.add_argument(
    "--archive_interval", 
    type=int, 
    default=0,
    help="Save an archived checkpoint every this many updates.")
parser.add_argument(
    "--checkpoint_basis",
    type=str,
    default="num_updates",
    choices=["num_updates", "student_grad_updates"],
    help=f'''Archive interval basis. 
             num_updates: By # update cycles (full rollout cycle across all agents); 
             student_grad_updates: By # grad updates performed by the student agent.''')
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=1,
    help="Save level weights every this many updates. *Only for PLR with a fixed level buffer.*")
parser.add_argument(
    "--screenshot_interval", 
    type=int, 
    default=5000,
    help="Save screenshot of the training environment every this many updates.")
parser.add_argument(
    "--screenshot_batch_size", 
    type=int, 
    default=1,
    help="Number of training environments to screenshot each screenshot_interval.")
parser.add_argument(
    '--render',
    type=str2bool, nargs='?', const=True, default=False,
    help='Render to environment to screen.')
parser.add_argument(
    "--checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Begin training from checkpoint. Needed for preemptible training on clusters.")
parser.add_argument(
    "--disable_checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Disable checkpointing.")
parser.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log the gradient norm of the actor-critic.")
parser.add_argument(
    '--log_action_complexity',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log action-trajectory complexity metrics throughout training.")
parser.add_argument(
    '--log_replay_complexity',
    type=str2bool, nargs='?', const=True, default=True,
    help="Log complexity metrics of replay levels.")
parser.add_argument(
    '--log_plr_buffer_stats',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log PLR buffer stats.")
parser.add_argument(
    "--verbose", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Whether to print logs to stdout.")

# Evaluation arguments.
parser.add_argument(
    '--test_interval',
    type=int,
    default=250,
    help='Evaluate on test environments every this many updates.')
parser.add_argument(
    '--test_num_episodes',
    type=int,
    default=1,
    help='Number of test episodes per environment.')
parser.add_argument(
    '--eval_record_video',
    type=str2bool, nargs='?', const=True, default=False,
    help='Record rollout videos during evaluation.')
parser.add_argument(
    '--eval_video_dir',
    type=str,
    default=None,
    help='Override directory for evaluation videos. Default uses log_dir/xpid/videos.')
parser.add_argument(
    '--eval_screenshot',
    type=str2bool, nargs='?', const=True, default=True,
    help='Save a screenshot at the start of each evaluation episode.')
parser.add_argument(
    '--test_num_processes',
    type=int,
    default=2,
    help='Number of test processes per environment.')
parser.add_argument(
    '--test_env_names',
    type=str,
    default='Nocturne-CtrlSim-v0',
    help='CSV string of test environments for evaluation during training.')

# Environment arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='Nocturne-CtrlSim-v0',
    help='Environment to train on.')
parser.add_argument(
    '--handle_timelimits',
    type=str2bool, nargs='?', const=True, default=False,
    help="Bootstrap off of early termination states. Requires env to be wrapped by envs.wrappers.TimeLimit.")
parser.add_argument(
    '--singleton_env',
    type=str2bool, nargs='?', const=True, default=False,
    help="When using a fixed env, whether the same environment should also be reused across workers.")
parser.add_argument(
    '--use_global_critic',
    type=str2bool, nargs='?', const=True, default=False,
    help="Student's critic is fully observable. *Only for MultiGrid.*")
parser.add_argument(
    '--use_global_policy',
    type=str2bool, nargs='?', const=True, default=False,
    help="Student's policy is fully observable. *Only for MultiGrid.*")

# CarRacing-specific arguments.
parser.add_argument(
    '--grayscale',
    type=str2bool, nargs='?', const=True, default=False,
    help="Convert observations to grayscale for CarRacing.")
parser.add_argument(
    '--crop_frame',
    type=str2bool, nargs='?', const=True, default=False,
    help="Convert observations to grayscale for CarRacing.")
parser.add_argument(
    '--reward_shaping',
    type=str2bool, nargs='?', const=True, default=False,
    help="Use custom shaped rewards for CarRacing.")
parser.add_argument(
    '--num_action_repeat',
    type=int, default=1,
    help="Repeat actions this many times for CarRacing.")
parser.add_argument(
    '--frame_stack',
    type=int, default=1,
    help="Number of observation frames to stack for CarRacing.")
parser.add_argument(
    '--num_control_points',
    type=int, default=12,
    help="Number of bezier control points for CarRacing-Bezier environments.")
parser.add_argument(
    '--min_rad_ratio',
    type=float, default=0.333333333,
    help="Default minimum radius ratio for CarRacing-Classic (polar coordinates).")
parser.add_argument(
    '--max_rad_ratio',
    type=float, default=1.0,
    help="Default minimum radius ratio for CarRacing-Classic (polar coordinates).")
parser.add_argument(
    '--use_skip',
    type=str2bool, nargs='?', const=True, default=False,
    help="CarRacing teacher can use a skip action.")
parser.add_argument(
    '--choose_start_pos',
    type=str2bool, nargs='?', const=True, default=False,
    help="CarRacing teacher also chooses the start position.")
parser.add_argument(
    '--use_sketch',
    type=str2bool, nargs='?', const=True, default=True,
    help="CarRacing teacher designs tracks on a downsampled grid.")
parser.add_argument(
    '--use_categorical_adv',
    type=str2bool, nargs='?', const=True, default=False,
    help="CarRacing teacher uses a categorical policy.")
parser.add_argument(
    '--sparse_rewards',
    type=str2bool, nargs='?', const=True, default=False,
    help="Use sparse rewards + goal placement for CarRacing.")
parser.add_argument(
    '--num_goal_bins',
    type=int, default=1,
    help="Number of goal bins when using sparse rewards for CarRacing.")

# ============== Nocturne Student policy parameters ==============
parser.add_argument(
    '--student_input_dim',
    type=int,
    default=64,
    help='The embedding dimension of each modality in Student Late Fusion')
parser.add_argument(
    '--student_hidden_dim',
    type=int,
    default=128,
    help='The hidden dimension of the fused Student Late Fusion')
parser.add_argument(
    '--student_num_neighbors',
    type=int,
    default=16,
    help='The number of nearest neighbors in the observation (number of nearest cars)')
parser.add_argument(
    '--student_top_k_road',
    type=int,
    default=64,
    help='The number of nearest road points in the observation (number of nearest road points)')
parser.add_argument(
    '--student_accel_discretization',
    type=int,
    default=7,
    help='Number of acceleration bins for student discrete action space.')
parser.add_argument(
    '--student_steer_discretization',
    type=int,
    default=13,
    help='Number of steering bins for student discrete action space.')
parser.add_argument(
    '--student_dropout',
    type=float,
    default=0.0,
    help='The dropout probability of the Student network')
parser.add_argument(
    '--student_act_func',
    type=str,
    default='tanh',
    choices=['tanh', 'gelu', 'relu'],
    help='The activation function of the Student network')

# ============== Nocturne-Ctrlsim environment configuration parameter ==============
parser.add_argument(
    '--tilting_mode',
    type=str,
    default='per_vehicle',
    choices=['global', 'per_vehicle', 'ego', 'none'],
    help=(
        'Tilting mode: '
        'global (all opponents share same tilts), '
        'per_vehicle (each opponent has independent tilts), '
        'ego (tilts stored for ego only; opponents use 0), '
        'none (all tilts are 0, scenario-only adversary).'
    ))
parser.add_argument(
    '--mutation_mode',
    type=str,
    default='all',
    choices=['one', 'all'],
    help='Mutation mode for tilting parameters: one or all.')
parser.add_argument(
    '--mutation_range',
    type=float,
    default=5.0,
    help='Mutation range for tilting parameters (delta sampled from [-range, range]).')
parser.add_argument(
    '--tilt_range',
    type=int,
    nargs=2,
    default=[-25, 25],
    metavar=('MIN', 'MAX'),
    help='Absolute tilt range for parameters, formatted as: MIN MAX (e.g., -25 25).')
parser.add_argument(
    '--scenario_index_path',
    type=str,
    default='data/scenarios_index.json',
    help='The path to the Nocturne scenario index JSON file')
parser.add_argument(
    '--opponent_checkpoint',
    type=str,
    default='checkpoints/model.ckpt',
    help='The path to the ctrl-sim opponent model checkpoint')
parser.add_argument(
    '--opponent_vehicle_number',
    type=int,
    default=7,
    help='Number of opponent vehicles used for per-vehicle tilting (K).')
parser.add_argument(
    '--scenario_data_dir',
    type=str,
    default='data/nocturne_waymo/formatted_json_v2_no_tl_train',
    help='The directory of the Nocturne scenario data')
parser.add_argument(
    '--preprocess_dir',
    type=str,
    default='data/preprocess/test',
    help='The directory of the ctrl-sim preprocessed data')
parser.add_argument(
    '--preproc_cache_size',
    type=int,
    default=64,
    help='LRU cache size for preprocessed scenario data in each Nocturne worker.')
parser.add_argument(
    '--vehicle_map_path',
    type=str,
    default='data/vehicle_map_valid.json',
    help='The path to the vehicle map JSON file')
parser.add_argument(
    '--nocturne_max_episode_steps',
    type=int,
    default=90,
    help='The maximum number of steps in the Nocturne environment (default: 90).')
parser.add_argument(
    '--done_on_position_reached_only',
    type=str2bool, nargs='?', const=True, default=True,
    help='Whether Nocturne check_done should use position_reached as the success condition.')
parser.add_argument(
    '--remove_background_vehicles',
    type=str2bool, nargs='?', const=True, default=True,
    help='Remove moving vehicles other than ego/opponent in Nocturne env.')

# ============== reward coefficient ==============
parser.add_argument(
    '--veh_veh_collision_rew_multiplier',
    type=float,
    default=10.0,
    help='Collision penalty multiplier for vehicle-vehicle collisions (ctrl-sim default).')
parser.add_argument(
    '--veh_edge_collision_rew_multiplier',
    type=float,
    default=10.0,
    help='Collision penalty multiplier for vehicle-road-edge collisions (ctrl-sim default).')
parser.add_argument(
    '--pos_target_achieved_rew_multiplier',
    type=float,
    default=10.0,
    help='Goal position achieved reward multiplier for student reward (ctrl-sim default).')
parser.add_argument(
    '--shaped_goal_distance_scaling',
    type=float,
    default=0.5,
    help='Scaling factor for shaped goal/speed/heading rewards in student reward.')
parser.add_argument(
    '--approaching_goal_scaling',
    type=float,
    default=1.0,
    help='Scaling factor for approaching-goal reward in student reward.')
parser.add_argument(
    '--use_approaching_goal',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include approaching-goal reward in student reward aggregation.')
parser.add_argument(
    '--use_persistent_position_reward',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use persistent position reached reward (CtrlSim-style lock); if False, use one-time position reward.')
parser.add_argument(
    '--shaped_goal_reward',
    type=str2bool, nargs='?', const=True, default=True,
    help='Whether to enable shaped goal reward components in student reward.')
parser.add_argument(
    '--use_heading_shaped',
    type=str2bool, nargs='?', const=True, default=True,
    help='Whether to include shaped heading reward in student reward aggregation.')
parser.add_argument(
    '--use_pos_shaped',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include shaped position reward in student reward aggregation.')
parser.add_argument(
    '--use_speed_shaped',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include shaped speed reward in student reward aggregation.')
parser.add_argument(
    '--use_speed_heading_target',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include binary speed/heading target achieved rewards in student reward aggregation.')
parser.add_argument(
    '--use_veh_veh_shaped',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include veh-veh shaped reward in student reward aggregation.')
parser.add_argument(
    '--use_veh_edge_shaped',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to include veh-edge shaped reward in student reward aggregation.')
parser.add_argument(
    '--max_veh_veh_distance',
    type=float,
    default=15.0,
    help='Maximum distance used to normalize veh-veh shaped reward (ctrl-sim default).')
parser.add_argument(
    '--veh_edge_reward_distance_clip',
    type=float,
    default=5.0,
    help='Distance clip used to normalize veh-edge shaped reward (ctrl-sim default).')

# ============== warm-up stage ==============
parser.add_argument(
    '--warmup_opponent_disable_ratio',
    type=float,
    default=0,
    help='Warmup ratio for opponent-disable stage (fraction of num_env_steps).')
parser.add_argument(
    '--warmup_opponent_replay_ratio',
    type=float,
    default=0.2,
    help='Warmup ratio for opponent-replay stage (fraction of num_env_steps).')
parser.add_argument(
    '--warm_up_level_replay_seed_buffer_size',
    type=int,
    default=4000,
    help='PLR level buffer size used during warm-up stages (disable/replay).')

# ============== render options ==============
parser.add_argument(
    '--show_tilting_params',
    type=str2bool, nargs='?', const=True, default=True,
    help='Show tilting parameter text for opponent vehicles in Nocturne render.')
parser.add_argument(
    '--show_vehicle_ids',
    type=str2bool, nargs='?', const=True, default=True,
    help='Show vehicle id text for ego/opponent vehicles in level image.')
parser.add_argument(
    '--show_ego_vehicle_selection',
    type=str2bool, nargs='?', const=True, default=True,
    help='Show ego vehicle selection mode text in level image.')
