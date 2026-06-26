# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from algos import PPO, RolloutStorage, ACAgent
from dcd_models import \
    MultigridNetwork, MultigridGlobalCriticNetwork, \
    CarRacingNetwork, \
    CarRacingBezierAdversaryEnvNetwork, \
    BipedalWalkerStudentPolicy, \
    BipedalWalkerAdversaryPolicy, \
    BipedalWalkerRecurrentStudentPolicy, \
    BipedalWalkerRecurrentAdversaryPolicy, \
    StudentPolicy

def model_for_multigrid_agent(
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256,
    use_global_critic=False,
    use_global_policy=False):
    if agent_type == 'adversary_env':
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
        adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]

        model = MultigridNetwork(
            observation_space=adversary_observation_space, 
            action_space=adversary_action_space,
            conv_filters=128,
            scalar_fc=10,
            scalar_dim=adversary_max_timestep,
            random_z_dim=adversary_random_z_dim,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_directions = observation_space['direction'].high[0] + 1 
        model_kwargs = dict(
            observation_space=observation_space, 
            action_space=action_space,
            scalar_fc=5,
            scalar_dim=num_directions,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size)

        model_constructor = MultigridNetwork
        if use_global_critic:
            model_constructor = MultigridGlobalCriticNetwork

        if use_global_policy:
            model_kwargs.update({'use_global_policy': True})

        model = model_constructor(**model_kwargs)

    return model

def model_for_car_racing_agent(
    env,
    agent_type='agent',
    use_skip=False,
    choose_start_pos=False,
    use_popart=False,
    adv_use_popart=False,
    use_categorical_adv=False,
    use_goal=False,
    num_goal_bins=1):
    if agent_type == 'adversary_env':
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space
        model = CarRacingBezierAdversaryEnvNetwork(
            observation_space=adversary_observation_space,
            action_space=adversary_action_space,
            use_categorical=use_categorical_adv,
            use_skip=use_skip,
            choose_start_pos=choose_start_pos,
            use_popart=adv_use_popart,
            use_goal=use_goal,
            num_goal_bins=num_goal_bins)
    else:
        action_space = env.action_space
        obs_shape = env.observation_space.shape
        model = CarRacingNetwork(
            obs_shape=obs_shape,
            action_space = action_space,
            hidden_size=100,
            use_popart=use_popart) 

    return model

def model_for_bipedalwalker_agent(
    env,
    agent_type='agent',
    recurrent_arch=False,
    use_lstm=False):
    if 'adversary_env' in agent_type:
        adversary_observation_space = env.adversary_observation_space
        adversary_action_space = env.adversary_action_space

        if use_lstm:
            model = BipedalWalkerRecurrentAdversaryPolicy(
                observation_space=adversary_observation_space,
                action_space=adversary_action_space,
                recurrent_hidden_size=256
            )
        else:
            model = BipedalWalkerAdversaryPolicy(
                    observation_space=adversary_observation_space,
                    action_space=adversary_action_space)

    else:
        if use_lstm:
            model = BipedalWalkerRecurrentStudentPolicy(
                obs_shape=env.observation_space.shape,
                action_space=env.action_space,
                recurrent=recurrent_arch,
                recurrent_hidden_size=256
            )
        else:
            model = BipedalWalkerStudentPolicy(
                obs_shape=env.observation_space.shape,
                action_space=env.action_space,
                recurrent=recurrent_arch)

    return model

def model_for_nocturne_agent(
    env,
    agent_type='agent',
    input_dim=64,
    hidden_dim=128,
    num_neighbors=16,
    top_k_road_points=64,
    dropout=0.0,
    act_func="tanh",
    student_partner_pooling="attention",
    student_road_pooling="attention",
    recurrent=False,
    recurrent_arch=None,
    recurrent_hidden_size=256,
    random_teacher=False,
    # CtRL-Sim student hyperparams (used when student_model_type='ctrlsim')
    ctrlsim_hidden_dim=256,
    ctrlsim_num_heads=8,
    ctrlsim_encoder_layers=2,
    ctrlsim_decoder_layers=4,
):
    """    
    Args:
        env: Environment instance (observation_space and action_space are required)
        agent_type: 'agent' (student) or 'adversary_env' (teacher)
        input_dim: Embedding dimension for each modality (Late Fusion intermediate layer)
        hidden_dim: Hidden layer dimension after fusion
        num_neighbors: Number of neighboring vehicles to consider (not K)
        top_k_road_points: Number of road points to consider (R)
        dropout: Dropout probability
        act_func: Activation function ("tanh" or "gelu")
        student_partner_pooling: Pooling mode for partner tokens
        student_road_pooling: Pooling mode for road tokens
        recurrent: Whether to use recurrent network
        recurrent_arch: RNN architecture type ('lstm' or 'gru')
        recurrent_hidden_size: Hidden size for recurrent policies
        random_teacher: Whether to use random Teacher (baseline comparison)
    
    Returns:
        model: StudentPolicy or NocturneAdversaryPolicy instance
    """
    from dcd_models import NocturneAdversaryPolicy

    # Teacher policy, level generation
    if 'adversary_env' in agent_type:
        obs_space = env.adversary_observation_space
        action_space = env.adversary_action_space

        model = NocturneAdversaryPolicy(
            observation_space=obs_space,
            action_space=action_space,
            random=random_teacher,
            recurrent=recurrent,
            recurrent_arch=recurrent_arch,
            base_kwargs={'hidden_size': hidden_dim}
        )
        return model

    action_space = env.action_space

    # CtRL-Sim aligned student policy
    if getattr(env, 'student_model_type', 'late_fusion') == 'ctrlsim':
        from dcd_models.student_models import CtrlSimStudentPolicy
        obs_cfg = env.ctrlsim_obs_cfg
        action_dim = int(action_space.n)
        model = CtrlSimStudentPolicy(
            obs_cfg=obs_cfg,
            action_dim=action_dim,
            hidden_dim=ctrlsim_hidden_dim,
            num_heads=ctrlsim_num_heads,
            num_encoder_layers=ctrlsim_encoder_layers,
            num_decoder_layers=ctrlsim_decoder_layers,
            dropout=dropout,
        )
        return model

    # Legacy LateFusion student policy
    obs_shape = env.observation_space.shape
    model = StudentPolicy(
        obs_shape=obs_shape,
        action_space=action_space,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_controlled_agents=num_neighbors + 1,  # +1 for ego
        top_k_road_points=top_k_road_points,
        dropout=dropout,
        act_func=act_func,
        student_partner_pooling=student_partner_pooling,
        student_road_pooling=student_road_pooling,
        recurrent=recurrent,
        recurrent_arch=recurrent_arch if recurrent else None,
        recurrent_hidden_size=recurrent_hidden_size,
    )

    return model

def model_for_env_agent(
    env_name,
    env,
    agent_type='agent',
    recurrent_arch=None,
    recurrent_hidden_size=256,
    use_global_critic=False,
    use_global_policy=False,
    use_skip=False,
    choose_start_pos=False,
    use_popart=False,
    adv_use_popart=False,
    use_categorical_adv=False,
    use_goal=False,
    num_goal_bins=1,
    use_lstm=False,
    # Nocturne Student parameters
    student_input_dim=64,
    student_hidden_dim=128,
    student_num_neighbors=16,
    student_top_k_road=64,
    student_dropout=0.0,
    student_act_func="tanh",
    student_partner_pooling="attention",
    student_road_pooling="attention",
    # Nocturne CtRL-Sim student hyperparams
    ctrlsim_hidden_dim=256,
    ctrlsim_num_heads=8,
    ctrlsim_encoder_layers=2,
    ctrlsim_decoder_layers=4,
    # Nocturne Teacher parameters
    random_teacher=False):
    assert agent_type in ['agent', 'adversary_agent', 'adversary_env']
        
    if env_name.startswith('MultiGrid'):
        model = model_for_multigrid_agent(
            env=env, 
            agent_type=agent_type,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size,
            use_global_critic=use_global_critic,
            use_global_policy=use_global_policy)
    elif env_name.startswith('CarRacing'):
        model = model_for_car_racing_agent(
            env=env, 
            agent_type=agent_type,
            use_skip=use_skip,
            choose_start_pos=choose_start_pos,
            use_popart=use_popart,
            adv_use_popart=adv_use_popart,
            use_categorical_adv=use_categorical_adv,
            use_goal=use_goal,
            num_goal_bins=num_goal_bins)
    elif env_name.startswith('BipedalWalker'):
        model = model_for_bipedalwalker_agent(
            env=env,
            agent_type=agent_type,
            recurrent_arch=recurrent_arch,
            use_lstm=use_lstm)
    elif env_name.startswith('Nocturne') or env_name.startswith('nocturne'):
        model = model_for_nocturne_agent(
            env=env,
            agent_type=agent_type,
            input_dim=student_input_dim,
            hidden_dim=student_hidden_dim,
            num_neighbors=student_num_neighbors,
            top_k_road_points=student_top_k_road,
            dropout=student_dropout,
            act_func=student_act_func,
            student_partner_pooling=student_partner_pooling,
            student_road_pooling=student_road_pooling,
            recurrent=recurrent_arch is not None,
            recurrent_arch=recurrent_arch,
            recurrent_hidden_size=recurrent_hidden_size,
            random_teacher=random_teacher,
            ctrlsim_hidden_dim=ctrlsim_hidden_dim,
            ctrlsim_num_heads=ctrlsim_num_heads,
            ctrlsim_encoder_layers=ctrlsim_encoder_layers,
            ctrlsim_decoder_layers=ctrlsim_decoder_layers,
        )
    else:
        raise ValueError(f'Unsupported environment {env_name}.')

    return model


def _arg_get(args, key, default):
    """Return an argument value from Namespace-like or mapping-like inputs."""
    if hasattr(args, 'get'):
        return args.get(key, default)
    return getattr(args, key, default)


def make_agent(name, env, args, device='cpu'):
    # Create model instance
    is_adversary_env = 'env' in name

    if (
        not is_adversary_env
        and (args.env_name.startswith('Nocturne') or args.env_name.startswith('nocturne'))
    ):
        student_model_type = _arg_get(args, 'student_model_type', 'late_fusion')
        env.student_model_type = student_model_type
        if student_model_type == 'ctrlsim':
            from envs.nocturne_ctrlsim.student.ctrlsim_observation import (
                CtrlSimObsConfig,
            )

            env.ctrlsim_obs_cfg = CtrlSimObsConfig(
                num_agents=int(
                    _arg_get(args, 'ctrlsim_student_num_neighbors', 8)
                ) + 1,
                seq_len=int(_arg_get(args, 'ctrlsim_student_seq_len', 10)),
            )

    if is_adversary_env:
        observation_space = env.adversary_observation_space
        action_space = env.adversary_action_space
        num_steps = observation_space['time_step'].high[0]
        recurrent_arch = (
            args.recurrent_arch if args.recurrent_adversary_env else None
        )
        entropy_coef = args.adv_entropy_coef
        ppo_epoch = args.adv_ppo_epoch
        num_mini_batch = args.adv_num_mini_batch
        max_grad_norm = args.adv_max_grad_norm
        use_popart = _arg_get(args, 'adv_use_popart', False)
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_steps = args.num_steps
        recurrent_arch = args.recurrent_arch if args.recurrent_agent else None
        entropy_coef = args.entropy_coef
        ppo_epoch = args.ppo_epoch
        num_mini_batch = args.num_mini_batch
        max_grad_norm = args.max_grad_norm
        use_popart = _arg_get(args, 'use_popart', False)

    recurrent_hidden_size = args.recurrent_hidden_size

    actor_critic = model_for_env_agent(
        args.env_name, env, name, 
        recurrent_arch=recurrent_arch,
        recurrent_hidden_size=recurrent_hidden_size,
        use_global_critic=args.use_global_critic,
        use_global_policy=_arg_get(args, 'use_global_policy', False),
        use_skip=_arg_get(args, 'use_skip', False),
        choose_start_pos=_arg_get(args, 'choose_start_pos', False),
        use_popart=_arg_get(args, 'use_popart', False),
        adv_use_popart=_arg_get(args, 'adv_use_popart', False),
        use_categorical_adv=_arg_get(args, 'use_categorical_adv', False),
        use_goal=_arg_get(args, 'sparse_rewards', False),
        num_goal_bins=_arg_get(args, 'num_goal_bins', 1),
        use_lstm=_arg_get(args, 'use_lstm', False),
        # Nocturne Student parameters
        student_input_dim=_arg_get(args, 'student_input_dim', 64),
        student_hidden_dim=_arg_get(args, 'student_hidden_dim', 128),
        student_num_neighbors=_arg_get(args, 'student_num_neighbors', 16),
        student_top_k_road=_arg_get(args, 'student_top_k_road', 64),
        student_dropout=_arg_get(args, 'student_dropout', 0.0),
        student_act_func=_arg_get(args, 'student_act_func', 'tanh'),
        student_partner_pooling=_arg_get(
            args, 'student_partner_pooling', 'attention'
        ),
        student_road_pooling=_arg_get(args, 'student_road_pooling', 'attention'),
        # Nocturne CtRL-Sim student hyperparams
        ctrlsim_hidden_dim=_arg_get(args, 'ctrlsim_student_hidden_dim', 256),
        ctrlsim_num_heads=_arg_get(args, 'ctrlsim_student_num_heads', 8),
        ctrlsim_encoder_layers=_arg_get(args, 'ctrlsim_student_encoder_layers', 2),
        ctrlsim_decoder_layers=_arg_get(args, 'ctrlsim_student_decoder_layers', 4),
        # Nocturne Teacher parameters
        random_teacher=_arg_get(args, 'random_teacher', False))

    actor_critic = actor_critic.to(device)

    # Compile the model for faster forward passes. dynamic=True avoids recompilation
    # on variable batch sizes. Falls back to eager silently on unsupported patterns.
    actor_critic = torch.compile(actor_critic, dynamic=True)
    print(f"[make_agent] Compiled actor_critic '{name}' with torch.compile (dynamic=True).")

    algo = None
    storage = None
    agent = None

    # Adversary action dimension consistency check (Nocturne joint action)
    if (
        is_adversary_env
        and (args.env_name.startswith('Nocturne') or args.env_name.startswith('nocturne'))
        and action_space.__class__.__name__ == 'Box'
    ):
        if not hasattr(actor_critic, 'action_dim'):
            raise ValueError(
                "Adversary model must expose `action_dim` for Nocturne env."
            )

        expected_action_dim = int(action_space.shape[0])
        model_action_dim = int(actor_critic.action_dim)
        if model_action_dim != expected_action_dim:
            raise ValueError(
                f"Adversary action dim mismatch: model={model_action_dim}, "
                f"env_space={expected_action_dim}. "
                "Please check env tilting_mode, adversary_action_space, and checkpoint compatibility."
            )


    use_proper_time_limits = \
        hasattr(env, 'get_max_episode_steps') \
        and env.get_max_episode_steps() is not None \
        and _arg_get(args, 'handle_timelimits', False)

    if args.algo == 'ppo':
        # Create PPO
        algo = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=entropy_coef,
            kl_loss_coef=args.kl_loss_coef,
            use_ego_ctrlsim_kl_loss=args.use_ego_ctrlsim_kl_loss,
            ego_ctrlsim_kl_schedule=args.ego_ctrlsim_kl_schedule,
            ego_ctrlsim_kl_init_coef=args.ego_ctrlsim_kl_init_coef,
            ego_ctrlsim_kl_min_coef=args.ego_ctrlsim_kl_min_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=max_grad_norm,
            clip_value_loss=args.clip_value_loss,
            log_grad_norm=args.log_grad_norm
        )

        # Create storage
        storage = RolloutStorage(
            model=actor_critic,
            num_steps=num_steps,
            num_processes=args.num_processes,
            observation_space=observation_space,
            action_space=action_space,
            recurrent_hidden_state_size=args.recurrent_hidden_size,
            recurrent_arch=args.recurrent_arch,
            use_proper_time_limits=use_proper_time_limits,
            use_popart=use_popart,
            use_ego_ctrlsim_action_logits=args.use_ego_ctrlsim_kl_loss,
        )

        agent = ACAgent(algo=algo, storage=storage).to(device)

    else:
        raise ValueError(f'Unsupported RL algorithm {algo}.')

    return agent
