import logging
import os
import sys
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.logger import HumanOutputFormat
from tqdm import tqdm

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from train_bootstrap import resolve_runtime_args, setup_virtual_display


def _remap_clearml_dataset_paths(args, dataset_dir):
    """Resolve dataset-backed resource paths on a ClearML worker."""
    default_paths = {
        'scenario_index_path': 'preparation_file/scenarios_index_filtered_train.json',
        'scenario_data_dir': 'scenario_data/formatted_json_v2_no_tl_train',
        'preprocess_dir': 'compressed_preprocessed_data/train',
        'opponent_checkpoint': 'checkpoints/model_fp16.ckpt',
        'vehicle_map_path': 'preparation_file/vehicle_map_filtered_train.json',
    }
    for key, default_relative_path in default_paths.items():
        path = getattr(args, key, None)
        if not path:
            continue
        if os.path.isabs(path):
            if os.path.exists(path):
                continue
            setattr(args, key, os.path.join(dataset_dir, default_relative_path))
            continue
        setattr(args, key, os.path.join(dataset_dir, path))

def init_clearml(args):
    """Initialize ClearML experiment tracking and resolve dataset paths.

    Returns the ClearML Task object, or None when ClearML is disabled.
    """
    is_clearml_worker = bool(os.environ.get("CLEARML_TASK_ID"))
    if args.use_clearml and args.clearml_monitor_only:
        raise ValueError(
            "--use_clearml and --clearml_monitor_only are mutually exclusive"
        )

    clearml_enabled = args.use_clearml or args.clearml_monitor_only or is_clearml_worker
    if not clearml_enabled:
        print('Running locally (ClearML disabled)')
        return None

    from clearml import Task
    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task,
        reuse_last_task_id=False,
        tags=['test run'],
        output_uri="s3://tks-zx.fzi.de:9000/ri928",
        auto_connect_frameworks={"tensorboard": False},
        auto_resource_monitoring=False,
    )

    if is_clearml_worker:
        from util.clearml import download_clearml_dataset

        if args.clearml_dataset_project and args.clearml_dataset_name:
            dataset_dir = download_clearml_dataset(
                args.clearml_dataset_project, args.clearml_dataset_name,
            )
            _remap_clearml_dataset_paths(args, dataset_dir)
            print(f'Using ClearML worker dataset: {dataset_dir}')
        else:
            print('Running on ClearML worker without dataset remap')

        return task

    task.connect(vars(args))

    if args.use_clearml:
        # Configure the ClearML agent runtime only for remote execution.
        task.set_base_docker(
            "tks-zx.fzi.de/hu778/dcd-nocturne",
            docker_setup_bash_script=[
                "apt-get install -y libgl1 ffmpeg imagemagick",
            ],
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all --network=host",
        )
        task.execute_remotely('default', clone=False, exit_process=True)

    if args.clearml_monitor_only:
        print('Using ClearML monitor in local mode')
    else:
        print('Using ClearML')
    return task


def _build_external_teacher(args, device):
    """Create an ExternalTeacher for batched ctrl-sim inference."""
    from batch_inference import ExternalTeacher, build_external_teacher_kwargs

    teacher = ExternalTeacher(
        **build_external_teacher_kwargs(
            checkpoint_path=args.opponent_checkpoint,
            device=device,
            inference_precision=getattr(args, 'inference_precision', 'fp32'),
            config_source=args,
        )
    )
    teacher.validate_student_action_space(
        student_accel_discretization=args.student_accel_discretization,
        student_steer_discretization=args.student_steer_discretization,
    )
    return teacher

def main(args, clearml_task=None):
    display = None
    filewriter = None
    training_succeeded = False

    from util import ignore_warning
    from util import (
        FileWriter,
        create_parallel_env,
        get_tilt_range_from_args,
        make_agent,
        make_plr_args,
        safe_checkpoint,
        save_images,
    )
    from util.clearml import finalize_clearml_run
    import envs.multigrid.adversarial
    import envs.box2d
    import envs.nocturne_ctrlsim
    from envs.bipedalwalker import bipedalwalker_df_from_encodings
    from envs.runners.adversarial_runner import AdversarialRunner
    from eval import Evaluator

    ignore_warning.configure_subprocess_env()
    display = setup_virtual_display()
    try:
        warmup_disable_steps = int(args.num_env_steps * args.warmup_opponent_disable_ratio)
        warmup_replay_steps = int(args.num_env_steps * args.warmup_opponent_replay_ratio)

        # Set opponent vehicle runtime mode: disable / replay / normal.
        def resolve_opponent_runtime_mode_by_steps(total_env_steps: int) -> str:
            if total_env_steps < warmup_disable_steps:
                return 'disable'
            replay_end_step = warmup_disable_steps + warmup_replay_steps
            if total_env_steps < replay_end_step:
                return 'replay'
            return 'normal'
        
        # === Configure logging ==
        if args.xpid is None:
            args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
        filewriter = FileWriter(
            xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir
        )
        screenshot_dir = os.path.join(log_dir, args.xpid, 'screenshots')
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir, exist_ok=True)

        def log_stats(stats, tick=None):
            key_aliases = {
                'collision_occurred': 'collision',
                'offroad_occurred': 'offroad',
                'goal_reached_occurred': 'goal_reached',
                'position_reached_occurred': 'position_reached',
                'avg_progress': 'progress',
                'scenario_episode_reward': 'update_reward',
                'plr_scenario_episode_reward': 'plr_update_reward',
            }
            ordered_prefix = [
                'process_idx',
                'seed',
                'scenario_id',
                'collision',
                'offroad',
                'position_reached',
                'goal_reached',
                'progress',
                'plr_progress',
                'opponent_vehicle_num',
                'plr_opponent_vehicle_num',
                'steps',
                'total_episodes',
                'episode_reward',
                'plr_episode_reward',
                'total_student_grad_updates',
                'mean_agent_return',
                'agent_value_loss',
                'agent_pg_loss',
                'agent_dist_entropy',
            ]

            def normalize_and_reorder(row):
                normalized = {}
                for k, v in row.items():
                    normalized[key_aliases.get(k, k)] = v

                reordered = {}
                for k in ordered_prefix:
                    if k in normalized:
                        reordered[k] = normalized[k]
                for k, v in normalized.items():
                    if k not in reordered:
                        reordered[k] = v
                return reordered

            per_process_stats = stats.get('_per_process_stats')
            if per_process_stats:
                shared_stats = {
                    k: v for k, v in stats.items()
                    if k not in ('_per_process_stats', '_tb_per_process_stats')
                }
                for process_stats in per_process_stats:
                    row = shared_stats.copy()
                    row.update(process_stats)
                    row = normalize_and_reorder(row)
                    filewriter.log(row, tick=tick)
                    if args.verbose:
                        key_excluded = {k: () for k in row.keys()}
                        HumanOutputFormat(sys.stdout).write(row, key_excluded=key_excluded, step=0)
                return

            stats = normalize_and_reorder(stats)
            filewriter.log(stats, tick=tick)
            if args.verbose:
                # 构造 key_excluded 字典（所有 key 都不排除）
                # Build the key_excluded dictionary so that no key is excluded.
                key_excluded = {k: () for k in stats.keys()}
                HumanOutputFormat(sys.stdout).write(stats, key_excluded=key_excluded, step=0)

        def log_final_test_eval(env_returns, agent_name):
            if env_returns is None:
                return
            for env_name, returns in env_returns.items():
                if returns:
                    mean_return = float(np.mean(returns))
                    median_return = float(np.median(returns))
                else:
                    mean_return = ''
                    median_return = ''
                filewriter.log_final_test_eval({
                    'env_name': env_name,
                    'agent_name': agent_name,
                    'num_test_seeds': '',
                    'mean_episode_return': mean_return,
                    'median_episode_return': median_return,
                })

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.disable(logging.CRITICAL)

        # === Determine device ====
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda:0" if args.cuda else "cpu")
        if 'cuda' in device.type:
            torch.backends.cudnn.benchmark = True
            print('Using CUDA\n')

        # Set initial opponent runtime mode before env construction.
        args.opponent_runtime_mode = resolve_opponent_runtime_mode_by_steps(0)

        # === Create parallel envs ===
        venv, ued_venv = create_parallel_env(args)

        is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
        is_paired = args.ued_algo in ['paired', 'flexible_paired']

        agent = make_agent(name='agent', env=venv, args=args, device=device)
        adversary_agent, adversary_env = None, None
        if is_paired or args.use_accel_paired:
            adversary_agent = make_agent(name='adversary_agent', env=venv, args=args, device=device)

        if is_training_env:
            adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
        if args.ued_algo == 'domain_randomization' and args.use_plr and not args.use_reset_random_dr:
            adversary_env = make_agent(name='adversary_env', env=venv, args=args, device=device)
            adversary_env.random()

        # === Create runner ===
        plr_args = None
        warmup_plr_args = None
        normal_plr_args = None
        if args.use_plr:
            warmup_plr_args = make_plr_args(
                args,
                venv.observation_space,
                venv.action_space,
                use_warmup_buffer=True,
            )
            normal_plr_args = make_plr_args(
                args,
                venv.observation_space,
                venv.action_space,
                use_warmup_buffer=False,
            )
            initial_use_warmup_buffer = args.opponent_runtime_mode in ('disable', 'replay')
            plr_args = warmup_plr_args if initial_use_warmup_buffer else normal_plr_args
        train_runner = AdversarialRunner(
            args=args,
            venv=venv,
            agent=agent, 
            ued_venv=ued_venv, 
            adversary_agent=adversary_agent,
            adversary_env=adversary_env,
            flexible_protagonist=False,
            train=True,
            plr_args=plr_args,
            device=device,
            external_teacher=(
                _build_external_teacher(args, device)
                if args.env_name.startswith('Nocturne')
                else None
            ),
        )

        # === Configure checkpointing ===
        timer = timeit.default_timer
        initial_update_count = 0
        last_logged_update_at_restart = -1
        checkpoint_dir = os.path.expandvars(
            os.path.expanduser("%s/%s" % (log_dir, args.xpid))
        )
        latest_checkpoint_path = None
        if os.path.isdir(checkpoint_dir):
            checkpoint_candidates = []
            for filename in os.listdir(checkpoint_dir):
                if not filename.startswith('model_') or not filename.endswith('.tar'):
                    continue
                checkpoint_index_text = filename[len('model_'):-len('.tar')]
                if checkpoint_index_text.isdigit():
                    checkpoint_candidates.append(
                        (
                            int(checkpoint_index_text),
                            os.path.join(checkpoint_dir, filename),
                        )
                    )
            if checkpoint_candidates:
                _, latest_checkpoint_path = max(
                    checkpoint_candidates,
                    key=lambda item: item[0],
                )
        ## This is only used for the first iteration of finetuning
        if args.xpid_finetune:
            model_fname = f'{args.model_finetune}.tar'
            base_checkpoint_path = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid_finetune, model_fname))
            )

        def checkpoint(index=None):
            if args.disable_checkpoint:
                return
            if index is None:
                return

            checkpoint_path = os.path.join(checkpoint_dir, f'model_{index}.tar')
            safe_checkpoint(
                {'runner_state_dict': train_runner.state_dict()},
                checkpoint_path,
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)


        # === Load checkpoint ===
        if args.checkpoint and latest_checkpoint_path is not None:
            checkpoint_states = torch.load(
                latest_checkpoint_path,
                map_location=lambda storage, loc: storage,
            )
            last_logged_update_at_restart = filewriter.latest_tick() # ticks are 0-indexed updates
            train_runner.load_state_dict(checkpoint_states['runner_state_dict'])
            initial_update_count = train_runner.num_updates
            logging.info(
                "Resuming preempted job from %s after %d updates\n",
                latest_checkpoint_path,
                initial_update_count,
            ) # 0-indexed next update
        elif args.xpid_finetune and latest_checkpoint_path is None:
            checkpoint_states = torch.load(base_checkpoint_path)
            state_dict = checkpoint_states['runner_state_dict']
            agent_state_dict = state_dict.get('agent_state_dict')
            optimizer_state_dict = state_dict.get('optimizer_state_dict')
            train_runner.agents['agent'].algo.actor_critic.load_state_dict(agent_state_dict['agent'])
            train_runner.agents['agent'].algo.optimizer.load_state_dict(optimizer_state_dict['agent'])

        # === Set up Evaluator ===
        evaluator = None
        if args.test_env_names:
                eval_video_dir = args.eval_video_dir
                if eval_video_dir is None:
                    eval_video_dir = os.path.join(log_dir, args.xpid, 'videos')
                evaluator = Evaluator(
                    args.test_env_names.split(','), 
                    num_processes=args.test_num_processes, 
                    num_episodes=args.test_num_episodes,
                    record_video=args.eval_record_video,
                    video_dir=eval_video_dir,
                    eval_screenshot=args.eval_screenshot,
                    eval_screenshot_dir=screenshot_dir,
                    frame_stack=args.frame_stack,
                    grayscale=args.grayscale,
                    num_action_repeat=args.num_action_repeat,
                    use_global_critic=args.use_global_critic,
                    use_global_policy=args.use_global_policy,
                    device=device,
                    scenario_index_path=args.scenario_index_path,
                    opponent_checkpoint=args.opponent_checkpoint,
                    student_accel_discretization=args.student_accel_discretization,
                    student_steer_discretization=args.student_steer_discretization,
                    opponent_vehicle_number=args.opponent_vehicle_number,
                    action_repeat_frequency=args.action_repeat_frequency,
                    kl_loss_computation_frequency=args.kl_loss_computation_frequency,
                    sparse_inference_action_repeat=args.sparse_inference_action_repeat,
                    scenario_data_dir=args.scenario_data_dir,
                    preprocess_dir=args.preprocess_dir,
                    vehicle_map_path=args.vehicle_map_path,
                    max_episode_steps=args.nocturne_max_episode_steps,
                    tilt_range=get_tilt_range_from_args(args),
                )

        # === Train === 
        last_checkpoint_idx = getattr(train_runner, args.checkpoint_basis)
        update_start_time = timer()
        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
        train_runner.total_updates = num_updates
        env_steps_per_update = args.num_processes * args.num_steps
        current_opponent_runtime_mode = None
        train_updates = range(initial_update_count, num_updates)
        with tqdm(
            train_updates,
            total=num_updates,
            initial=initial_update_count,
            desc='Training',
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            for j in pbar:
                completed_env_steps_before_update = j * env_steps_per_update

                target_mode = resolve_opponent_runtime_mode_by_steps(completed_env_steps_before_update)
                previous_mode_by_steps = resolve_opponent_runtime_mode_by_steps(
                    max(completed_env_steps_before_update - env_steps_per_update, 0)
                )
                entering_normal_stage = (
                    args.use_plr
                    and target_mode == 'normal'
                    and completed_env_steps_before_update > 0
                    and previous_mode_by_steps in ('disable', 'replay')
                )
                if entering_normal_stage:
                    if train_runner.reset_plr_buffers(plr_args=normal_plr_args):
                        logging.getLogger("logs/out").info(
                            (
                                "Reset PLR buffers at env_step=%d (update=%d) "
                                "when stage switches from %s to normal."
                            ),
                            completed_env_steps_before_update,
                            j,
                            previous_mode_by_steps,
                        )

                if target_mode != current_opponent_runtime_mode:
                    venv.remote_attr(
                        'set_opponent_runtime_mode',
                        data=target_mode,
                        flatten=True,
                    )
                    current_opponent_runtime_mode = target_mode
                    logging.getLogger("logs/out").info(
                        "Opponent runtime mode to %s at env_step=%d (update=%d).",
                        target_mode,
                        completed_env_steps_before_update,
                        j,
                    )

                stats = train_runner.run()
                filewriter.report_step_metrics(stats)

                # === Perform logging ===
                if train_runner.num_updates <= last_logged_update_at_restart:
                    continue

                log = (j % args.log_interval == 0) or j == num_updates - 1
                save_screenshot = \
                    args.screenshot_interval > 0 and \
                        (j % args.screenshot_interval == 0)

                if log:
                    # Eval
                    test_stats = {}
                    agent_env_returns = None
                    adv_env_returns = None
                    if evaluator is not None and (j % args.test_interval == 0 or j == num_updates - 1):
                        if j == num_updates - 1:
                            test_stats, agent_env_returns = evaluator.evaluate(
                                train_runner.agents['agent'],
                                return_episode_returns=True,
                                external_teacher=train_runner.external_teacher)
                        else:
                            test_stats = evaluator.evaluate(
                                train_runner.agents['agent'],
                                external_teacher=train_runner.external_teacher)
                        stats.update(test_stats)
                        if train_runner.agents.get('adversary_agent') is not None and (
                            args.use_accel_paired or j == num_updates - 1
                        ):
                            if j == num_updates - 1:
                                adv_test_stats, adv_env_returns = evaluator.evaluate(
                                    train_runner.agents['adversary_agent'],
                                    return_episode_returns=True,
                                    external_teacher=train_runner.external_teacher)
                            else:
                                adv_test_stats = evaluator.evaluate(
                                    train_runner.agents['adversary_agent'],
                                    external_teacher=train_runner.external_teacher)
                            if args.use_accel_paired:
                                curr_keys = list(adv_test_stats.keys())
                                for curr_key in curr_keys:
                                    adv_test_stats[f"advagent_{curr_key}"] = adv_test_stats[curr_key]
                                    adv_test_stats.pop(curr_key, None)
                                stats.update(adv_test_stats)

                        if j == num_updates - 1:
                            log_final_test_eval(agent_env_returns, 'agent')
                            log_final_test_eval(adv_env_returns, 'adversary_agent')
                    elif evaluator is not None:
                        stats.update({k:None for k in evaluator.get_stats_keys()})

                    update_end_time = timer()
                    num_incremental_updates = 1 if j == 0 else args.log_interval
                    sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
                    update_start_time = update_end_time
                    stats.update({'sps': sps})
                    stats.update(test_stats) # Ensures sps column is always before test stats
                    log_tick = j if '_per_process_stats' in stats else None
                    log_stats(stats, tick=log_tick)

                # === Log PLR level weights ===
                if args.weight_log_interval > 0 and \
                    j % args.weight_log_interval == 0 and \
                    train_runner.level_samplers:
                    # Get the first available level sampler (usually 'agent')
                    level_sampler = train_runner.all_level_samplers[0]
                    if level_sampler is not None and level_sampler._proportion_filled > 0:
                        weights = level_sampler.sample_weights()
                        seeds = level_sampler.seeds
                        filewriter.log_level_weights(weights, seeds)

                checkpoint_idx = getattr(train_runner, args.checkpoint_basis)

                if checkpoint_idx != last_checkpoint_idx:
                    is_last_update = j == num_updates - 1
                    if is_last_update or \
                        (train_runner.num_updates > 0 and checkpoint_idx % args.checkpoint_interval == 0):
                        checkpoint(checkpoint_idx)
                        logging.info(f"\nSaved checkpoint after update {j}")
                        logging.info(f"\nLast update: {is_last_update}")
                    elif train_runner.num_updates > 0 and args.archive_interval > 0 \
                        and checkpoint_idx % args.archive_interval == 0:
                        checkpoint(checkpoint_idx)
                        logging.info(f"\nArchived checkpoint after update {j}")

                if save_screenshot:
                    level_info = train_runner.sampled_level_info
                    if args.env_name.startswith('BipedalWalker'):
                        encodings = venv.get_level()
                        df = bipedalwalker_df_from_encodings(args.env_name, encodings)
                        if args.use_editor and level_info:
                            df.to_csv(os.path.join(
                                screenshot_dir, 
                                f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.csv"))
                        else:
                            df.to_csv(os.path.join(
                                screenshot_dir, 
                                f'update{j}.csv'))
                    else:
                        venv.reset_agent()
                        images = venv.get_images()
                        if args.env_name.startswith('Nocturne'):
                            for idx, img in enumerate(images):
                                save_images(
                                    [img],
                                    os.path.join(screenshot_dir, f'update{j}_process{idx}.png'),
                                    normalize=True, channels_first=False)
                        elif args.use_editor and level_info:
                            save_images(
                                images[:args.screenshot_batch_size],
                                os.path.join(
                                    screenshot_dir,
                                    f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"),
                                normalize=True, channels_first=False)
                        else:
                            save_images(
                                images[:args.screenshot_batch_size],
                                os.path.join(screenshot_dir, f'update{j}.png'),
                                normalize=True, channels_first=False)
                        plt.close()

        if evaluator is not None:
            evaluator.close()
        venv.close()
        training_succeeded = True
    finally:
        if filewriter is not None:
            finalize_clearml_run(
                filewriter=filewriter,
                clearml_task=clearml_task,
                successful=training_succeeded,
            )
        if display:
            display.stop()


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"

    args = resolve_runtime_args()
    clearml_task = init_clearml(args)
    main(args, clearml_task)
