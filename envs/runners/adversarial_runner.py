# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque, defaultdict

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

from level_replay import LevelSampler, LevelStore
from util import \
    array_to_csv, \
    is_discrete_actions, \
    get_obs_at_index

from teachDeepRL.teachers.teacher_controller import TeacherController

import matplotlib as mpl
import matplotlib.pyplot as plt

from ctrlsim_adapter.regret_enhancement_implementation import (
    append_nocturne_rtg_segment,
    build_nocturne_rtg_record,
    compute_nocturne_enhanced_regret_scores,
    resolve_rollout_seed,
)
from ctrlsim_adapter.ego_ctrlsim_rollout_implementation import (
    collect_ego_ctrlsim_action_logits,
    run_nocturne_batched_step,
)
from ctrlsim_adapter.nocturne_stats_implementation import (
    build_nocturne_process_stats,
    compute_nocturne_env_stats,
)


class AdversarialRunner(object):
    """
    Performs rollouts of an adversarial environment, given 
    protagonist (agent), antogonist (adversary_agent), and
    environment adversary (advesary_env)
    """
    def __init__(
        self,
        args,
        venv,
        agent,
        ued_venv=None,
        adversary_agent=None,
        adversary_env=None,
        flexible_protagonist=False,
        train=False,
        plr_args=None,
        device='cpu',
        external_teacher=None):
        """
        venv: Vectorized, adversarial gym env with agent-specific wrappers.
        agent: Protagonist trainer.
        ued_venv: Vectorized, adversarial gym env with adversary-env-specific wrappers.
        adversary_agent: Antogonist trainer.
        adversary_env: Environment adversary trainer.

        flexible_protagonist: Which agent plays the role of protagonist in
            calculating the regret depends on which has the lowest score.
        """
        self.args = args

        self.venv = venv
        if ued_venv is None:
            self.ued_venv = venv
        else:
            self.ued_venv = ued_venv # Since adv env can have different env wrappers

        self.is_discrete_actions = is_discrete_actions(self.venv)
        self.is_discrete_adversary_env_actions = is_discrete_actions(self.venv, adversary=True)

        self.agents = {
            'agent': agent,
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env,
        }

        self.agent_rollout_steps = args.num_steps
        self.adversary_env_rollout_steps = self.venv.adversary_observation_space['time_step'].high[0]

        self.is_dr = args.ued_algo == 'domain_randomization'
        self.is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
        self.is_paired = args.ued_algo in ['paired', 'flexible_paired']
        self.requires_batched_vloss = (args.use_editor and args.base_levels == 'easy' and args.use_accel_paired==False)

        self.is_alp_gmm = args.ued_algo == 'alp_gmm' 

        # Track running mean and std of env returns for return normalization
        if args.adv_normalize_returns:
            self.env_return_rms = RunningMeanStd(shape=())

        self.device = device

        # External teacher for batched inference (None = disabled)
        self.external_teacher = external_teacher

        if train:
            self.train()
        else:
            self.eval()

        self.reset()        
        self.use_accel_paired = args.use_accel_paired

        # Set up PLR
        self._plr_args = plr_args
        self.level_store = None
        self.level_samplers = {}
        self.warmup_level_samplers = {}
        self.current_level_seeds = None
        self._default_level_sampler = None
        self._enhanced_regret_base_sum = 0.0
        self._enhanced_regret_base_count = 0
        self._enhanced_regret_delta_sum = 0.0
        self._enhanced_regret_delta_count = 0
        self._default_warmup_level_sampler = None
        self.weighted_num_edits = 0
        self.latest_env_stats = defaultdict(float)
        self.latest_env_process_stats = []
        self._init_or_reset_plr_state(self._plr_args)

        # Set up ALP-GMM
        if self.is_alp_gmm:
            self._init_alp_gmm()

        # Runtime gate for PLR (enabled by default across stages).
        self.plr_active = True

    def _get_enhanced_regret_running_means(
        self,
    ) -> tuple[float | None, float | None]:
        """Return current running means for top-level enhanced regret normalization."""
        base_count = int(getattr(self, "_enhanced_regret_base_count", 0))
        delta_count = int(getattr(self, "_enhanced_regret_delta_count", 0))
        base_mean = None
        delta_mean = None
        if base_count > 0:
            base_mean = float(
                getattr(self, "_enhanced_regret_base_sum", 0.0)
            ) / float(base_count)
        if delta_count > 0:
            delta_mean = float(
                getattr(self, "_enhanced_regret_delta_sum", 0.0)
            ) / float(delta_count)
        return base_mean, delta_mean

    def _update_enhanced_regret_running_means(
        self,
        *,
        base_regret_by_seed: dict[int, float],
        delta_rtg_by_seed: dict[int, float],
    ) -> None:
        """Update running means from per-seed raw enhanced regret terms."""
        if not hasattr(self, "_enhanced_regret_base_sum"):
            self._enhanced_regret_base_sum = 0.0
            self._enhanced_regret_base_count = 0
        if not hasattr(self, "_enhanced_regret_delta_sum"):
            self._enhanced_regret_delta_sum = 0.0
            self._enhanced_regret_delta_count = 0

        for value in base_regret_by_seed.values():
            self._enhanced_regret_base_sum += float(value)
            self._enhanced_regret_base_count += 1
        for value in delta_rtg_by_seed.values():
            self._enhanced_regret_delta_sum += float(value)
            self._enhanced_regret_delta_count += 1

    def _build_level_samplers(self, plr_args):
        level_samplers = {}

        if self.is_paired:
            if not self.args.protagonist_plr and not self.args.antagonist_plr:
                level_samplers.update({
                    'agent': LevelSampler(**plr_args),
                    'adversary_agent': LevelSampler(**plr_args)
                })
            elif self.args.protagonist_plr:
                level_samplers['agent'] = LevelSampler(**plr_args)
            elif self.args.antagonist_plr:
                level_samplers['adversary_agent'] = LevelSampler(**plr_args)
        else:
            level_samplers['agent'] = LevelSampler(**plr_args)

        return level_samplers

    def _build_level_store(self):
        if self.use_byte_encoding:
            example = self.ued_venv.get_encodings()[0]
            data_info = {
                'numpy': True,
                'dtype': example.dtype,
                'shape': example.shape
            }
            return LevelStore(data_info=data_info)
        return LevelStore()

    def _build_warmup_plr_args(self, plr_args):
        """Build PLR kwargs for the warmup replay buffer."""
        warmup_plr_args = dict(plr_args)
        warmup_plr_args['seed_buffer_size'] = int(
            getattr(self.args, 'warmup_level_replay_seed_buffer_size', 8000)
        )
        return warmup_plr_args

    def _init_or_reset_plr_state(self, plr_args):
        self.level_store = None
        self.level_samplers = {}
        self.warmup_level_samplers = {}
        self.current_level_seeds = None
        self._default_level_sampler = None
        self._default_warmup_level_sampler = None

        if not plr_args:
            self.use_editor = False
            self.edit_prob = 0
            self.base_levels = None
            return False

        self.level_samplers = self._build_level_samplers(plr_args)
        if getattr(self.args, 'use_warmup_level_replay', False):
            warmup_plr_args = self._build_warmup_plr_args(plr_args)
            self.warmup_level_samplers = self._build_level_samplers(warmup_plr_args)
        self.level_store = self._build_level_store()
        self.current_level_seeds = [-1 for _ in range(self.args.num_processes)]

        if self.level_samplers:
            self._default_level_sampler = next(iter(self.level_samplers.values()))
        if self.warmup_level_samplers:
            self._default_warmup_level_sampler = next(
                iter(self.warmup_level_samplers.values())
            )

        self.use_editor = self.args.use_editor
        self.edit_prob = self.args.level_editor_prob
        self.base_levels = self.args.base_levels
        return True

    def reset_plr_buffers(self, plr_args=None):
        if not self.args.use_plr:
            return False
        if plr_args is not None:
            self._plr_args = plr_args
        return self._init_or_reset_plr_state(self._plr_args)

    def _get_opponent_runtime_mode(self) -> str:
        """Return the current opponent runtime mode tracked by the runner."""
        if hasattr(self, 'opponent_runtime_mode'):
            return str(self.opponent_runtime_mode)
        return str(getattr(self.args, 'opponent_runtime_mode', 'normal'))

    @property
    def plr_runtime_enabled(self) -> bool:
        runtime_mode = self._get_opponent_runtime_mode()
        use_warmup_level_replay = bool(
            getattr(self.args, 'use_warmup_level_replay', False)
        )
        return bool(
            self.args.use_plr
            and self.plr_active
            and (
                runtime_mode == 'normal'
                or (runtime_mode == 'replay' and use_warmup_level_replay)
            )
        )

    def set_opponent_runtime_mode(self, mode: str) -> str:
        """Update the opponent runtime mode used by PLR runtime gates."""
        self.opponent_runtime_mode = str(mode)
        self.args.opponent_runtime_mode = self.opponent_runtime_mode
        return self.opponent_runtime_mode

    def set_plr_active(self, active: bool) -> bool:
        self.plr_active = bool(active)
        return self.plr_active

    @property
    def use_byte_encoding(self): # add support for Nocturne
        env_name = self.args.env_name
        if self.args.use_editor \
           or env_name.startswith('BipedalWalker') \
           or env_name.startswith('Nocturne') \
           or (env_name.startswith('MultiGrid') and self.args.use_reset_random_dr):
            return True
        else:
            return False

    def _init_alp_gmm(self):
        args = self.args
        param_env_bounds = []
        if args.env_name.startswith('MultiGrid'):
            param_env_bounds = {'actions':[0,168,26]}
            reward_bounds = None
        elif args.env_name.startswith('Bipedal'):
            if 'POET' in args.env_name:
                param_env_bounds = {'actions': [0,2,5]}
            else:
                param_env_bounds = {'actions': [0,2,8]}
            reward_bounds = (-200, 350)
        # TODO: Add Nocturne support if using ALP-GMM
        else:
            raise ValueError(f'Environment {args.env_name} not supported for ALP-GMM')

        self.alp_gmm_teacher = TeacherController(
                    teacher='ALP-GMM',
                    nb_test_episodes=0,
                    param_env_bounds=param_env_bounds,
                    reward_bounds=reward_bounds,
                    seed=args.seed,
                    teacher_params={}) # Use defaults

    def reset(self):
        self.num_updates = 0
        self.total_updates = 0
        self.total_num_edits = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        self.student_grad_updates = 0
        self.sampled_level_info = None
        self._pending_replay_obs = None

        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    def train(self):
        self.is_training = True
        [agent.train() if agent else agent for _,agent in self.agents.items()]

    def eval(self):
        self.is_training = False
        [agent.eval() if agent else agent for _,agent in self.agents.items()]

    def state_dict(self):
        agent_state_dict = {}
        optimizer_state_dict = {}
        for k, agent in self.agents.items():
            if agent:
                agent_state_dict[k] = agent.algo.actor_critic.state_dict()
                optimizer_state_dict[k] = agent.algo.optimizer.state_dict()

        return {
            'agent_state_dict': agent_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'agent_returns': self.agent_returns,
            'adversary_agent_returns': self.adversary_agent_returns,
            'num_updates': self.num_updates,
            'total_episodes_collected': self.total_episodes_collected,
            'total_seeds_collected': self.total_seeds_collected,
            'total_num_edits': self.total_num_edits,
            'student_grad_updates': self.student_grad_updates,
            'latest_env_stats': self.latest_env_stats,
            'latest_env_process_stats': self.latest_env_process_stats,
            'level_store': self.level_store,
            'level_samplers': self.level_samplers,
            'warmup_level_samplers': self.warmup_level_samplers,
        }

    def load_state_dict(self, state_dict):

        agent_state_dict = state_dict.get('agent_state_dict')

        for k,state in agent_state_dict.items():
            self.agents[k].algo.actor_critic.load_state_dict(state)

        optimizer_state_dict = state_dict.get('optimizer_state_dict')

        for k, state in optimizer_state_dict.items():
            self.agents[k].algo.optimizer.load_state_dict(state)

        self.agent_returns = state_dict.get('agent_returns')
        self.adversary_agent_returns = state_dict.get('adversary_agent_returns')
        self.num_updates = state_dict.get('num_updates')
        self.total_episodes_collected = state_dict.get('total_episodes_collected')
        self.total_seeds_collected = state_dict.get('total_seeds_collected')
        self.total_num_edits = state_dict.get('total_num_edits')
        self.student_grad_updates = state_dict.get('student_grad_updates')
        self.latest_env_stats = state_dict.get('latest_env_stats')
        self.latest_env_process_stats = state_dict.get('latest_env_process_stats', [])

        self.level_store = state_dict.get('level_store')
        self.level_samplers = state_dict.get('level_samplers')
        self.warmup_level_samplers = state_dict.get('warmup_level_samplers')

        if self.warmup_level_samplers is None:
            self.warmup_level_samplers = {}
            if (
                self.args.use_plr
                and getattr(self.args, 'use_warmup_level_replay', False)
                and self._plr_args
            ):
                warmup_plr_args = self._build_warmup_plr_args(self._plr_args)
                self.warmup_level_samplers = self._build_level_samplers(warmup_plr_args)

        if self.args.use_plr:
            if self.level_samplers:
                self._default_level_sampler = next(iter(self.level_samplers.values()))
            if self.warmup_level_samplers:
                self._default_warmup_level_sampler = next(
                    iter(self.warmup_level_samplers.values())
                )

            if self.use_editor:
                self.weighted_num_edits = self._get_weighted_num_edits()

    def _get_batched_value_loss(self, agent, clipped=True, batched=True):
        batched_value_loss = agent.storage.get_batched_value_loss(
            signed=False, 
            positive_only=False, 
            clipped=clipped,
            batched=batched)

        return batched_value_loss

    def _get_rollout_return_stats(self, rollout_returns):
        mean_return = torch.zeros(self.args.num_processes, 1)
        max_return = torch.zeros(self.args.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns 
        }

        return stats
    
    def _calculate_paired_regret_scores(self, agent_rollout_info, adversary_agent_rollout_info, type="paired"):
        if type=="paired":
            external_scores = torch.max(adversary_agent_rollout_info['max_return'] - agent_rollout_info['mean_return'], \
                    torch.zeros_like(agent_rollout_info['mean_return']))
        elif type=="flex_paired":
            env_return = torch.zeros_like(agent_rollout_info['max_return'], dtype=torch.float)
            adversary_agent_max_idx = adversary_agent_rollout_info['max_return'] > agent_rollout_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_rollout_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_rollout_info['max_return'][agent_max_idx]
            
            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_rollout_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_rollout_info['mean_return'][agent_max_idx]

            external_scores = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))
        else:
            raise NotImplementedError
            
        return external_scores

    def _get_env_stats_multigrid(self, agent_info, adversary_agent_info):
        num_blocks = np.mean(agent_info.get(
            'num_blocks', self.venv.get_num_blocks()))
        
        passable_ratio = np.mean(agent_info.get(
            'passable_ratio', self.venv.get_passable()))

        shortest_path_lengths = agent_info.get(
            'shortest_path_lengths', self.venv.get_shortest_path_length())
        shortest_path_length = np.mean(shortest_path_lengths)

        solved_idx =  agent_info.get('solved_idx', None)
        if solved_idx is None:
            if 'max_returns' in adversary_agent_info:
                solved_idx = \
                    (torch.max(agent_info['max_return'], \
                        adversary_agent_info['max_return']) > 0).numpy().squeeze()
            else:
                solved_idx = (agent_info['max_return'] > 0).numpy().squeeze()

        solved_path_lengths = np.array(shortest_path_lengths)[solved_idx]
        solved_path_length = np.mean(solved_path_lengths) if len(solved_path_lengths) > 0 else 0

        stats = {
            'num_blocks': num_blocks,
            'passable_ratio': passable_ratio,
            'shortest_path_length': shortest_path_length,
            'solved_path_length': solved_path_length
        }

        return stats

    def _get_plr_buffer_stats(self):
        stats = {}
        for k,sampler in self.level_samplers.items():
            stats[k + '_plr_passable_mass'] = sampler.solvable_mass
            stats[k + '_plr_max_score'] = sampler.max_score 
            stats[k + '_plr_weighted_num_edits'] = self.weighted_num_edits

        return stats

    def _get_env_stats_car_racing(self, agent_info, adversary_agent_info):
        infos = self.venv.get_complexity_info()
        num_envs = len(infos)

        sums = defaultdict(float)
        for info in infos:
            for k,v in info.items():
                sums[k] += v

        stats = {}
        for k,v in sums.items():
            stats['track_' + k] = sums[k]/num_envs

        return stats

    def _get_env_stats_bipedalwalker(self, agent_info, adversary_agent_info):
        infos = self.venv.get_complexity_info()
        num_envs = len(infos)

        sums = defaultdict(float)
        for info in infos:
            for k,v in info.items():
                sums[k] += v

        stats = {}
        for k,v in sums.items():
            stats['track_' + k] = sums[k]/num_envs

        return stats

    def _get_env_stats(self, agent_info, adversary_agent_info, log_replay_complexity=False, nocturne_infos=None):
        env_name = self.args.env_name
        if env_name.startswith('MultiGrid'):
            stats = self._get_env_stats_multigrid(agent_info, adversary_agent_info)
        elif env_name.startswith('CarRacing'):
            stats = self._get_env_stats_car_racing(agent_info, adversary_agent_info)
        elif env_name.startswith('BipedalWalker'):
            stats = self._get_env_stats_bipedalwalker(agent_info, adversary_agent_info)
        elif env_name.startswith('Nocturne') or env_name.startswith('nocturne'):
            tilting_mode = getattr(self.args, 'tilting_mode', 'per_vehicle')
            stats = compute_nocturne_env_stats(
                agent_info=agent_info,
                infos=nocturne_infos,
                venv=self.venv,
                tilting_mode=tilting_mode,
            )
        else:
            raise ValueError(f'Unsupported environment, {self.args.env_name}')

        stats_ = {}
        for k,v in stats.items():
            stats_['plr_' + k] = v if log_replay_complexity else None
            stats_[k] = v if not log_replay_complexity else None
            
        return stats_

    def _get_active_levels(self):
        assert self.args.use_plr, 'Only call _get_active_levels when using PLR.'

        env_name = self.args.env_name

        is_multigrid = env_name.startswith('MultiGrid')
        is_car_racing = env_name.startswith('CarRacing')
        is_bipedal_walker = env_name.startswith('BipedalWalker')

        if self.use_byte_encoding: # True in Nocturne env
            return [x.tobytes() for x in self.ued_venv.get_encodings()]
        elif is_multigrid:
            return self.agents['adversary_env'].storage.get_action_traj(as_string=True)
        else:
            return self.ued_venv.get_level()

    def _get_level_sampler(self, name):
        active_level_samplers = self._get_active_level_sampler_map()
        other = 'adversary_agent'
        if name == 'adversary_agent':
            other = 'agent'

        level_sampler = active_level_samplers.get(name) or active_level_samplers.get(other)

        updateable = name in active_level_samplers

        return level_sampler, updateable

    def _is_warmup_replay_plr_active(self):
        """Return whether the dedicated warmup replay PLR buffer is active."""
        return bool(
            getattr(self.args, 'use_warmup_level_replay', False)
            and self._get_opponent_runtime_mode() == 'replay'
        )

    def _get_active_level_sampler_map(self):
        """Return the sampler mapping active for the current runtime stage."""
        if self._is_warmup_replay_plr_active():
            return self.warmup_level_samplers
        return self.level_samplers

    def _get_active_default_level_sampler(self):
        """Return the default sampler active for the current runtime stage."""
        if self._is_warmup_replay_plr_active():
            return self._default_warmup_level_sampler
        return self._default_level_sampler

    @property
    def active_level_samplers(self):
        """Return active sampler instances for the current runtime stage."""
        active_level_samplers = self._get_active_level_sampler_map()
        return list(
            filter(lambda x: x is not None, [v for _, v in active_level_samplers.items()])
        )

    @property
    def all_level_samplers(self):
        seen = set()
        samplers = []
        for sampler_map in (self.level_samplers, self.warmup_level_samplers):
            for sampler in sampler_map.values():
                if sampler is None or id(sampler) in seen:
                    continue
                seen.add(id(sampler))
                samplers.append(sampler)
        return samplers

    def _should_edit_level(self):
        if self.use_editor:
            return np.random.rand() < self.edit_prob
        else:
            return False

    def _update_plr_with_current_unseen_levels(self, parent_seeds=None):
        args = self.args
        levels = self._get_active_levels()
        self.current_level_seeds = \
            self.level_store.insert(levels, parent_seeds=parent_seeds)
        if args.log_plr_buffer_stats or args.reject_unsolvable_seeds:
            passable = self.venv.get_passable()
        else:
            passable = None
        self._update_level_samplers_with_external_unseen_sample(
            self.current_level_seeds, solvable=passable)

    def _update_level_samplers_with_external_unseen_sample(self, seeds, solvable=None):
        level_samplers = self.active_level_samplers

        if self.args.reject_unsolvable_seeds:
            solvable = np.array(solvable, dtype='bool')
            seeds = np.array(seeds, dtype=np.int64)[solvable]
            solvable = solvable[solvable]

        for level_sampler in level_samplers:
            level_sampler.observe_external_unseen_sample(seeds, solvable)

    def _reconcile_level_store_and_samplers(self):
        all_replay_seeds = set()
        for level_sampler in self.all_level_samplers:
            all_replay_seeds.update([x for x in level_sampler.seeds if x >= 0])
        self.level_store.reconcile_seeds(all_replay_seeds)

    def _get_weighted_num_edits(self):
        level_sampler = self._get_active_default_level_sampler()
        seed_num_edits = np.zeros(level_sampler.seed_buffer_size)
        for idx, value in enumerate(self.level_store.seed2parent.values()):
            seed_num_edits[idx] = len(value)
        weighted_num_edits = np.dot(level_sampler.sample_weights(), seed_num_edits)
        return weighted_num_edits

    def _resolve_non_plr_base_seed(self, seed):
        """
        Resolve the non-PLR base seed (root ancestor) for an edited replay seed.

        Returns:
            int: root/base seed if resolvable.
            None: when root seed can no longer be resolved from LevelStore.
        """
        if self.level_store is None or seed is None:
            return None

        seed = int(seed)
        if seed not in self.level_store.seed2level:
            return None

        parent_levels = self.level_store.seed2parent.get(seed, [])
        if not parent_levels:
            return seed

        root_level = parent_levels[0]
        root_seed = self.level_store.level2seed.get(root_level)
        if root_seed is None:
            return None
        if root_seed not in self.level_store.seed2level:
            return None

        return int(root_seed)

    def _sample_replay_decision(self):
        default_level_sampler = self._get_active_default_level_sampler()
        return default_level_sampler.sample_replay_decision()

    def _consume_pending_replay_obs(self):
        """Return and clear the one-shot replay observation prepared for agent rollout."""
        pending_obs = getattr(self, "_pending_replay_obs", None)
        self._pending_replay_obs = None
        return pending_obs

    def agent_rollout(self, 
                      agent, 
                      num_steps, 
                      update=False, 
                      is_env=False, 
                      level_replay=False, 
                      level_sampler=None, 
                      update_level_sampler=False,
                      discard_grad=False, 
                      sample_only=False,
                      edit_level=False,
                      num_edits=0, 
                      fixed_seeds=None,
                      kl_dict=None,
                      update_agent_separately=False):
        if sample_only and discard_grad:
            raise ValueError("sample_only and discard_grad are mutually exclusive")

        args = self.args
        plr_runtime_enabled = self.plr_runtime_enabled
        if is_env:
            if edit_level: # Get mutated levels
                levels = [self.level_store.get_level(seed) for seed in fixed_seeds]
                if args.env_name.startswith('Nocturne'):
                    # Single-pass mutation path: avoid reset_to_level_batch + mutate_level
                    # double initialization in Nocturne editing.
                    self.ued_venv.mutate_level_batch(levels)
                else:
                    self.ued_venv.reset_to_level_batch(levels)
                    self.ued_venv.mutate_level(num_edits=num_edits)
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                return
            if level_replay: # Get replay levels
                self.current_level_seeds = [
                    level_sampler.sample_replay_level()
                    for _ in range(args.num_processes)
                ]
                levels = [self.level_store.get_level(seed) for seed in self.current_level_seeds]
                self._pending_replay_obs = self.ued_venv.reset_to_level_batch(levels)
                return self.current_level_seeds
            elif self.is_dr and not plr_runtime_enabled: 
                obs = self.ued_venv.reset_random() # don't need obs here
                self.total_seeds_collected += args.num_processes
                return
            elif self.is_dr and plr_runtime_enabled and args.use_reset_random_dr:
                obs = self.ued_venv.reset_random() # don't need obs here
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                self.total_seeds_collected += args.num_processes
                return
            elif self.is_alp_gmm:
                obs = self.alp_gmm_teacher.set_env_params(self.ued_venv)
                self.total_seeds_collected += args.num_processes
                return
            else:
                obs = self.ued_venv.reset() # Prepare for constructive rollout
                self.total_seeds_collected += args.num_processes
        else:
            obs = self._consume_pending_replay_obs()
            if obs is None:
                obs = self.venv.reset_agent()

        # Initialize first observation
        agent.storage.copy_obs_to_index(obs,0)
        
        rollout_info = {}
        rollout_returns = [[] for _ in range(args.num_processes)]
        rollout_done_count_by_process = np.zeros(args.num_processes, dtype=np.int64)
        rollout_collision_done_count_by_process = np.zeros(
            args.num_processes, dtype=np.int64
        )
        rollout_offroad_done_count_by_process = np.zeros(
            args.num_processes, dtype=np.int64
        )
        is_nocturne_rollout = (
            (not is_env)
            and (
                args.env_name.startswith('Nocturne')
                or args.env_name.startswith('nocturne')
            )
        )
        first_done_info_by_process = {} if is_nocturne_rollout else None
        track_nocturne_enhanced_regret = (
            is_nocturne_rollout
            and bool(getattr(args, "use_enhanced_regret", False))
        )
        if track_nocturne_enhanced_regret:
            attempt_count_by_seed = defaultdict(int)
            success_count_by_seed = defaultdict(int)
            delta_rtg_segments_by_seed = defaultdict(list)
            active_seed_by_process = [None for _ in range(args.num_processes)]
            active_rtg_records_by_process = [
                [] for _ in range(args.num_processes)
            ]
            rtg_gap_method = getattr(
                args,
                "rtg_difference_in_regret",
                "first_inference_step_gap",
            )
            regret_component_weights = (
                float(getattr(args, "regret_enhancement_w_goal", 1.0)),
                float(getattr(args, "regret_enhancement_w_veh", 1.0)),
                float(getattr(args, "regret_enhancement_w_edge", 1.0)),
            )
        
        if self.use_accel_paired:
            actor_seeds = {i: [] for i in range(args.num_processes)}

        if level_sampler and level_replay:
            rollout_info.update({
                'solved_idx': np.zeros(args.num_processes, dtype='bool')
            })
            
        for step in range(num_steps):
            if args.render:
                self.venv.render_to_screen()
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = agent.act(
                    obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])
                is_discrete_action = (
                    self.is_discrete_adversary_env_actions if is_env else self.is_discrete_actions
                )
                if is_discrete_action:
                    action_log_prob = action_log_dist.gather(-1, action.long())
                else:
                    action_log_prob = action_log_dist

            # Observe reward and next obs
            # Keep post-done resets on the current level for student rollouts.
            reset_random = False
            auto_reset_on_done = not is_env
            _action = agent.process_action(action.cpu())

            if is_env:
                obs, reward, done, infos = self.ued_venv.step_adversary(_action)
            elif args.env_name.startswith('Nocturne'):
                obs, reward, done, infos = run_nocturne_batched_step(
                    venv=self.venv,
                    external_teacher=self.external_teacher,
                    args=args,
                    action=_action,
                    reset_random=reset_random,
                    auto_reset_on_done=auto_reset_on_done,
                )
                if args.clip_reward:
                    reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)
            else:
                obs, reward, done, infos = self.venv.step_env(
                    _action,
                    reset_random=reset_random,
                    auto_reset_on_done=auto_reset_on_done,
                )
                if args.clip_reward:
                    reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)

            if not is_env:
                rollout_done_count_by_process += done.astype(np.int64)
                for process_idx, (done_, info) in enumerate(zip(done, infos)):
                    if not bool(done_):
                        continue
                    if float(info.get("collision_occurred", 0.0)) > 0.0:
                        rollout_collision_done_count_by_process[process_idx] += 1
                    if float(info.get("offroad_occurred", 0.0)) > 0.0:
                        rollout_offroad_done_count_by_process[process_idx] += 1

            if not is_env and step >= num_steps - 1:
                # Handle early termination due to cliffhanger rollout
                if agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = get_obs_at_index(obs, i)

                done = np.ones_like(done, dtype=np.float64)

            for i, info in enumerate(infos):
                if track_nocturne_enhanced_regret:
                    step_seed = resolve_rollout_seed(
                        self.current_level_seeds,
                        i,
                        info,
                    )
                    if step_seed is not None:
                        active_seed_by_process[i] = step_seed
                    rtg_record = build_nocturne_rtg_record(info)
                    if rtg_record is not None:
                        active_rtg_records_by_process[i].append(rtg_record)

                if 'episode' in info.keys():
                    if is_nocturne_rollout:
                        if i not in first_done_info_by_process:
                            first_done_info_by_process[i] = dict(info)
                    if track_nocturne_enhanced_regret:
                        seed = active_seed_by_process[i]
                        if seed is not None:
                            attempt_count_by_seed[seed] += 1
                            success_count_by_seed[seed] += int(
                                float(info.get("success", 0.0)) > 0.0
                            )
                            append_nocturne_rtg_segment(
                                seed=seed,
                                records=active_rtg_records_by_process[i],
                                delta_rtg_segments_by_seed=delta_rtg_segments_by_seed,
                                rtg_gap_method=rtg_gap_method,
                                regret_component_weights=regret_component_weights,
                            )
                        active_rtg_records_by_process[i] = []
                    rollout_returns[i].append(info['episode']['r'])
                    
                    if self.use_accel_paired:
                        actor_seeds[i].append(self.current_level_seeds[i])

                    if reset_random:
                        self.total_seeds_collected += 1

                    if not is_env:
                        self.total_episodes_collected += 1

                        # Handle early termination
                        if agent.storage.use_proper_time_limits:
                            if 'truncated_obs' in info.keys():
                                truncated_obs = info['truncated_obs']
                                agent.storage.insert_truncated_obs(truncated_obs, index=i)

                        # During one PPO rollout, replay episodes stay on the
                        # same level and only reset agent/environment state.
                        if level_sampler and level_replay:
                            rollout_info['solved_idx'][i] = True

                        # If using ALP-GMM, sample next level
                        if self.is_alp_gmm:
                            self.alp_gmm_teacher.record_train_episode(rollout_returns[i][-1], index=i)
                            self.alp_gmm_teacher.set_env_params(self.venv)

            # If done then clean the history of observations.
            # Use from_numpy (zero-copy) for `done` which is already a numpy
            # bool array; avoid list-of-lists construction for all three masks.
            masks = torch.from_numpy(
                (1.0 - done.astype(np.float32))[:, None]
            )
            bad_masks = torch.as_tensor(
                [0.0 if 'truncated' in info else 1.0 for info in infos],
                dtype=torch.float32,
            ).unsqueeze(1)
            cliffhanger_masks = torch.as_tensor(
                [0.0 if 'cliffhanger' in info else 1.0 for info in infos],
                dtype=torch.float32,
            ).unsqueeze(1)

            # Need to store level seeds alongside non-env agent steps
            current_level_seeds = None
            if (not is_env) and level_sampler:
                current_level_seeds = torch.tensor(self.current_level_seeds, dtype=torch.int).view(-1, 1)
            ego_ctrlsim_action_logits = None
            ego_ctrlsim_valid = None
            if is_nocturne_rollout:
                ego_ctrlsim_action_logits, ego_ctrlsim_valid = (
                    collect_ego_ctrlsim_action_logits(infos)
                )

            agent.insert(
                obs, recurrent_hidden_states, 
                action, action_log_prob, action_log_dist, 
                value, reward, masks, bad_masks, 
                level_seeds=current_level_seeds,
                cliffhanger_masks=cliffhanger_masks,
                ego_ctrlsim_action_logits=ego_ctrlsim_action_logits,
                ego_ctrlsim_valid=ego_ctrlsim_valid)

        # Add generated env to level store (as a constructive string representation)
        if is_env and plr_runtime_enabled and not level_replay:
            self._update_plr_with_current_unseen_levels()

        if track_nocturne_enhanced_regret:
            if bool(getattr(args, "include_truncated_rtg_gap", False)):
                for i, records in enumerate(active_rtg_records_by_process):
                    if records:
                        seed = active_seed_by_process[i]
                        append_nocturne_rtg_segment(
                            seed=seed,
                            records=records,
                            delta_rtg_segments_by_seed=delta_rtg_segments_by_seed,
                            rtg_gap_method=rtg_gap_method,
                            regret_component_weights=regret_component_weights,
                        )
            valid_rtg_segment_count_by_seed = {
                int(seed): len(delta_rtg_segments_by_seed.get(seed, ()))
                for seed in (
                    set(attempt_count_by_seed.keys())
                    | set(delta_rtg_segments_by_seed.keys())
                )
            }
            rollout_info.update({
                "attempt_count_by_seed": dict(attempt_count_by_seed),
                "success_count_by_seed": dict(success_count_by_seed),
                "delta_rtg_segments_by_seed": {
                    int(seed): list(segments)
                    for seed, segments in delta_rtg_segments_by_seed.items()
                },
                "valid_rtg_segment_count_by_seed": valid_rtg_segment_count_by_seed,
            })

        rollout_info.update(self._get_rollout_return_stats(rollout_returns))
        if not is_env:
            rollout_info["rollout_done_count_by_process"] = (
                rollout_done_count_by_process.tolist()
            )
            rollout_info["rollout_collision_done_count_by_process"] = (
                rollout_collision_done_count_by_process.tolist()
            )
            rollout_info["rollout_offroad_done_count_by_process"] = (
                rollout_offroad_done_count_by_process.tolist()
            )
        if is_nocturne_rollout:
            first_done_infos = [
                first_done_info_by_process[i]
                for i in sorted(first_done_info_by_process.keys())
            ]
            rollout_info['nocturne_first_done_infos'] = first_done_infos
            rollout_info['nocturne_first_done_info_by_process'] = first_done_info_by_process
        if self.use_accel_paired:
            rollout_info['actor_seeds'] = actor_seeds

        # Update non-env agent if required
        if not is_env and update: 
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            agent.storage.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda)

            # Compute batched value loss if using value_l1-maximizing adversary
            if self.requires_batched_vloss:
                # Don't clip value loss reward if env adversary normalizes returns
                clipped = not args.adv_use_popart and not args.adv_normalize_returns
                batched_value_loss = self._get_batched_value_loss(
                    agent, clipped=clipped, batched=True)
                rollout_info.update({'batched_value_loss': batched_value_loss})

            # Update level sampler and remove any ejected seeds from level store
            if not update_agent_separately:
                enhanced_regret_external_scores = None
                enhanced_regret_external_score_metrics = {}
                should_compute_enhanced_regret_scores = (
                    track_nocturne_enhanced_regret
                )
                use_enhanced_regret_sampler_scores = (
                    should_compute_enhanced_regret_scores
                    and plr_runtime_enabled
                    and level_sampler
                    and update_level_sampler
                )
                if should_compute_enhanced_regret_scores:
                    (
                        running_mean_base_regret,
                        running_mean_delta_rtg,
                    ) = self._get_enhanced_regret_running_means()
                    (
                        enhanced_regret_external_scores,
                        enhanced_regret_external_score_metrics,
                    ) = compute_nocturne_enhanced_regret_scores(
                        agent.storage,
                        rollout_info,
                        running_mean_base_regret=running_mean_base_regret,
                        running_mean_delta_rtg=running_mean_delta_rtg,
                        use_solvable_rate=bool(
                            getattr(args, "regret_enhancement_use_solvable_rate", True)
                        ),
                        use_ctrlsim_rtg_gap=bool(
                            getattr(args, "regret_enhancement_use_ctrlsim_rtg_gap", True)
                        ),
                    )
                    rollout_info.update(enhanced_regret_external_score_metrics)
                    self._update_enhanced_regret_running_means(
                        base_regret_by_seed=enhanced_regret_external_score_metrics.get(
                            "base_regret_by_seed",
                            {},
                        ),
                        delta_rtg_by_seed=enhanced_regret_external_score_metrics.get(
                            "delta_rtg_by_seed",
                            {},
                        ),
                    )

                if sample_only:
                    rollout_info.update(
                        self._finalize_sample_only_update(
                            agent=agent,
                            level_sampler=level_sampler,
                            update_level_sampler=update_level_sampler,
                            external_scores=enhanced_regret_external_scores,
                            external_scores_apply_to_partial=(
                                enhanced_regret_external_scores is not None
                            ),
                        )
                    )
                else:
                    if plr_runtime_enabled and level_sampler and update_level_sampler:
                        level_sampler.update_with_rollouts(
                            agent.storage,
                            external_scores=enhanced_regret_external_scores,
                            external_scores_apply_to_partial=(
                                enhanced_regret_external_scores is not None
                            ),
                        )

                    value_loss, action_loss, dist_entropy, info = agent.update(
                        discard_grad=discard_grad,
                        kl_dict=kl_dict,
                        current_update=self.num_updates,
                        total_updates=self.total_updates,
                    )

                    if plr_runtime_enabled and level_sampler and update_level_sampler:
                        level_sampler.after_update()
                    
                    if 'kl_loss' in info.keys():
                        kl_loss = info.pop('kl_loss')
                        rollout_info.update({'kl_loss': kl_loss})
                    if 'ego_ctrlsim_kl_loss' in info.keys():
                        ego_ctrlsim_kl_loss = info.pop('ego_ctrlsim_kl_loss')
                        rollout_info.update({'ego_ctrlsim_kl_loss': ego_ctrlsim_kl_loss})

                    rollout_info.update({
                        'value_loss': value_loss,
                        'action_loss': action_loss,
                        'dist_entropy': dist_entropy,
                        'update_info': info,
                    })

                    # Compute LZ complexity of action trajectories
                    if args.log_action_complexity:
                        rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})

        return rollout_info

    def _finalize_sample_only_update(
        self,
        agent,
        level_sampler=None,
        update_level_sampler=False,
        external_scores=None,
        external_scores_apply_to_partial=False,
    ):
        """Finalize a sample-only rollout without entering PPO update."""
        plr_runtime_enabled = self.plr_runtime_enabled

        if plr_runtime_enabled and level_sampler and update_level_sampler:
            level_sampler.update_with_rollouts(
                agent.storage,
                external_scores=external_scores,
                external_scores_apply_to_partial=external_scores_apply_to_partial,
            )

        agent.storage.after_update()

        if plr_runtime_enabled and level_sampler and update_level_sampler:
            level_sampler.after_update()

        rollout_info = {
            'value_loss': None,
            'action_loss': None,
            'dist_entropy': None,
            'update_info': {},
            'sample_only': True,
        }

        if self.args.log_action_complexity:
            rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})

        return rollout_info
    
    def _update_agent_separately(self, 
                                 agent, 
                                 level_sampler=None, 
                                 update_level_sampler=False,
                                 discard_grad=False,
                                 sample_only=False,
                                 kl_dict=None,
                                 external_scores=None,
                                 external_scores_apply_to_partial=False):
        if sample_only and discard_grad:
            raise ValueError("sample_only and discard_grad are mutually exclusive")

        plr_runtime_enabled = self.plr_runtime_enabled

        if sample_only:
            return self._finalize_sample_only_update(
                agent=agent,
                level_sampler=level_sampler,
                update_level_sampler=update_level_sampler,
                external_scores=external_scores,
                external_scores_apply_to_partial=external_scores_apply_to_partial,
            )

        # Update level sampler and remove any ejected seeds level store
        if plr_runtime_enabled and level_sampler and update_level_sampler:
            level_sampler.update_with_rollouts(
                agent.storage,
                external_scores=external_scores,
                external_scores_apply_to_partial=external_scores_apply_to_partial,
            )

        value_loss, action_loss, dist_entropy, info = agent.update(
            discard_grad=discard_grad,
            kl_dict=kl_dict,
            current_update=self.num_updates,
            total_updates=self.total_updates,
        )

        if plr_runtime_enabled and level_sampler and update_level_sampler:
            level_sampler.after_update()
        
        rollout_info = {
            'value_loss': value_loss,
            'action_loss': action_loss,
            'dist_entropy': dist_entropy,
            'update_info': info,
        }
        
        if 'kl_loss' in info.keys():
            kl_loss = info.pop('kl_loss')
            rollout_info.update({'kl_loss': kl_loss})
        if 'ego_ctrlsim_kl_loss' in info.keys():
            ego_ctrlsim_kl_loss = info.pop('ego_ctrlsim_kl_loss')
            rollout_info.update({'ego_ctrlsim_kl_loss': ego_ctrlsim_kl_loss})

        # Compute LZ complexity of action trajectories
        if self.args.log_action_complexity:
            rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})
        
        return rollout_info

    def _compute_env_return(self, agent_info, adversary_agent_info):
        args = self.args
        if args.ued_algo == 'paired':
            env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
                torch.zeros_like(agent_info['mean_return']))

        elif args.ued_algo == 'flexible_paired':
            env_return = torch.zeros_like(agent_info['max_return'], dtype=torch.float, device=self.device)
            adversary_agent_max_idx = adversary_agent_info['max_return'] > agent_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_info['max_return'][agent_max_idx]
            
            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_info['mean_return'][agent_max_idx]

            env_return = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))

        elif args.ued_algo == 'minimax':
            env_return = -agent_info['max_return']

        else:
            env_return = torch.zeros_like(agent_info['mean_return'])

        if args.adv_normalize_returns:
            self.env_return_rms.update(env_return.flatten().cpu().numpy())
            env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

        if args.adv_clip_reward is not None:
            clip_max_abs = args.adv_clip_reward
            env_return = env_return.clamp(-clip_max_abs, clip_max_abs)
        
        return env_return

    def run(self):
        args = self.args
        plr_runtime_enabled = self.plr_runtime_enabled
        is_nocturne_env = (
            args.env_name.startswith('Nocturne') or args.env_name.startswith('nocturne')
        )

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']
        nocturne_first_done_infos_for_update = []
        nocturne_first_done_info_by_process = {}
        has_merged_first_nocturne_rollout = False

        def merge_nocturne_first_done_infos(rollout_info):
            nonlocal has_merged_first_nocturne_rollout
            if (not is_nocturne_env) or (not isinstance(rollout_info, dict)):
                return
            if has_merged_first_nocturne_rollout:
                return

            done_infos = rollout_info.get('nocturne_first_done_infos')
            if done_infos:
                nocturne_first_done_infos_for_update.extend(done_infos)

            first_by_process = rollout_info.get('nocturne_first_done_info_by_process')
            if isinstance(first_by_process, dict):
                for process_idx, info in first_by_process.items():
                    if info is None:
                        continue
                    nocturne_first_done_info_by_process[int(process_idx)] = info
            has_merged_first_nocturne_rollout = True

        level_replay = False
        if plr_runtime_enabled and self.is_training:
            level_replay = self._sample_replay_decision()

        # Use sample-only updates when scoring newly sampled levels without PPO.
        student_discard_grad = False
        student_sample_only = False
        no_exploratory_grad_updates = \
            vars(args).get('no_exploratory_grad_updates', False)
        if plr_runtime_enabled and (not level_replay) and no_exploratory_grad_updates:
            student_sample_only = True

        if self.is_training and not student_discard_grad and not student_sample_only:
            self.student_grad_updates += 1

        # Generate a batch of adversarial environments
        env_info = self.agent_rollout(
            agent=adversary_env, 
            num_steps=self.adversary_env_rollout_steps, 
            update=False,
            is_env=True,
            level_replay=level_replay,
            level_sampler=self._get_level_sampler('agent')[0],
            update_level_sampler=False)

        # Run agent episodes
        level_sampler, is_updateable = self._get_level_sampler('agent')
        
        kl_dict_agent = None
        if self.use_accel_paired:
            kl_dict_agent = None
        elif self.is_training and self.args.use_behavioural_cloning:
            if ((self.student_grad_updates) % self.args.kl_update_step == 0):
                kl_dict_agent = {}
                adversary_agent.eval()
                kl_dict_agent['antagonist_model'] = adversary_agent.algo.actor_critic
                
        agent_info = self.agent_rollout(
            agent=agent, 
            num_steps=self.agent_rollout_steps,
            update=self.is_training,
            level_replay=level_replay,
            level_sampler=level_sampler,
            update_level_sampler=is_updateable,
            discard_grad=student_discard_grad,
            sample_only=student_sample_only,
            kl_dict=kl_dict_agent,
            update_agent_separately=self.use_accel_paired)
        merge_nocturne_first_done_infos(agent_info)
        
        if kl_dict_agent is not None:
            adversary_agent.train()

        # Use a separate PLR curriculum for the antagonist
        if level_replay and self.is_paired and (args.protagonist_plr == args.antagonist_plr):
            self.agent_rollout(
                agent=adversary_env, 
                num_steps=self.adversary_env_rollout_steps, 
                update=False,
                is_env=True,
                level_replay=level_replay,
                level_sampler=self._get_level_sampler('adversary_agent')[0],
                update_level_sampler=False)

        adversary_agent_info = defaultdict(float)
        if self.is_paired:
            # Run adversary agent episodes
            level_sampler, is_updateable = self._get_level_sampler('adversary_agent')
            
            kl_dict_adv_agent = None
            if not self.args.use_kl_only_agent:
                if self.is_training and self.args.use_behavioural_cloning:
                    if ((self.student_grad_updates) % self.args.kl_update_step == 0):
                        kl_dict_adv_agent = {}
                        agent.eval()
                        kl_dict_adv_agent['antagonist_model'] = agent.algo.actor_critic
                        
            adversary_agent_info = self.agent_rollout(
                agent=adversary_agent, 
                num_steps=self.agent_rollout_steps, 
                update=self.is_training,
                level_replay=level_replay,
                level_sampler=level_sampler,
                update_level_sampler=is_updateable,
                discard_grad=student_discard_grad,
                sample_only=student_sample_only,
                kl_dict=kl_dict_adv_agent)
            merge_nocturne_first_done_infos(adversary_agent_info)
            
            if kl_dict_adv_agent is not None:
                agent.train()
                
        elif self.use_accel_paired:
            
            adversary_agent_info = self.agent_rollout(
                agent=adversary_agent, 
                num_steps=self.agent_rollout_steps, 
                update=self.is_training,
                level_replay=False,
                level_sampler=None,
                update_level_sampler=False,
                discard_grad=student_discard_grad,
                sample_only=student_sample_only,
                kl_dict=None,
                update_agent_separately=self.use_accel_paired
            )
            merge_nocturne_first_done_infos(adversary_agent_info)
            
            # calculate PAIRED regret estimate
            external_scores = self._calculate_paired_regret_scores(agent_info, adversary_agent_info, type=args.accel_paired_score_function)
            
            # update agent and its level sampler
            level_sampler, is_updateable = self._get_level_sampler('agent')
            
            kl_dict_agent = None
            if self.is_training and self.args.use_behavioural_cloning:
                if ((self.student_grad_updates) % self.args.kl_update_step == 0):
                    kl_dict_agent = {}
                    adversary_agent.eval()
                    kl_dict_agent['antagonist_model'] = adversary_agent.algo.actor_critic
            
            agent_update_rollout_info = self._update_agent_separately(agent, 
                                 level_sampler=level_sampler, 
                                 update_level_sampler=is_updateable,
                                 discard_grad=student_discard_grad,
                                 sample_only=student_sample_only,
                                 kl_dict=kl_dict_agent,
                                 external_scores=external_scores)
            
            if kl_dict_agent is not None:
                adversary_agent.train()
            
            agent_info.update(agent_update_rollout_info)
            
            # update antagonist agent too
            kl_dict_adv_agent = None
            if not self.args.use_kl_only_agent:
                if self.is_training and self.args.use_behavioural_cloning:
                    if ((self.student_grad_updates) % self.args.kl_update_step == 0):
                        kl_dict_adv_agent = {}
                        agent.eval()
                        kl_dict_adv_agent['antagonist_model'] = agent.algo.actor_critic
            
            adversary_agent_update_rollout_info = self._update_agent_separately(adversary_agent,
                                 level_sampler=level_sampler, 
                                 update_level_sampler=is_updateable,
                                 discard_grad=student_discard_grad,
                                 sample_only=student_sample_only,
                                 kl_dict=kl_dict_adv_agent,
                                 external_scores=external_scores)
            
            if kl_dict_adv_agent is not None:
                agent.train()
                
            adversary_agent_info.update(adversary_agent_update_rollout_info)

        # Sample whether the decision to edit levels
        edit_level = self._should_edit_level() and level_replay

        if level_replay:
            sampled_level_info = {
                'level_replay': True,
                'num_edits': [len(self.level_store.seed2parent[x])+1 for x in env_info],
            }
        else:
            sampled_level_info = {
                'level_replay': False,
                'num_edits': [0 for _ in range(args.num_processes)]
            }

        # ==== This part performs ACCEL ====
        # If editing, mutate levels just replayed by PLR
        if level_replay and edit_level:
            # Choose base levels for mutation
            if self.base_levels == 'batch':
                fixed_seeds = env_info
            elif self.base_levels == 'easy':
                if self.use_accel_paired:
                    # paired signed regret score
                    regret_score = self._calculate_paired_regret_scores(agent_info, adversary_agent_info, type=args.accel_paired_score_function)
                if args.num_processes >= 4:
                    # take top 4
                    if self.use_accel_paired:
                        easy = list(np.argsort((regret_score.detach().cpu().numpy()).flatten())[:4])
                    else:
                        easy = list(np.argsort((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()).flatten())[:4])
                    fixed_seeds = [env_info[x.item()] for x in easy] * int(args.num_processes/4)
                else:
                    # take top 1
                    if self.use_accel_paired:
                        easy = np.argmax((regret_score.detach().cpu().numpy()).flatten())
                    else:
                        easy = np.argmax((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()).flatten())
                    fixed_seeds = [env_info[easy]] * args.num_processes

            # Always edit from the non-PLR base level (root ancestor), not from
            # already edited replay descendants.
            base_fixed_seeds = [self._resolve_non_plr_base_seed(s) for s in fixed_seeds]
            if all(s is not None for s in base_fixed_seeds):
                fixed_seeds = base_fixed_seeds
                level_sampler, is_updateable = self._get_level_sampler('agent')

                # Edit selected levels
                self.agent_rollout(
                    agent=None,
                    num_steps=None,
                    is_env=True,
                    edit_level=True,
                    num_edits=args.num_edits,
                    fixed_seeds=fixed_seeds)

                self.total_num_edits += 1
                sampled_level_info['num_edits'] = [x+1 for x in sampled_level_info['num_edits']]

                # Evaluate edited levels
                agent_info_edited_level = self.agent_rollout(
                    agent=agent,
                    num_steps=self.agent_rollout_steps,
                    update=self.is_training,
                    level_replay=False,
                    level_sampler=level_sampler,
                    update_level_sampler=is_updateable,
                    update_agent_separately=self.use_accel_paired,
                    sample_only=True)
                merge_nocturne_first_done_infos(agent_info_edited_level)
                
                if self.use_accel_paired:
                    adversary_agent_info_edited_level = self.agent_rollout(
                        agent=adversary_agent,
                        num_steps=self.agent_rollout_steps,
                        update=self.is_training,
                        level_replay=False,
                        level_sampler=None,
                        update_level_sampler=False,
                        update_agent_separately=self.use_accel_paired,
                        discard_grad=True)
                    merge_nocturne_first_done_infos(adversary_agent_info_edited_level)
                    
                    external_scores = self._calculate_paired_regret_scores(agent_info_edited_level, adversary_agent_info_edited_level, type=args.accel_paired_score_function)
                    
                    # update agent level sampler
                    _ = self._update_agent_separately(agent, 
                                        level_sampler=level_sampler, 
                                        update_level_sampler=is_updateable,
                                        discard_grad=True,
                                        kl_dict=None,
                                        external_scores=external_scores)
                    
                    
                    # update antagonist agent too
                    _ = self._update_agent_separately(adversary_agent,
                                        level_sampler=level_sampler, 
                                        update_level_sampler=is_updateable,
                                        discard_grad=True,
                                        kl_dict=None,
                                        external_scores=external_scores)
        # ==== ACCEL end ====

        if plr_runtime_enabled:
            self._reconcile_level_store_and_samplers()
            if self.use_editor:
                self.weighted_num_edits = self._get_weighted_num_edits()

        # Update adversary agent final return
        env_return = self._compute_env_return(agent_info, adversary_agent_info)

        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy,
                'update_info': info
            })

        if self.is_training:
            self.num_updates += 1

        # === LOGGING ===
        # Only update env-related stats when run generates new envs (not level replay)
        log_replay_complexity = level_replay and args.log_replay_complexity
        per_process_stats = []
        tb_per_process_stats = []
        nocturne_infos = None
        nocturne_process_infos = None
        if is_nocturne_env:
            tilting_mode = getattr(args, 'tilting_mode', 'per_vehicle')
            nocturne_infos = list(nocturne_first_done_infos_for_update)
            nocturne_process_infos = []
            for process_idx in range(args.num_processes):
                process_info = nocturne_first_done_info_by_process.get(process_idx)
                nocturne_process_infos.append(dict(process_info) if process_info else {})
            tb_per_process_stats = build_nocturne_process_stats(
                infos=nocturne_process_infos,
                tilting_mode=tilting_mode,
                log_replay_complexity=False,
            )

        if (not level_replay) or log_replay_complexity:

            stats = self._get_env_stats(agent_info, adversary_agent_info, 
                log_replay_complexity=log_replay_complexity,
                nocturne_infos=nocturne_infos)
            if is_nocturne_env:
                per_process_stats = build_nocturne_process_stats(
                    infos=nocturne_process_infos,
                    tilting_mode=tilting_mode,
                    log_replay_complexity=log_replay_complexity)
            stats.update({
                'mean_env_return': env_return.mean().item(),
                'adversary_env_pg_loss': adversary_env_info['action_loss'],
                'adversary_env_value_loss': adversary_env_info['value_loss'],
                'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
            })
            if args.use_plr:
                self.latest_env_stats.update(stats) # Log latest UED curriculum stats instead of PLR env stats
                if is_nocturne_env:
                    self.latest_env_process_stats = [s.copy() for s in per_process_stats]
        else:
            stats = self.latest_env_stats.copy()
            if is_nocturne_env:
                per_process_stats = [s.copy() for s in self.latest_env_process_stats]
                if not per_process_stats:
                    per_process_stats = build_nocturne_process_stats(
                        venv=self.venv,
                        tilting_mode=tilting_mode,
                    )

        rollout_done_count_by_process = agent_info.get("rollout_done_count_by_process")
        if rollout_done_count_by_process is not None:
            rollout_done_count_by_process = list(rollout_done_count_by_process)
            if is_nocturne_env:
                for process_idx, done_count in enumerate(rollout_done_count_by_process):
                    if process_idx < len(per_process_stats):
                        per_process_stats[process_idx]["rollout_done_count"] = int(done_count)
                    if process_idx < len(tb_per_process_stats):
                        tb_per_process_stats[process_idx]["rollout_done_count"] = int(done_count)
            if len(rollout_done_count_by_process) > 0:
                stats["avg_rollout_done_count"] = float(
                    np.mean(rollout_done_count_by_process)
                )
        rollout_collision_done_count_by_process = agent_info.get(
            "rollout_collision_done_count_by_process"
        )
        if rollout_collision_done_count_by_process is not None:
            rollout_collision_done_count_by_process = list(
                rollout_collision_done_count_by_process
            )
            if is_nocturne_env:
                for process_idx, done_count in enumerate(
                    rollout_collision_done_count_by_process
                ):
                    if process_idx < len(per_process_stats):
                        per_process_stats[process_idx][
                            "rollout_collision_done_count"
                        ] = int(done_count)
                    if process_idx < len(tb_per_process_stats):
                        tb_per_process_stats[process_idx][
                            "rollout_collision_done_count"
                        ] = int(done_count)
            if len(rollout_collision_done_count_by_process) > 0:
                stats["avg_rollout_collision_done_count"] = float(
                    np.mean(rollout_collision_done_count_by_process)
                )
        rollout_offroad_done_count_by_process = agent_info.get(
            "rollout_offroad_done_count_by_process"
        )
        if rollout_offroad_done_count_by_process is not None:
            rollout_offroad_done_count_by_process = list(
                rollout_offroad_done_count_by_process
            )
            if is_nocturne_env:
                for process_idx, done_count in enumerate(
                    rollout_offroad_done_count_by_process
                ):
                    if process_idx < len(per_process_stats):
                        per_process_stats[process_idx][
                            "rollout_offroad_done_count"
                        ] = int(done_count)
                    if process_idx < len(tb_per_process_stats):
                        tb_per_process_stats[process_idx][
                            "rollout_offroad_done_count"
                        ] = int(done_count)
            if len(rollout_offroad_done_count_by_process) > 0:
                stats["avg_rollout_offroad_done_count"] = float(
                    np.mean(rollout_offroad_done_count_by_process)
                )

        # Log PLR buffer stats
        if args.use_plr and args.log_plr_buffer_stats:
            stats.update(self._get_plr_buffer_stats())

        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired or self.use_accel_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        self.sampled_level_info = sampled_level_info

        stats.update({
            'steps': (self.num_updates + self.total_num_edits) * args.num_processes * args.num_steps,
            'total_episodes': self.total_episodes_collected,
            'total_seeds': self.total_seeds_collected,
            'total_student_grad_updates': self.student_grad_updates,
            'level_replay': level_replay,

            'mean_agent_return': mean_agent_return,
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],

            'mean_adversary_agent_return': mean_adversary_agent_return,
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
            'ego_ctrlsim_kl_loss': agent_info.get('ego_ctrlsim_kl_loss', None),
        })
        enhanced_regret_metric_keys = (
            'base_regret',
            'solvable_rate',
            'learnability',
            'delta_rtg',
            'enhanced_regret_score',
        )
        stats.update({
            key: value
            for key in enhanced_regret_metric_keys
            if (value := agent_info.get(key)) is not None
        })
        if (
            is_nocturne_env
            and self.external_teacher is not None
            and bool(getattr(args, "use_policy_reweighting", False))
        ):
            stats.update(self.external_teacher.consume_policy_reweighting_update_stats())

        if args.log_grad_norm:
            def _get_mean_grad_norm(rollout_info):
                """Return the logged grad norm mean when available."""
                if rollout_info.get('sample_only'):
                    return None

                update_info = rollout_info.get('update_info', {})
                grad_norms = update_info.get('grad_norms')
                if not grad_norms:
                    return None

                return np.mean(grad_norms)

            agent_grad_norm = _get_mean_grad_norm(agent_info)
            adversary_grad_norm = None
            adversary_env_grad_norm = None
            if self.is_paired:
                adversary_grad_norm = _get_mean_grad_norm(adversary_agent_info)
            if self.is_training_env:
                adversary_env_grad_norm = _get_mean_grad_norm(adversary_env_info)
            stats.update({
                'agent_grad_norm': agent_grad_norm,
                'adversary_grad_norm': adversary_grad_norm,
                'adversary_env_grad_norm': adversary_env_grad_norm
            })

        if args.log_action_complexity:
            stats.update({
                'agent_action_complexity': agent_info['action_complexity'],
                'adversary_action_complexity': adversary_agent_info['action_complexity']  
            }) 

        if per_process_stats:
            stats['_per_process_stats'] = per_process_stats
        if tb_per_process_stats:
            stats['_tb_per_process_stats'] = tb_per_process_stats

        return stats
