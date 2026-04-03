# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import HumanOutputFormat
from tqdm import tqdm

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from envs.registration import make as gym_make
# from envs.multigrid.maze import *
# from envs.multigrid.crossing import *
# from envs.multigrid.fourrooms import *
# from envs.multigrid.mst_maze import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.nocturne_ctrlsim import *  # Nocturne + CtRL-Sim  env
from envs.nocturne_ctrlsim.services.runtime import split_prepared_pack_batch
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv, \
	MultiGridFullyObsWrapper, VecFrameStack, CarRacingWrapper
from util import DotDict, str2bool, make_agent, create_parallel_env, is_discrete_actions, save_images
from util.eval_helper import (
	build_eval_csv_headers,
	build_eval_csv_row,
	set_eval_worker_seeds,
)
from arguments import parser
from util import ignore_warning
ignore_warning.configure_subprocess_env()

"""
Example usage:

python -m eval \
--env_name=MultiGrid-SixteenRooms-v0 \
--base_path="~/logs/dcd/latest" \
--verbose
"""
def parse_args():
	parser = argparse.ArgumentParser(description='Eval')

	parser.add_argument(
		'--base_path',
		type=str,
		default='~/logs/dcd',
		help='Directory containing model checkpoint and meta.json.')
	parser.add_argument(
		'--xpid',
		type=str,
		default='latest',
		help='Deprecated. Ignored by eval.py.')
	parser.add_argument(
		'--prefix',
		type=str,
		default=None,
		help='Deprecated. Ignored by eval.py.'
	)
	parser.add_argument(
		'--env_names',
		type=str,
		default='MultiGrid-Labyrinth-v0',
		help='CSV string of evaluation environments.')
	parser.add_argument(
		'--result_path',
		type=str,
		default='.',
		help='Output directory for evaluation results, relative to base_path.')
	parser.add_argument(
		'--benchmark',
		type=str,
		default=None,
		choices=['maze', 'f1', 'bipedal', 'poetrose'],
		help="Name of benchmark for evaluation.")
	parser.add_argument(
		'--accumulator',
		type=str,
		default=None,
		help="Function for accumulating across multiple evaluation runs.")
	parser.add_argument(
		'--singleton_env',
		type=str2bool, nargs='?', const=True, default=False,
		help="When using a fixed env, whether the same environment should also be reused across workers.")
	parser.add_argument(
		'--seed', 
		type=int, 
		default=1, 
		help='Random seed.')
	parser.add_argument(
		'--max_seeds', 
		type=int, 
		default=None, 
		help='Maximum number of matched experiment IDs to evaluate.')
	parser.add_argument(
		'--num_processes',
		type=int,
		default=1,
		help='Number of CPU processes to use.')
	parser.add_argument(
		'--max_num_processes',
		type=int,
		default=10,
		help='Maximum number of CPU processes to use.')
	parser.add_argument(
		'--num_episodes',
		type=int,
		default=100,
		help='Number of evaluation episodes per xpid per environment.')
	parser.add_argument(
		'--tilt_range',
		type=int,
		nargs=2,
		default=[-25, 25],
		metavar=('MIN', 'MAX'),
		help='Absolute tilt range for Nocturne, formatted as: MIN MAX (e.g., -25 25).')
	parser.add_argument(
		'--opponent_vehicle_number',
		type=int,
		default=7,
		help='Number of opponent vehicles used for per-vehicle tilting (K).')
	parser.add_argument(
		'--model_tar',
		type=str,
		default='model',
		help='Name of .tar to evaluate.')
	parser.add_argument(
		'--model_name',
		type=str,
		default='agent',
		choices=['agent', 'adversary_agent'],
		help='Which agent to evaluate.')
	parser.add_argument(
		'--deterministic',
		type=str2bool, nargs='?', const=True, default=False,
		help="Evaluate policy greedily.")
	parser.add_argument(
		'--verbose',
		type=str2bool, nargs='?', const=True, default=False,
		help="Show logging messages in stdout")
	parser.add_argument(
		'--render',
		type=str2bool, nargs='?', const=True, default=False,
		help="Render environment in first evaluation process to screen.")
	parser.add_argument(
		'--record_video',
		type=str2bool, nargs='?', const=True, default=False,
		help="Record video of first environment evaluation process.")
	parser.add_argument(
		'--inference_precision',
		type=str,
		choices=['fp32', 'amp_fp16', 'amp_bf16'],
		default='fp32',
		help='Inference precision for batched ExternalTeacher inference.')

	return parser.parse_args()


class Evaluator(object):
	def __init__(self, 
		env_names, 
		num_processes, 
		num_episodes=10, 
		record_video=False, 
		device='cuda',
		eval_screenshot=False,
		eval_screenshot_dir=None,
		**kwargs):
		self.kwargs = kwargs # kwargs for env wrappers
		self._init_parallel_envs(
			env_names, num_processes, device=device, record_video=record_video, **kwargs)
		self.num_episodes = num_episodes
		self.eval_screenshot = eval_screenshot
		self.eval_screenshot_dir = eval_screenshot_dir
		if 'Bipedal' in env_names[0]:
			self.solved_threshold = 230
		else:
			self.solved_threshold = 0

	def get_stats_keys(self):
		keys = []
		for env_name in self.env_names:
			keys += [f'solved_rate:{env_name}', f'test_returns:{env_name}']
		return keys

	@staticmethod
	def _make_env(env_name, record_video=False, process_idx=None, **kwargs):
		is_nocturne = env_name.startswith('Nocturne')
		
		if env_name in ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']:
			env = gym.make(env_name)
		elif is_nocturne:
			# make Nocturne env
			from envs.nocturne_ctrlsim import NocturneCtrlSimAdversarial
			allowed_nocturne_keys = {
					'scenario_index_path',
					'opponent_checkpoint',
					'scenario_data_dir',
					'preprocess_dir',
					'vehicle_map_path',
					'max_episode_steps',
					'done_on_position_reached_only',
					'device',
					'student_num_neighbors',
					'student_top_k_road',
					'student_accel_discretization',
					'student_steer_discretization',
					'tilting_mode',
					'tilt_range',
				'show_tilting_params',
				'show_vehicle_ids',
				'show_ego_vehicle_selection',
				'opponent_vehicle_number',
				'action_repeat_frequency',
				'kl_loss_computation_frequency',
				'sparse_inference_action_repeat',
				'veh_veh_collision_rew_multiplier',
				'veh_edge_collision_rew_multiplier',
				'pos_target_achieved_rew_multiplier',
				'use_pos_shaped',
				'use_approaching_goal',
				'use_speed_shaped',
				'use_heading_shaped',
				'use_speed_heading_target',
				'shaped_goal_reward',
				'shaped_goal_distance_scaling',
				'approaching_goal_scaling',
				'use_veh_veh_shaped',
					'use_veh_edge_shaped',
					'max_veh_veh_distance',
					'veh_edge_reward_distance_clip',
					'remove_background_vehicles',
				}
			nocturne_kwargs = {
				k: v for k, v in kwargs.items() if k in allowed_nocturne_keys
			}
			if 'opponent_vehicle_number' in nocturne_kwargs:
				nocturne_kwargs['opponent_k'] = int(nocturne_kwargs.pop('opponent_vehicle_number'))
			# Set default tilting_mode if not provided
			if 'tilting_mode' not in nocturne_kwargs:
				nocturne_kwargs['tilting_mode'] = 'per_vehicle'
			env = NocturneCtrlSimAdversarial(**nocturne_kwargs)
		else:
			env = gym_make(env_name)

		is_multigrid = env_name.startswith('MultiGrid')
		is_car_racing = env_name.startswith('CarRacing')

		if is_car_racing:
			grayscale = kwargs.get('grayscale', False)
			num_action_repeat = kwargs.get('num_action_repeat', 8)
			nstack = kwargs.get('frame_stack', 4)
			crop = kwargs.get('crop_frame', False)

			env = CarRacingWrapper(
				env=env,
				grayscale=grayscale, 
				reward_shaping=False,
				num_action_repeat=num_action_repeat,
				nstack=nstack,
				crop=crop,
				eval_=True)

			if record_video:
				from gym.wrappers.monitor import Monitor
				video_dir = kwargs.get('video_dir', 'videos/')
				env = Monitor(env, video_dir, force=True)
				print('Recording video!', flush=True)

		if is_nocturne:
			# Nocturne env rendering method (not rely on gym render)
			if record_video:
				video_dir = kwargs.get('video_dir', 'videos/')
				# Use a simple wrapper to manage video recording 
				original_env = env
				
				class VideoWrapper:
					def __init__(self):
						self.env = original_env
						self.video_dir = video_dir
						self.episode_count = 0
						self.recording_started = False
						self.process_idx = process_idx
						self.observation_space = original_env.observation_space
						self.action_space = original_env.action_space

					def _episode_name(self):
						if self.process_idx is None:
							return f"episode_{self.episode_count:04d}"
						return f"process{self.process_idx}_episode_{self.episode_count:04d}"

					def _start_if_needed(self):
						if self.recording_started:
							return
						self.env.start_recording(self.video_dir, self._episode_name(), fps=10, dpi=100)
						self.recording_started = True

					def _stop_if_recording(self):
						if not self.recording_started:
							return
						if getattr(self.env, 'recording_video', False):
							self.env.stop_recording(self._episode_name())
						self.episode_count += 1
						self.recording_started = False
					
					def reset(self, **kw):
						# in nocturne env, using reset_random()
						if hasattr(self.env, 'reset_random') and not kw:
							obs = self.env.reset_random()
						else:
							obs = self.env.reset(**kw)
						return obs

					def reset_random(self, **kw):
						obs = self.env.reset_random(**kw)
						return obs

					def reset_agent(self, **kw):
						if kw:
							obs = self.env.reset_agent(**kw)
						else:
							obs = self.env.reset_agent()
						return obs

					def step(self, action):
						self._start_if_needed()
						obs, reward, done, info = self.env.step(action)
						if done:
							self._stop_if_recording()
						return obs, reward, done, info

					def step_prepare(self, action):
						self._start_if_needed()
						return self.env.step_prepare(action)

					def step_complete(self, model_output):
						self._start_if_needed()
						obs, reward, done, info = self.env.step_complete(model_output)
						if done:
							self._stop_if_recording()
						return obs, reward, done, info
					
					def close(self):
						self._stop_if_recording()
						self.env.close()
					
					def __getattr__(self, name):
						return getattr(self.env, name)
				
				env = VideoWrapper()
			return env

		if is_multigrid and kwargs.get('use_global_policy'):
			if MultiGridFullyObsWrapper is None:
				raise ImportError("MultiGrid environment requires gym_minigrid. Install with: pip install gym-minigrid")
			env = MultiGridFullyObsWrapper(env, is_adversarial=False)

		return env

	@staticmethod
	def wrap_venv(venv, env_name, device='cuda'):
		is_multigrid = env_name.startswith('MultiGrid') or env_name.startswith('MiniGrid')
		is_car_racing = env_name.startswith('CarRacing')
		is_bipedal = env_name.startswith('BipedalWalker')
		is_nocturne = env_name.startswith('Nocturne')

		obs_key = None
		scale = None
		if is_multigrid:
			obs_key = 'image'
			scale = 10.0

		# Channels first
		transpose_order = [2,0,1]

		if is_bipedal or is_nocturne:
			transpose_order = None

		venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

		venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
				transpose_order=transpose_order, scale=scale, device=device)

		return venv

	def _init_parallel_envs(self, env_names, num_processes, device=None, record_video=False, **kwargs):
		self.env_names = env_names
		self.num_processes = num_processes
		self.device = device
		self.venv = {env_name:None for env_name in env_names}
		eval_seed = kwargs.get('seed')
		singleton_env = bool(kwargs.get('singleton_env', False))

		for env_name in env_names:
			make_fn = [
				(lambda idx: lambda: Evaluator._make_env(
					env_name,
					record_video,
					process_idx=idx,
					**kwargs,
				))(i)
				for i in range(self.num_processes)
			]
			venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
			venv = Evaluator.wrap_venv(venv, env_name, device=device)
			set_eval_worker_seeds(
				venv=venv,
				seed=eval_seed,
				num_processes=self.num_processes,
				singleton_env=singleton_env,
			)
			self.venv[env_name] = venv

		self.is_discrete_actions = is_discrete_actions(self.venv[env_names[0]])

	def close(self):
		for _, venv in self.venv.items():
			venv.close()

	def save_eval_screenshot(
		self,
		out_dir: str,
		update_idx: int,
		batch_size: int = 1,
		prefix: str = "eval",
	) -> None:
		"""Save a screenshot grid from evaluation environments."""
		if batch_size <= 0:
			return
		os.makedirs(out_dir, exist_ok=True)
		for env_name in self.env_names:
			venv = self.venv[env_name]
			images = venv.get_images()
			if not images:
				continue
			images = images[:batch_size]
			path = os.path.join(out_dir, f"{prefix}-{env_name}-update{update_idx}.png")
			save_images(images, path, normalize=True, channels_first=False)

	def evaluate(self, 
		agent, 
		deterministic=False, 
		show_progress=False,
		render=False,
		accumulator='mean',
		return_episode_returns=False,
		external_teacher=None):

		# Evaluate agent for N episodes
		venv = self.venv
		env_returns = {}
		env_solved_episodes = {}
		
		for env_name, venv in self.venv.items():
			returns = []
			solved_episodes = 0
			episode_counts = [0 for _ in range(self.num_processes)]
			if self.eval_screenshot and self.eval_screenshot_dir:
				os.makedirs(self.eval_screenshot_dir, exist_ok=True)

			def _save_episode_images(start_indices):
				if not self.eval_screenshot or not self.eval_screenshot_dir:
					return
				images = venv.get_images()
				if not images:
					return
				for idx in start_indices:
					if episode_counts[idx] >= self.num_episodes:
						continue
					if idx >= len(images):
						continue
					name = f"eval_process{idx}_episode_{episode_counts[idx]:02d}.png"
					path = os.path.join(self.eval_screenshot_dir, name)
					save_images([images[idx]], path, normalize=True, channels_first=False)
					episode_counts[idx] += 1
			# PAIRED/Minimax returns adversary obs, level info, reset 
			# DR/PLR envs return agent obs, vehicle and map info, reset_random
			# reset() returns adversary observation (dict), but we need agent observation (array)
			# For Nocturne-CtrlSim-Adversarial env, use reset_random during evaluation
			if env_name.startswith('Nocturne') and hasattr(venv, 'reset_random'):
				obs = venv.reset_random()
			else:
				obs = venv.reset()
			_save_episode_images(range(self.num_processes))
			recurrent_hidden_states = torch.zeros(
				self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
			if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
				recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
			masks = torch.ones(self.num_processes, 1, device=self.device)

			pbar = None
			if show_progress:
				pbar = tqdm(total=self.num_episodes)

			while len(returns) < self.num_episodes:
				# Sample actions
				with torch.no_grad():
					_, action, _, recurrent_hidden_states = agent.act(
						obs, recurrent_hidden_states, masks, deterministic=deterministic)

					# Observe reward and next obs
					action = action.cpu().numpy()
					if env_name.startswith('Nocturne'):
						action = agent.process_action(action)
						if external_teacher is None:
							raise RuntimeError("Nocturne evaluation requires an ExternalTeacher.")
						per_env_prepared = venv.step_prepare(action)
						opponent_prepared, _ = split_prepared_pack_batch(per_env_prepared)
						model_outputs = external_teacher.run_batched_forward(opponent_prepared)
						obs, reward, done, infos = venv.step_complete(model_outputs, reset_random=True)
					elif not self.is_discrete_actions:
						action = agent.process_action(action)
						obs, reward, done, infos = venv.step(action)
					else:
						obs, reward, done, infos = venv.step(action)

				masks = torch.tensor(
					[[0.0] if done_ else [1.0] for done_ in done],
					dtype=torch.float32,
					device=self.device)

				for i, info in enumerate(infos):
					if 'episode' in info.keys():
						returns.append(info['episode']['r'])
						if returns[-1] > self.solved_threshold:
							solved_episodes += 1
						if pbar:
							pbar.update(1)
						if show_progress and env_name.startswith('Nocturne'):
							pass

						# zero hidden states
						if agent.is_recurrent:
							recurrent_hidden_states[0][i].zero_()
							recurrent_hidden_states[1][i].zero_()

						if len(returns) >= self.num_episodes:
							break
				if self.eval_screenshot and len(returns) < self.num_episodes:
					done_indices = [i for i, d in enumerate(done) if d]
					if done_indices:
						_save_episode_images(done_indices)

				if render:
					venv.render_to_screen()

			if pbar:
				pbar.close()
	
			env_returns[env_name] = returns
			env_solved_episodes[env_name] = solved_episodes

		stats = {}
		for env_name in self.env_names:
			if accumulator == 'mean':
				stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]/self.num_episodes

			if accumulator == 'mean':
				stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])
			else:
				stats[f"test_returns:{env_name}"] = env_returns[env_name]

		if return_episode_returns:
			return stats, env_returns
		return stats


def _collect_nocturne_required_args(flags, cli_args):
	keys = [
		"scenario_index_path",
		"opponent_checkpoint",
		"opponent_vehicle_number",
		"action_repeat_frequency",
		"kl_loss_computation_frequency",
		"sparse_inference_action_repeat",
		"inference_precision",
		"scenario_data_dir",
		"preprocess_dir",
		"vehicle_map_path",
		"student_accel_discretization",
		"student_steer_discretization",
		"student_num_neighbors",
		"student_top_k_road",
		"tilt_range",
		"use_speed_heading_target",
		"done_on_position_reached_only",
	]
	required = {}
	for key in keys:
		if key in flags:
			required[key] = flags[key]
		elif key in cli_args:
			required[key] = cli_args[key]
	return required


def _get_f1_env_names():
	env_names = [f'CarRacingF1-{name}-v0' for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)]
	env_names.remove('CarRacingF1-LagunaSeca-v0')
	return env_names


def _get_zs_minigrid_env_names():
	env_names = [
		'MultiGrid-SixteenRooms-v0',
		'MultiGrid-SixteenRoomsFewerDoors-v0'
		'MultiGrid-Labyrinth-v0',
		'MultiGrid-Labyrinth2-v0',
		'MultiGrid-Maze-v0',
		'MultiGrid-Maze2-v0',
		"MultiGrid-LargeCorridor-v0",
		"MultiGrid-PerfectMazeMedium-v0",
		"MultiGrid-PerfectMazeLarge-v0",
		"MultiGrid-PerfectMazeXL-v0",
	]
	return env_names


def _get_bipedal_env_names():
	env_names = [
		"BipedalWalker-v3",
		"BipedalWalkerHardcore-v3",
		"BipedalWalker-Med-Stairs-v0",
		"BipedalWalker-Med-PitGap-v0",
		"BipedalWalker-Med-StumpHeight-v0",
		"BipedalWalker-Med-Roughness-v0",
	]
	return env_names


def _get_poet_rose_env_names():
	env_names = [f'BipedalWalker-POET-Rose-{id}-v0' for id in ['1a', '1b', '2a', '2b', '3a', '3b']]
	return env_names


def _resolve_output_dir(base_path, output_path):
	"""Resolve an output directory against the experiment base path."""
	expanded_output_path = os.path.expandvars(os.path.expanduser(output_path))
	if os.path.isabs(expanded_output_path):
		return expanded_output_path
	return os.path.join(base_path, expanded_output_path)


if __name__ == '__main__':
	os.environ["OMP_NUM_THREADS"] = "1"

	display = None
	if sys.platform.startswith('linux'):
		print('Setting up virtual display')

		import pyvirtualdisplay
		display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
		display.start()

	args = DotDict(vars(parse_args()))
	args.num_processes = min(args.num_processes, args.num_episodes)

	# === Determine device ====
	device = 'cuda'

	# === Load checkpoint ===
	# Load meta.json into flags object
	base_path = os.path.expandvars(os.path.expanduser(args.base_path))
	output_base_path = base_path
	result_path = _resolve_output_dir(output_base_path, args.result_path)
	video_dir = output_base_path

	# Set up results management
	os.makedirs(result_path, exist_ok=True)
	if args.record_video:
		os.makedirs(video_dir, exist_ok=True)
	result_fname = f"eval-{args.model_tar}-{args.model_name}"
	result_fpath = os.path.join(result_path, result_fname)
	if os.path.exists(f'{result_fpath}.csv'):
		result_fpath = os.path.join(result_path, f'{result_fname}_redo')
	result_fpath = f'{result_fpath}.csv'

	csvout = open(result_fpath, 'w', newline='')
	csvwriter = csv.writer(csvout)

	env_results = defaultdict(list)

	# Get envs
	if args.benchmark == 'maze':
		env_names = _get_zs_minigrid_env_names()
	elif args.benchmark == 'f1':
		env_names = _get_f1_env_names()
	elif args.benchmark == 'bipedal':
		env_names = _get_bipedal_env_names()
	elif args.benchmark == 'poetrose':
		env_names = _get_poet_rose_env_names()
	else:
		env_names = args.env_names.split(',')

	num_envs = len(env_names)
	if num_envs*args.num_processes > args.max_num_processes:
		chunk_size = args.max_num_processes//args.num_processes
	else:
		chunk_size = num_envs

	num_chunks = int(np.ceil(num_envs/chunk_size))

	if args.record_video:
		num_chunks = 1
		chunk_size = 1
		args.num_processes = 1

	num_seeds = 0
	meta_json_path = os.path.join(base_path, 'meta.json')
	model_tar = f'{args.model_tar}.tar'
	checkpoint_path = os.path.join(base_path, model_tar)

	if os.path.exists(checkpoint_path):
		with open(meta_json_path) as meta_json_file:
			xpid_flags = DotDict(json.load(meta_json_file)['args'])
		xpid_flags_meta = DotDict(dict(xpid_flags))

		nocturne_required = _collect_nocturne_required_args(xpid_flags_meta, args)
		make_fn = [lambda: Evaluator._make_env(env_names[0], **nocturne_required)]
		dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
		dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=device)

		# Load the agent
		agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

		try:
			checkpoint = torch.load(checkpoint_path, map_location='cpu')
		except:
			checkpoint = None

		if checkpoint is not None:
			model_name = args.model_name

			if 'runner_state_dict' in checkpoint:
				agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])
			else:
				agent.algo.actor_critic.load_state_dict(checkpoint)

			num_seeds = 1

			external_teacher = None
			if any(name.startswith("Nocturne") for name in env_names):
				from batch_inference import ExternalTeacher, build_external_teacher_kwargs
				opponent_checkpoint = nocturne_required.get("opponent_checkpoint")
				if opponent_checkpoint is None:
					raise ValueError(
						"Nocturne evaluation requires opponent_checkpoint."
					)
				external_teacher = ExternalTeacher(
					**build_external_teacher_kwargs(
						checkpoint_path=opponent_checkpoint,
						device=device,
						inference_precision=nocturne_required.get("inference_precision", "fp32"),
						config_source=nocturne_required,
					)
				)

			# Evaluate environment batch in increments of chunk size
			for i in range(num_chunks):
				start_idx = i*chunk_size
				env_names_ = env_names[start_idx:start_idx+chunk_size]

				# Evaluate the model
				xpid_flags.update(args)
				xpid_flags.update({"use_skip": False})
				nocturne_required = _collect_nocturne_required_args(xpid_flags_meta, args)

				evaluator = Evaluator(env_names_,
					num_processes=args.num_processes,
					num_episodes=args.num_episodes,
					frame_stack=xpid_flags.frame_stack,
					grayscale=xpid_flags.grayscale,
					use_global_critic=xpid_flags.use_global_critic,
					video_dir=video_dir,
					seed=args.seed,
					singleton_env=args.singleton_env,
					**nocturne_required,
					record_video=args.record_video)

				stats = evaluator.evaluate(agent,
					deterministic=args.deterministic,
					show_progress=args.verbose,
					render=args.render,
					accumulator=args.accumulator,
					external_teacher=external_teacher)

				for k,v in stats.items():
					if args.accumulator:
						env_results[k].append(v)
					else:
						env_results[k] += v

				evaluator.close()
	else:
		raise FileNotFoundError(f'No model path {checkpoint_path}')

	output_results = {}
	for k,_ in stats.items():
		results = env_results[k]
		output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
		q1 = np.percentile(results, 25, method='midpoint')
		q3 = np.percentile(results, 75, method='midpoint')
		median = np.median(results)
		output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
		print(f"{k}: {output_results[k]}")
	key_excluded = {k: () for k in output_results.keys()}
	HumanOutputFormat(sys.stdout).write(output_results, key_excluded=key_excluded, step=0)

	first_metric_values = next(iter(env_results.values()), [])
	csvwriter.writerow(
		build_eval_csv_headers(
			accumulator=args.accumulator,
			value_count=len(first_metric_values),
		)
	)
	for metric_key, values in env_results.items():
		csvwriter.writerow(build_eval_csv_row(metric_key, values))

	if display:
		display.stop()
