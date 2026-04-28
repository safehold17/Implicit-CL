# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.logger import HumanOutputFormat
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from envs.registration import make as gym_make
# from envs.multigrid.maze import *
# from envs.multigrid.crossing import *
# from envs.multigrid.fourrooms import *
# from envs.multigrid.mst_maze import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.nocturne_ctrlsim import *  # Nocturne + CtRL-Sim  env
from envs.nocturne_ctrlsim.core.episode_runtime import split_prepared_pack_batch
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv, \
	MultiGridFullyObsWrapper, CarRacingWrapper
from evaluation.evaluation_common import (
	extract_episode_metrics,
	resolve_csv_output_path,
	start_headless_display,
	stop_headless_display,
	wrap_nocturne_video_env,
	write_episode_metrics_csv,
)
from util import DotDict, str2bool, make_agent, is_discrete_actions, save_images
from util.eval_helper import (
	build_eval_csv_headers,
	build_eval_csv_row,
	set_eval_worker_seeds,
)
from util import ignore_warning
ignore_warning.configure_subprocess_env()

"""
Example usage:

python -m evaluation.eval \
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
		'--tilt_range_min',
		type=int,
		default=-25,
		help='Minimum absolute tilt value for Nocturne.')
	parser.add_argument(
		'--tilt_range_max',
		type=int,
		default=25,
		help='Maximum absolute tilt value for Nocturne.')
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

def load_actor_critic_checkpoint(actor_critic, state_dict: dict[str, Any]) -> None:
	"""Load checkpoint weights into compiled or eager actor_critic modules."""
	if hasattr(actor_critic, "_orig_mod"):
		if any(key.startswith("_orig_mod.") for key in state_dict):
			actor_critic.load_state_dict(state_dict)
		else:
			actor_critic._orig_mod.load_state_dict(state_dict)
		return

	if any(key.startswith("_orig_mod.") for key in state_dict):
		trimmed_state_dict = {
			key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
		}
		actor_critic.load_state_dict(trimmed_state_dict)
		return

	actor_critic.load_state_dict(state_dict)


def _build_nocturne_env_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
	"""Filter shared kwargs down to the Nocturne env constructor."""
	allowed_nocturne_keys = {
		'scenario_index_path',
		'opponent_checkpoint',
		'scenario_data_dir',
		'preprocess_dir',
		'vehicle_map_path',
		'max_episode_steps',
		'done_on_position_reached_only',
		'goal_pos_tolerance',
		'device',
		'student_num_neighbors',
		'student_top_k_road',
		'student_model_type',
		'ctrlsim_student_seq_len',
		'ctrlsim_student_num_neighbors',
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
	nocturne_kwargs = {k: v for k, v in kwargs.items() if k in allowed_nocturne_keys}
	if 'opponent_vehicle_number' in nocturne_kwargs:
		nocturne_kwargs['opponent_k'] = int(nocturne_kwargs.pop('opponent_vehicle_number'))
	nocturne_kwargs.setdefault('tilting_mode', 'per_vehicle')
	return nocturne_kwargs


def _init_recurrent_hidden_states(agent: Any, num_processes: int, device: str) -> Any:
	"""Build the recurrent state container used during evaluation rollout."""
	recurrent_hidden_states = torch.zeros(
		num_processes,
		agent.algo.actor_critic.recurrent_hidden_state_size,
		device=device,
	)
	if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
		recurrent_hidden_states = (
			recurrent_hidden_states,
			torch.zeros_like(recurrent_hidden_states),
		)
	return recurrent_hidden_states


def _build_done_masks(done: list[bool], device: str) -> torch.Tensor:
	"""Convert per-env done flags into the masks expected by the actor."""
	return torch.tensor(
		[[0.0] if done_ else [1.0] for done_ in done],
		dtype=torch.float32,
		device=device,
	)


def _reset_eval_env(venv: Any, env_name: str) -> Any:
	"""Reset an evaluation env, preferring Nocturne random reset."""
	if env_name.startswith('Nocturne') and hasattr(venv, 'reset_random'):
		return venv.reset_random()
	return venv.reset()


def _select_env_names(args: DotDict) -> list[str]:
	"""Resolve the evaluation env list from benchmark flags or explicit names."""
	if args.benchmark == 'maze':
		return _get_zs_minigrid_env_names()
	if args.benchmark == 'f1':
		return _get_f1_env_names()
	if args.benchmark == 'bipedal':
		return _get_bipedal_env_names()
	if args.benchmark == 'poetrose':
		return _get_poet_rose_env_names()
	return args.env_names.split(',')


def _compute_chunking(args: DotDict, num_envs: int) -> tuple[int, int]:
	"""Compute env chunk size and chunk count for evaluation."""
	chunk_size = num_envs
	if num_envs * args.num_processes > args.max_num_processes:
		chunk_size = args.max_num_processes // args.num_processes
	if args.record_video:
		args.num_processes = 1
		return 1, 1
	return chunk_size, int(np.ceil(num_envs / chunk_size))


def _load_eval_runtime(
	base_path: str,
	env_name: str,
	device: str,
	args: DotDict,
) -> tuple[DotDict, DotDict, Any, Any]:
	"""Load flags, build a dummy env, and restore the evaluation agent."""
	meta_json_path = os.path.join(base_path, 'meta.json')
	checkpoint_path = os.path.join(base_path, f'{args.model_tar}.tar')
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f'No model path {checkpoint_path}')

	with open(meta_json_path) as meta_json_file:
		xpid_flags = DotDict(json.load(meta_json_file)['args'])
	xpid_flags_meta = DotDict(dict(xpid_flags))
	nocturne_required = _collect_nocturne_required_args(xpid_flags_meta, args)
	make_fn = [lambda: Evaluator._make_env(env_name, **nocturne_required)]
	dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
	dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_name, device=device)

	agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	model_name = args.model_name
	if 'runner_state_dict' in checkpoint:
		load_actor_critic_checkpoint(
			agent.algo.actor_critic,
			checkpoint['runner_state_dict']['agent_state_dict'][model_name],
		)
	else:
		load_actor_critic_checkpoint(agent.algo.actor_critic, checkpoint)
	return xpid_flags, xpid_flags_meta, agent, dummy_venv


def _build_nocturne_external_teacher(
	env_names: list[str],
	nocturne_required: dict[str, Any],
	device: str,
) -> Any | None:
	"""Build the batched ExternalTeacher when any target env is Nocturne."""
	if not any(name.startswith("Nocturne") for name in env_names):
		return None

	from batch_inference import ExternalTeacher, build_external_teacher_kwargs

	opponent_checkpoint = nocturne_required.get("opponent_checkpoint")
	if opponent_checkpoint is None:
		raise ValueError("Nocturne evaluation requires opponent_checkpoint.")
	return ExternalTeacher(
		**build_external_teacher_kwargs(
			checkpoint_path=opponent_checkpoint,
			device=device,
			inference_precision=nocturne_required.get("inference_precision", "fp32"),
			config_source=nocturne_required,
		)
	)


def _accumulate_eval_stats(
	env_results: defaultdict[str, list[Any]],
	stats: dict[str, Any],
	accumulator: str | None,
) -> None:
	"""Merge one evaluator run's stats into the accumulated result store."""
	for key, values in stats.items():
		if accumulator:
			env_results[key].append(values)
		else:
			env_results[key] += values


def _write_scalar_eval_csv(
	output_path: str,
	env_results: defaultdict[str, list[Any]],
	stats: dict[str, Any],
	accumulator: str | None,
) -> None:
	"""Write aggregated scalar metrics when no per-episode rows are collected."""
	with open(output_path, 'w', newline='') as csvout:
		csvwriter = csv.writer(csvout)
		output_results = {}
		for key in stats:
			results = env_results[key]
			output_results[key] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
			q1 = np.percentile(results, 25, method='midpoint')
			q3 = np.percentile(results, 75, method='midpoint')
			median = np.median(results)
			output_results[f'iq_{key}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
			print(f"{key}: {output_results[key]}")

		key_excluded = {key: () for key in output_results.keys()}
		HumanOutputFormat(sys.stdout).write(
			output_results,
			key_excluded=key_excluded,
			step=0,
		)

		first_metric_values = next(iter(env_results.values()), [])
		csvwriter.writerow(
			build_eval_csv_headers(
				accumulator=accumulator,
				value_count=len(first_metric_values),
			)
		)
		for metric_key, values in env_results.items():
			csvwriter.writerow(build_eval_csv_row(metric_key, values))


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
			env = NocturneCtrlSimAdversarial(**_build_nocturne_env_kwargs(kwargs))
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
				env = wrap_nocturne_video_env(
					env,
					video_dir=video_dir,
					process_idx=process_idx,
				)
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
		return_episode_metrics=False,
		external_teacher=None):

		# Evaluate agent for N episodes
		env_returns = {}
		env_solved_episodes = {}
		env_episode_metrics = {}
		
		for env_name, venv in self.venv.items():
			returns = []
			solved_episodes = 0
			episode_metrics = []
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
			obs = _reset_eval_env(venv, env_name)
			_save_episode_images(range(self.num_processes))
			recurrent_hidden_states = _init_recurrent_hidden_states(
				agent,
				self.num_processes,
				self.device,
			)
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

				masks = _build_done_masks(done, self.device)

				for i, info in enumerate(infos):
					if 'episode' in info.keys():
						returns.append(info['episode']['r'])
						if env_name.startswith('Nocturne'):
							episode_metrics.append(extract_episode_metrics(info))
						if returns[-1] > self.solved_threshold:
							solved_episodes += 1
						if pbar:
							pbar.update(1)

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
			env_episode_metrics[env_name] = episode_metrics

		stats = {}
		for env_name in self.env_names:
			if accumulator == 'mean':
				stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]/self.num_episodes

			if accumulator == 'mean':
				stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])
			else:
				stats[f"test_returns:{env_name}"] = env_returns[env_name]

		if return_episode_returns and return_episode_metrics:
			return stats, env_returns, env_episode_metrics
		if return_episode_metrics:
			return stats, env_episode_metrics
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
		"student_model_type",
		"ctrlsim_student_seq_len",
		"ctrlsim_student_num_neighbors",
		"use_speed_heading_target",
		"done_on_position_reached_only",
		"goal_pos_tolerance",
	]
	required = {}
	for key in keys:
		if key in flags:
			required[key] = flags[key]
		elif key in cli_args:
			required[key] = cli_args[key]
	if "tilt_range" in flags:
		required["tilt_range"] = flags["tilt_range"]
	else:
		tilt_range_min = flags.get("tilt_range_min", cli_args.get("tilt_range_min"))
		tilt_range_max = flags.get("tilt_range_max", cli_args.get("tilt_range_max"))
		if tilt_range_min is not None and tilt_range_max is not None:
			required["tilt_range"] = (tilt_range_min, tilt_range_max)
	return required


def _get_f1_env_names():
	env_names = [f'CarRacingF1-{name}-v0' for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)]
	env_names.remove('CarRacingF1-LagunaSeca-v0')
	return env_names


def _get_zs_minigrid_env_names():
	env_names = [
		'MultiGrid-SixteenRooms-v0',
		'MultiGrid-SixteenRoomsFewerDoors-v0',
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


def main() -> None:
	"""Run one evaluation pass and write either per-episode or aggregate CSVs."""
	os.environ["OMP_NUM_THREADS"] = "1"
	display = start_headless_display()
	try:
		args = DotDict(vars(parse_args()))
		args.num_processes = min(args.num_processes, args.num_episodes)
		device = 'cuda'

		base_path = os.path.expandvars(os.path.expanduser(args.base_path))
		result_path = _resolve_output_dir(base_path, args.result_path)
		video_dir = base_path
		os.makedirs(result_path, exist_ok=True)
		if args.record_video:
			os.makedirs(video_dir, exist_ok=True)

		result_fpath = resolve_csv_output_path(
			result_path,
			f"eval-{args.model_tar}-{args.model_name}",
		)

		env_names = _select_env_names(args)
		chunk_size, num_chunks = _compute_chunking(args, len(env_names))
		xpid_flags, xpid_flags_meta, agent, dummy_venv = _load_eval_runtime(
			base_path,
			env_names[0],
			device,
			args,
		)

		try:
			nocturne_required = _collect_nocturne_required_args(xpid_flags_meta, args)
			external_teacher = _build_nocturne_external_teacher(
				env_names,
				nocturne_required,
				device,
			)
			env_results = defaultdict(list)
			episode_metrics_rows = []
			stats = {}

			for i in range(num_chunks):
				start_idx = i * chunk_size
				env_names_ = env_names[start_idx:start_idx + chunk_size]
				collect_episode_metrics = (
					args.accumulator is None
					and all(name.startswith("Nocturne") for name in env_names_)
				)

				xpid_flags.update(args)
				xpid_flags.update({"use_skip": False})
				nocturne_required = _collect_nocturne_required_args(xpid_flags_meta, args)

				evaluator = Evaluator(
					env_names_,
					num_processes=args.num_processes,
					num_episodes=args.num_episodes,
					frame_stack=xpid_flags.frame_stack,
					grayscale=xpid_flags.grayscale,
					use_global_critic=xpid_flags.use_global_critic,
					video_dir=video_dir,
					seed=args.seed,
					singleton_env=args.singleton_env,
					**nocturne_required,
					record_video=args.record_video,
				)
				try:
					eval_output = evaluator.evaluate(
						agent,
						deterministic=args.deterministic,
						show_progress=args.verbose,
						render=args.render,
						accumulator=args.accumulator,
						return_episode_metrics=collect_episode_metrics,
						external_teacher=external_teacher,
					)
				finally:
					evaluator.close()

				if collect_episode_metrics:
					stats, env_episode_metrics = eval_output
					for env_name in env_names_:
						episode_metrics_rows.extend(env_episode_metrics.get(env_name, []))
				else:
					stats = eval_output

				_accumulate_eval_stats(env_results, stats, args.accumulator)
		finally:
			dummy_venv.close()

		if episode_metrics_rows:
			write_episode_metrics_csv(result_fpath, episode_metrics_rows)
		else:
			_write_scalar_eval_csv(
				result_fpath,
				env_results,
				stats,
				args.accumulator,
			)
	finally:
		stop_headless_display(display)


if __name__ == '__main__':
	main()
