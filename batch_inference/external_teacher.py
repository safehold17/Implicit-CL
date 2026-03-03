"""
ExternalTeacher: 主进程跨 env 批量推理引擎。

将所有 env 子进程的 focal batch 合并为扁平批，
在主进程 GPU 上执行两阶段 forward（RTG → action），
然后按 env_idx 拆散回每个 env 的 model_outputs。
"""
import os as _os
import sys as _sys
_CTRLSIM_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), 'third_party', 'ctrl-sim'
)
if _CTRLSIM_PATH not in _sys.path:
    _sys.path.insert(0, _CTRLSIM_PATH)

import warnings
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple

from models.ctrl_sim import CtRLSim
from utils.data import from_numpy, MotionData

from .discretization_utils import (
    get_tilt_logits,
    decode_predicted_rtg,
    decode_predicted_action,
    undiscretize_rtgs,
)
from .ipc_codec import pack_model_outputs, unpack_prepared


class ExternalTeacher:
    """主进程 GPU 批量推理引擎。

    Args:
        checkpoint_path: ctrl-sim checkpoint 路径
        cfg: Hydra 配置（用于提取离散化参数并校验）
        device: 推理设备
        micro_batch: micro-batch 大小（None = 不限制，所有 focal 合并为单次 forward）
        base_seed: 基础随机种子（每个 env 的 generator seed = base_seed + env_idx * 100003）
    """

    def __init__(self, checkpoint_path: str, cfg: Any = None, device: str = 'cuda',
                 micro_batch: Optional[int] = None, base_seed: int = 1):
        self.DEFAULT_MAX_CHUNK_TOKENS = 64512
        self.device = device
        self.micro_batch = micro_batch
        self.base_seed = base_seed
        self._profile_enabled = self._read_env_flag('CTRLSIM_EXTERNAL_TEACHER_PROFILE', default='0')
        try:
            self._profile_every = max(
                1,
                int(_os.getenv('CTRLSIM_EXTERNAL_TEACHER_PROFILE_EVERY', '50')),
            )
        except ValueError:
            self._profile_every = 50
        self._profile_counter = 0

        # 加载模型
        print(f"[ExternalTeacher] Loading CtRL-Sim model from {checkpoint_path}...")
        self.model = CtRLSim.load_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        print("[ExternalTeacher] Model loaded successfully.")

        # 从 checkpoint cfg 中提取离散化参数
        ckpt_cfg = self.model.cfg
        ds = ckpt_cfg.dataset.waymo
        mdl = ckpt_cfg.model

        self.rtg_discretization = ds.rtg_discretization
        self.accel_discretization = ds.accel_discretization
        self.steer_discretization = ds.steer_discretization
        self.min_accel = ds.min_accel
        self.max_accel = ds.max_accel
        self.min_steer = ds.min_steer
        self.max_steer = ds.max_steer
        self.min_rtg_pos = ds.min_rtg_pos
        self.max_rtg_pos = ds.max_rtg_pos
        self.min_rtg_veh = ds.min_rtg_veh
        self.max_rtg_veh = ds.max_rtg_veh
        self.min_rtg_road = ds.min_rtg_road
        self.max_rtg_road = ds.max_rtg_road
        self.num_reward_components = mdl.num_reward_components

        # per-env RNG generators（按需创建）
        self._generators: Dict[int, torch.Generator] = {}
        # collate 阶段 numpy 缓冲复用（按 shape + dtype 缓存）
        self._collate_numpy_buffers: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def batched_forward(self, per_env_prepared: List[Optional[Dict]]) -> List[Optional[Dict]]:
        """跨 env 批量推理。

        Args:
            per_env_prepared: 长度 = num_envs 的列表，每个元素为
                adapter.prepare_step() 的返回值（prepared_dict | None）

        Returns:
            长度 = num_envs 的列表，每个元素为 model_outputs（dict | None）
        """
        profile_enabled = self._profile_enabled
        total_start = time.perf_counter() if profile_enabled else 0.0

        unpack_start = time.perf_counter() if profile_enabled else 0.0
        decoded_prepared: List[Optional[Dict]] = []
        for prepared in per_env_prepared:
            if prepared is None:
                decoded_prepared.append(None)
            else:
                decoded_prepared.append(unpack_prepared(prepared))
        unpack_ms = (time.perf_counter() - unpack_start) * 1000.0 if profile_enabled else 0.0

        num_envs = len(decoded_prepared)
        results: List[Optional[Dict]] = [None] * num_envs

        # 1. 收集所有需要推理的 flat_jobs
        collect_start = time.perf_counter() if profile_enabled else 0.0
        flat_jobs = self._collect_flat_jobs(decoded_prepared, results)
        collect_ms = (time.perf_counter() - collect_start) * 1000.0 if profile_enabled else 0.0

        if not flat_jobs:
            pack_start = time.perf_counter() if profile_enabled else 0.0
            packed = [pack_model_outputs(r) if r is not None else None for r in results]
            pack_ms = (time.perf_counter() - pack_start) * 1000.0 if profile_enabled else 0.0
            self._maybe_log_profile(
                num_envs=num_envs,
                flat_jobs=flat_jobs,
                chunks=[],
                stage_ms={
                    'unpack': unpack_ms,
                    'collect': collect_ms,
                    'build_chunks': 0.0,
                    'forward': 0.0,
                    'scatter': 0.0,
                    'pack': pack_ms,
                    'total': (time.perf_counter() - total_start) * 1000.0 if profile_enabled else 0.0,
                },
            )
            return packed

        # 2. 分桶+切块 & forward
        build_chunks_start = time.perf_counter() if profile_enabled else 0.0
        chunks = self._build_chunks(flat_jobs)
        build_chunks_ms = (
            (time.perf_counter() - build_chunks_start) * 1000.0 if profile_enabled else 0.0
        )

        forward_start = time.perf_counter() if profile_enabled else 0.0
        all_per_focal: List[Dict] = []
        for chunk in chunks:
            chunk_results = self._forward_chunk(chunk)
            all_per_focal.extend(chunk_results)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0 if profile_enabled else 0.0

        # 3. 按 env_idx 聚合并回填
        scatter_start = time.perf_counter() if profile_enabled else 0.0
        per_env_outputs = self._scatter_chunk_results(all_per_focal, decoded_prepared, results)
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0 if profile_enabled else 0.0

        pack_start = time.perf_counter() if profile_enabled else 0.0
        packed_outputs = [pack_model_outputs(r) if r is not None else None for r in per_env_outputs]
        pack_ms = (time.perf_counter() - pack_start) * 1000.0 if profile_enabled else 0.0

        self._maybe_log_profile(
            num_envs=num_envs,
            flat_jobs=flat_jobs,
            chunks=chunks,
            stage_ms={
                'unpack': unpack_ms,
                'collect': collect_ms,
                'build_chunks': build_chunks_ms,
                'forward': forward_ms,
                'scatter': scatter_ms,
                'pack': pack_ms,
                'total': (time.perf_counter() - total_start) * 1000.0 if profile_enabled else 0.0,
            },
        )
        return packed_outputs

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _read_env_flag(name: str, default: str = '0') -> bool:
        value = str(_os.getenv(name, default)).strip().lower()
        return value in {'1', 'true', 'yes', 'on'}

    def _maybe_log_profile(
        self,
        num_envs: int,
        flat_jobs: List[Dict],
        chunks: List[List[Dict]],
        stage_ms: Dict[str, float],
    ) -> None:
        if not self._profile_enabled:
            return

        self._profile_counter += 1
        if self._profile_counter % self._profile_every != 0:
            return

        chunk_sizes = [len(chunk) for chunk in chunks]
        chunk_tokens = [sum(self._estimate_job_tokens(job) for job in chunk) for chunk in chunks]

        def _summary(values: List[int]) -> Tuple[int, float, int]:
            if not values:
                return 0, 0.0, 0
            return min(values), float(sum(values)) / len(values), max(values)

        cs_min, cs_avg, cs_max = _summary(chunk_sizes)
        ct_min, ct_avg, ct_max = _summary(chunk_tokens)

        print(
            (
                "[ExternalTeacher][Profile] call=%d envs=%d flat_jobs=%d chunks=%d "
                "chunk_jobs[min/avg/max]=%d/%.2f/%d chunk_tokens[min/avg/max]=%d/%.1f/%d "
                "ms(unpack=%.2f collect=%.2f build=%.2f forward=%.2f scatter=%.2f pack=%.2f total=%.2f)"
            ) % (
                self._profile_counter,
                num_envs,
                len(flat_jobs),
                len(chunks),
                cs_min,
                cs_avg,
                cs_max,
                ct_min,
                ct_avg,
                ct_max,
                stage_ms.get('unpack', 0.0),
                stage_ms.get('collect', 0.0),
                stage_ms.get('build_chunks', 0.0),
                stage_ms.get('forward', 0.0),
                stage_ms.get('scatter', 0.0),
                stage_ms.get('pack', 0.0),
                stage_ms.get('total', 0.0),
            )
        )

    def _collect_flat_jobs(
        self,
        per_env_prepared: List[Optional[Dict]],
        results: List[Optional[Dict]],
    ) -> List[Dict]:
        """收集并排序 flat jobs，保持可复现顺序。"""
        flat_jobs: List[Dict] = []
        for env_idx, prepared in enumerate(per_env_prepared):
            if prepared is None:
                continue

            if prepared.get('status') == 'skip':
                results[env_idx] = {
                    'status': 'skip',
                    'env_idx': env_idx,
                    'step_t': prepared.get('step_t'),
                    'token_index': prepared.get('token_index'),
                    'action_results': {},
                    'rtg_results': {},
                    'processed_rtg_veh_ids': [],
                    'dead_ids': prepared.get('dead_ids', []),
                    'error': None,
                }
                continue

            for fb in prepared.get('focal_batches', []):
                flat_jobs.append({
                    'env_idx': env_idx,
                    'prepared': prepared,
                    'focal_batch': fb,
                })

        # 固定顺序：先 env_idx，再 focal_id，保证采样与调度可复现
        flat_jobs.sort(
            key=lambda job: (job['env_idx'], job['focal_batch'].get('focal_id', -1))
        )
        return flat_jobs

    def _build_empty_env_result(self, prepared: Dict, env_idx: int, status: str = 'ok') -> Dict:
        return {
            'status': status,
            'env_idx': env_idx,
            'step_t': prepared.get('step_t'),
            'token_index': prepared.get('token_index'),
            'action_results': {},
            'rtg_results': {},
            'processed_rtg_veh_ids': [],
            'dead_ids': prepared.get('dead_ids', []),
            'error': None,
        }

    def _scatter_chunk_results(
        self,
        all_per_focal: List[Dict],
        per_env_prepared: List[Optional[Dict]],
        results: List[Optional[Dict]],
    ) -> List[Optional[Dict]]:
        """将 per-focal 结果按 env 聚合并回填到 per-env 输出。"""
        env_accum: Dict[int, Dict] = {}
        for pf in all_per_focal:
            eidx = int(pf['env_idx'])
            if eidx not in env_accum:
                prepared = pf['prepared']
                env_accum[eidx] = self._build_empty_env_result(
                    prepared=prepared,
                    env_idx=eidx,
                    status='ok',
                )

            acc = env_accum[eidx]
            acc['action_results'].update(pf.get('action_results', {}))
            acc['rtg_results'].update(pf.get('rtg_results', {}))
            acc['processed_rtg_veh_ids'].extend(pf.get('processed_rtg_veh_ids', []))
            if pf.get('error') and acc.get('error') is None:
                acc['error'] = pf.get('error')

        for eidx, acc in env_accum.items():
            results[eidx] = acc

        # 对于 status=ok 但没有 focal 结果的 env（如都 dead），补齐空结果。
        for env_idx, prepared in enumerate(per_env_prepared):
            if prepared is None:
                continue
            if prepared.get('status') != 'ok':
                continue
            if results[env_idx] is None:
                results[env_idx] = self._build_empty_env_result(
                    prepared=prepared,
                    env_idx=env_idx,
                    status='ok',
                )

        return results

    def _job_shape_key(self, job: Dict) -> Tuple[int, int]:
        """用于分桶的 shape key（激进模式：只按 seq_len 与 agent 维度分桶）。"""
        fb = job['focal_batch']
        motion_data_np = fb.get('motion_data_np', {})
        agent_states = motion_data_np.get('agent_states')

        seq_len = int(
            fb.get(
                'seq_len',
                agent_states.shape[1] if agent_states is not None and agent_states.ndim >= 2 else 1,
            )
        )
        num_agents = int(
            agent_states.shape[0] if agent_states is not None and agent_states.ndim >= 1 else 1
        )
        return (seq_len, num_agents)

    def _estimate_job_tokens(self, job: Dict) -> int:
        """估算 job 计算开销，用于切块避免 OOM。"""
        fb = job['focal_batch']
        motion_data_np = fb.get('motion_data_np', {})
        agent_states = motion_data_np.get('agent_states')

        seq_len = int(
            fb.get(
                'seq_len',
                agent_states.shape[1] if agent_states is not None and agent_states.ndim >= 2 else 1,
            )
        )
        max_num_agents = int(
            agent_states.shape[0] if agent_states is not None and agent_states.ndim >= 1 else 1
        )
        seq_len = max(seq_len, 1)
        max_num_agents = max(max_num_agents, 1)
        # ctrl-sim 默认 token 类型数：state/rtg/action
        return seq_len * max_num_agents * 3

    def _build_chunks(self, flat_jobs: List[Dict]) -> List[List[Dict]]:
        """按 shape 分桶，再按 token 预算和 micro_batch 切块。"""
        if not flat_jobs:
            return []

        buckets: Dict[Tuple[int, int], List[Dict]] = {}
        for job in flat_jobs:
            key = self._job_shape_key(job)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(job)

        max_chunk_tokens = int(
            getattr(self, '_max_chunk_tokens', getattr(self, 'DEFAULT_MAX_CHUNK_TOKENS', 64512))
        )
        max_chunk_jobs = self.micro_batch if self.micro_batch and self.micro_batch > 0 else None

        chunks: List[List[Dict]] = []
        for shape_key in sorted(buckets.keys()):
            current_chunk: List[Dict] = []
            current_tokens = 0

            for job in buckets[shape_key]:
                job_tokens = self._estimate_job_tokens(job)
                exceed_token_budget = (
                    bool(current_chunk)
                    and max_chunk_tokens > 0
                    and current_tokens + job_tokens > max_chunk_tokens
                )
                exceed_job_budget = (
                    max_chunk_jobs is not None
                    and len(current_chunk) >= max_chunk_jobs
                )
                if exceed_token_budget or exceed_job_budget:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0

                current_chunk.append(job)
                current_tokens += job_tokens

            if current_chunk:
                chunks.append(current_chunk)

        return chunks

    def _get_or_create_collate_buffers(
        self,
        cache_key: Tuple[Any, ...],
        specs: Dict[str, Tuple[Tuple[int, ...], np.dtype, Any]],
    ) -> Dict[str, np.ndarray]:
        """按 cache_key 复用 collate 缓冲，减少重复分配。"""
        if not hasattr(self, '_collate_numpy_buffers'):
            self._collate_numpy_buffers = {}
        buffers = self._collate_numpy_buffers.get(cache_key)
        if buffers is None:
            # 形状组合异常增多时直接清空，避免缓存无限增长。
            if len(self._collate_numpy_buffers) >= 64:
                self._collate_numpy_buffers.clear()
            buffers = {}
            for name, (shape, dtype, fill_value) in specs.items():
                arr = np.zeros(shape, dtype=dtype)
                if fill_value != 0:
                    arr.fill(fill_value)
                buffers[name] = arr
            self._collate_numpy_buffers[cache_key] = buffers
            return buffers

        for name, (_, _, fill_value) in specs.items():
            buffers[name].fill(fill_value)
        return buffers

    def _collate_chunk_with_padding(self, chunk: List[Dict]) -> Tuple[MotionData, Dict[str, Any]]:
        """将一个 chunk 的 jobs collate 成 batched MotionData，并生成 pad/mask 元信息。"""
        if not chunk:
            raise ValueError("chunk must not be empty")

        def _get_field_shape(field_name: str) -> Tuple[int, ...]:
            for job in chunk:
                arr = job['focal_batch']['motion_data_np'].get(field_name)
                if arr is not None:
                    return tuple(arr.shape)
            raise ValueError(f"missing field '{field_name}' in chunk")

        batch_size = len(chunk)
        agent_shape = _get_field_shape('agent_states')   # [A, T, F]
        road_shape = _get_field_shape('road_points')     # [R, P, F]
        agent_feat_dim = int(agent_shape[2])
        road_pts_dim = int(road_shape[2])
        road_type_dim = int(_get_field_shape('road_types')[1])
        agent_type_dim = int(_get_field_shape('agent_types')[1])
        goal_dim = int(_get_field_shape('goals')[1])
        rtg_dim = int(_get_field_shape('rtgs')[2])

        max_agents = 1
        max_seq_len = 1
        max_roads = 1
        max_road_pts = 1
        max_timestep_feat_dim = 1
        has_timestep_feat_dim = False
        for job in chunk:
            motion_data_np = job['focal_batch']['motion_data_np']
            max_agents = max(max_agents, int(motion_data_np['agent_states'].shape[0]))
            max_seq_len = max(max_seq_len, int(motion_data_np['agent_states'].shape[1]))
            max_roads = max(max_roads, int(motion_data_np['road_points'].shape[0]))
            max_road_pts = max(max_road_pts, int(motion_data_np['road_points'].shape[1]))
            cur_timesteps = motion_data_np['timesteps']
            if cur_timesteps.ndim == 3:
                has_timestep_feat_dim = True
                max_timestep_feat_dim = max(max_timestep_feat_dim, int(cur_timesteps.shape[2]))

        first_motion = chunk[0]['focal_batch']['motion_data_np']
        timesteps_shape: Tuple[int, ...]
        if has_timestep_feat_dim:
            timesteps_shape = (batch_size, max_agents, max_seq_len, max_timestep_feat_dim)
        else:
            timesteps_shape = (batch_size, max_agents, max_seq_len)

        cache_key = (
            batch_size,
            max_agents,
            max_seq_len,
            max_roads,
            max_road_pts,
            max_timestep_feat_dim if has_timestep_feat_dim else 0,
            int(has_timestep_feat_dim),
            np.dtype(first_motion['agent_states'].dtype).str,
            np.dtype(first_motion['agent_types'].dtype).str,
            np.dtype(first_motion['goals'].dtype).str,
            np.dtype(first_motion['actions'].dtype).str,
            np.dtype(first_motion['rtgs'].dtype).str,
            np.dtype(first_motion['timesteps'].dtype).str,
            np.dtype(first_motion['moving_agent_mask'].dtype).str,
            np.dtype(first_motion['road_points'].dtype).str,
            np.dtype(first_motion['road_types'].dtype).str,
        )
        specs = {
            'agent_states_b': (
                (batch_size, max_agents, max_seq_len, agent_feat_dim),
                np.dtype(first_motion['agent_states'].dtype),
                0,
            ),
            'agent_types_b': (
                (batch_size, max_agents, agent_type_dim),
                np.dtype(first_motion['agent_types'].dtype),
                -1,
            ),
            'goals_b': (
                (batch_size, max_agents, goal_dim),
                np.dtype(first_motion['goals'].dtype),
                0,
            ),
            'actions_b': (
                (batch_size, max_agents, max_seq_len),
                np.dtype(first_motion['actions'].dtype),
                0,
            ),
            'rtgs_b': (
                (batch_size, max_agents, max_seq_len, rtg_dim),
                np.dtype(first_motion['rtgs'].dtype),
                0,
            ),
            'timesteps_b': (
                timesteps_shape,
                np.dtype(first_motion['timesteps'].dtype),
                0,
            ),
            'moving_agent_mask_b': (
                (batch_size, max_agents),
                np.dtype(first_motion['moving_agent_mask'].dtype),
                0,
            ),
            'road_points_b': (
                (batch_size, max_roads, max_road_pts, road_pts_dim),
                np.dtype(first_motion['road_points'].dtype),
                0,
            ),
            'road_types_b': (
                (batch_size, max_roads, road_type_dim),
                np.dtype(first_motion['road_types'].dtype),
                -1,
            ),
            'agent_valid_mask': ((batch_size, max_agents), np.dtype(np.bool_), False),
            'road_valid_mask': ((batch_size, max_roads), np.dtype(np.bool_), False),
            'time_valid_mask': ((batch_size, max_seq_len), np.dtype(np.bool_), False),
            'token_index_per_job': ((batch_size,), np.dtype(np.int64), 0),
            'seq_len_per_job': ((batch_size,), np.dtype(np.int64), 0),
        }
        buffers = self._get_or_create_collate_buffers(cache_key=cache_key, specs=specs)
        agent_states_b = buffers['agent_states_b']
        agent_types_b = buffers['agent_types_b']
        goals_b = buffers['goals_b']
        actions_b = buffers['actions_b']
        rtgs_b = buffers['rtgs_b']
        timesteps_b = buffers['timesteps_b']
        moving_agent_mask_b = buffers['moving_agent_mask_b']
        road_points_b = buffers['road_points_b']
        road_types_b = buffers['road_types_b']
        agent_valid_mask = buffers['agent_valid_mask']
        road_valid_mask = buffers['road_valid_mask']
        time_valid_mask = buffers['time_valid_mask']
        token_index_per_job = buffers['token_index_per_job']
        seq_len_per_job = buffers['seq_len_per_job']

        for batch_idx, job in enumerate(chunk):
            prepared = job['prepared']
            fb = job['focal_batch']
            motion_data_np = fb['motion_data_np']

            n_agents = int(motion_data_np['agent_states'].shape[0])
            seq_len = int(motion_data_np['agent_states'].shape[1])
            n_roads = int(motion_data_np['road_points'].shape[0])
            n_road_pts = int(motion_data_np['road_points'].shape[1])

            agent_states_b[batch_idx, :n_agents, :seq_len] = motion_data_np['agent_states']
            agent_types_b[batch_idx, :n_agents] = motion_data_np['agent_types']
            goals_b[batch_idx, :n_agents] = motion_data_np['goals']
            actions_b[batch_idx, :n_agents, :seq_len] = motion_data_np['actions']
            rtgs_b[batch_idx, :n_agents, :seq_len] = motion_data_np['rtgs']
            cur_timesteps = motion_data_np['timesteps']
            if has_timestep_feat_dim:
                if cur_timesteps.ndim == 3:
                    t_feat_dim = int(cur_timesteps.shape[2])
                    timesteps_b[batch_idx, :n_agents, :seq_len, :t_feat_dim] = cur_timesteps
                elif cur_timesteps.ndim == 2:
                    timesteps_b[batch_idx, :n_agents, :seq_len, 0] = cur_timesteps
                else:
                    raise ValueError(
                        f"Unsupported timesteps ndim={cur_timesteps.ndim}, expected 2 or 3"
                    )
            else:
                if cur_timesteps.ndim == 3:
                    if cur_timesteps.shape[2] != 1:
                        raise ValueError(
                            f"Unsupported timesteps shape={cur_timesteps.shape} for 3D->2D cast"
                        )
                    timesteps_b[batch_idx, :n_agents, :seq_len] = cur_timesteps[..., 0]
                elif cur_timesteps.ndim == 2:
                    timesteps_b[batch_idx, :n_agents, :seq_len] = cur_timesteps
                else:
                    raise ValueError(
                        f"Unsupported timesteps ndim={cur_timesteps.ndim}, expected 2 or 3"
                    )
            moving_agent_mask_b[batch_idx, :n_agents] = motion_data_np['moving_agent_mask']
            road_points_b[batch_idx, :n_roads, :n_road_pts] = motion_data_np['road_points']
            road_types_b[batch_idx, :n_roads] = motion_data_np['road_types']

            valid_agent_count = int(fb.get('valid_agent_count', n_agents))
            valid_road_count = int(fb.get('valid_road_count', n_roads))
            valid_agent_count = max(0, min(valid_agent_count, max_agents))
            valid_road_count = max(0, min(valid_road_count, max_roads))
            agent_valid_mask[batch_idx, :valid_agent_count] = True
            road_valid_mask[batch_idx, :valid_road_count] = True

            seq_len_per_job[batch_idx] = seq_len
            time_valid_mask[batch_idx, :seq_len] = True

            raw_token_index = int(prepared.get('token_index', 0))
            clamped_token_index = 0 if seq_len <= 0 else max(0, min(raw_token_index, seq_len - 1))
            token_index_per_job[batch_idx] = clamped_token_index

        batched_np = {
            'agent': {
                'agent_states': agent_states_b,
                'agent_types': agent_types_b,
                'goals': goals_b,
                'actions': actions_b,
                'rtgs': rtgs_b,
                'timesteps': timesteps_b,
                'moving_agent_mask': moving_agent_mask_b,
            },
            'map': {
                'road_points': road_points_b,
                'road_types': road_types_b,
            },
        }
        batched_data = MotionData(from_numpy(batched_np))
        batched_data = batched_data.to(self.device)

        batch_meta = {
            'jobs': chunk,
            'token_index_per_job': torch.from_numpy(token_index_per_job).to(self.device),
            'seq_len_per_job': torch.from_numpy(seq_len_per_job).to(self.device),
            'agent_valid_mask': torch.from_numpy(agent_valid_mask).to(self.device),
            'road_valid_mask': torch.from_numpy(road_valid_mask).to(self.device),
            'time_valid_mask': torch.from_numpy(time_valid_mask).to(self.device),
        }
        return batched_data, batch_meta

    @torch.no_grad()
    def _decode_rtg_stage_batched(
        self,
        batched_data: MotionData,
        batch_meta: Dict[str, Any],
        rtg_cache: Dict[Tuple[int, int], Dict[str, Any]],
    ) -> Tuple[MotionData, List[Dict[int, Tuple[float, float, float]]], List[List[int]]]:
        """RTG batched 阶段：一次 forward，逐 job 解码并写回离散 RTG。"""
        preds = self.model(batched_data, eval=True)
        rtg_logits = preds['rtg_preds']

        jobs: List[Dict] = batch_meta['jobs']
        token_index_per_job: torch.Tensor = batch_meta['token_index_per_job']

        rtg_results_by_job: List[Dict[int, Tuple[float, float, float]]] = []
        processed_rtg_veh_ids_by_job: List[List[int]] = []

        for batch_idx, job in enumerate(jobs):
            env_idx = int(job['env_idx'])
            prepared = job['prepared']
            fb = job['focal_batch']
            token_index = int(token_index_per_job[batch_idx].item())
            veh_id_to_idx = prepared.get('veh_id_to_idx', {})
            tilt_by_veh_id = prepared.get('tilt_by_veh_id', {})
            default_tilt = prepared.get('default_tilt', (0, 0, 0))
            prior_rtgs = prepared.get('prior_rtg_results', {})

            new_agent_idx_dict = fb.get('new_agent_idx_dict', {})
            data_veh_ids = fb.get('data_veh_ids', [])
            veh_ids_in_context = fb.get('veh_ids_in_context', [])
            generator = self._get_generator(env_idx)

            rtg_results: Dict[int, Tuple[float, float, float]] = {}
            processed_rtg_veh_ids: List[int] = []
            if not bool(fb.get('predict_rtgs', True)):
                rtg_results_by_job.append(rtg_results)
                processed_rtg_veh_ids_by_job.append(processed_rtg_veh_ids)
                continue

            for veh_id in veh_ids_in_context:
                if veh_id not in veh_id_to_idx:
                    continue
                agent_key = veh_id_to_idx[veh_id]
                if agent_key not in new_agent_idx_dict:
                    continue
                idx_in_model = int(new_agent_idx_dict[agent_key])
                cache_key = (env_idx, veh_id)

                # 已有 prior（兼容旧逻辑入口）
                if veh_id in prior_rtgs and isinstance(prior_rtgs[veh_id], dict):
                    discrete = prior_rtgs[veh_id].get('discrete')
                    continuous = prior_rtgs[veh_id].get('continuous')
                    if discrete is not None and len(discrete) == 3:
                        g_idx, v_idx, r_idx = int(discrete[0]), int(discrete[1]), int(discrete[2])
                        batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 0] = g_idx
                        batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 1] = v_idx
                        batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 2] = r_idx
                        if continuous is None:
                            rtg_np = np.array([[[g_idx, v_idx, r_idx]]], dtype=np.int64)
                            rtg_cont = undiscretize_rtgs(
                                rtg_np,
                                self.rtg_discretization,
                                self.min_rtg_pos, self.max_rtg_pos,
                                self.min_rtg_veh, self.max_rtg_veh,
                                self.min_rtg_road, self.max_rtg_road,
                            )
                            continuous = (
                                float(rtg_cont[0, 0, 0]),
                                float(rtg_cont[0, 0, 1]),
                                float(rtg_cont[0, 0, 2]),
                            )
                        rtg_cache[cache_key] = {
                            'discrete': (g_idx, v_idx, r_idx),
                            'continuous': (
                                float(continuous[0]),
                                float(continuous[1]),
                                float(continuous[2]),
                            ),
                        }
                        continue

                # 命中 cache：复用离散 RTG，避免重复采样
                if cache_key in rtg_cache:
                    cached = rtg_cache[cache_key]
                    g_idx, v_idx, r_idx = cached['discrete']
                    batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 0] = int(g_idx)
                    batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 1] = int(v_idx)
                    batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 2] = int(r_idx)
                    continue

                # 新采样
                rtg_logits_3 = rtg_logits[batch_idx, idx_in_model, token_index].reshape(
                    self.rtg_discretization, self.num_reward_components
                )

                is_tilted = veh_id in data_veh_ids
                if is_tilted and veh_id in tilt_by_veh_id:
                    g_tilt, v_tilt, e_tilt = tilt_by_veh_id[veh_id]
                elif is_tilted:
                    g_tilt, v_tilt, e_tilt = default_tilt
                else:
                    g_tilt, v_tilt, e_tilt = 0, 0, 0

                tilt_logits_np = get_tilt_logits(
                    self.rtg_discretization,
                    g_tilt, v_tilt, e_tilt,
                )
                (g_idx_t, v_idx_t, r_idx_t), (g_val, v_val, r_val) = decode_predicted_rtg(
                    rtg_logits_3,
                    tilt_logits_np,
                    self.rtg_discretization,
                    self.min_rtg_pos, self.max_rtg_pos,
                    self.min_rtg_veh, self.max_rtg_veh,
                    self.min_rtg_road, self.max_rtg_road,
                    device=self.device,
                    generator=generator,
                )

                g_idx = int(g_idx_t.item() if hasattr(g_idx_t, 'item') else g_idx_t)
                v_idx = int(v_idx_t.item() if hasattr(v_idx_t, 'item') else v_idx_t)
                r_idx = int(r_idx_t.item() if hasattr(r_idx_t, 'item') else r_idx_t)

                batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 0] = g_idx
                batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 1] = v_idx
                batched_data['agent'].rtgs[batch_idx, idx_in_model, token_index, 2] = r_idx

                continuous_vals = (float(g_val), float(v_val), float(r_val))
                rtg_results[veh_id] = continuous_vals
                processed_rtg_veh_ids.append(veh_id)
                rtg_cache[cache_key] = {
                    'discrete': (g_idx, v_idx, r_idx),
                    'continuous': continuous_vals,
                }

            rtg_results_by_job.append(rtg_results)
            processed_rtg_veh_ids_by_job.append(processed_rtg_veh_ids)

        return batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job

    @torch.no_grad()
    def _decode_action_stage_batched(
        self,
        batched_data: MotionData,
        batch_meta: Dict[str, Any],
    ) -> List[Dict[int, Tuple[float, float]]]:
        """Action batched 阶段：一次 forward，逐 job 解码 data_veh_ids。"""
        preds = self.model(batched_data, eval=True)
        action_logits = preds['action_preds']

        jobs: List[Dict] = batch_meta['jobs']
        token_index_per_job: torch.Tensor = batch_meta['token_index_per_job']
        action_results_by_job: List[Dict[int, Tuple[float, float]]] = []

        for batch_idx, job in enumerate(jobs):
            env_idx = int(job['env_idx'])
            prepared = job['prepared']
            fb = job['focal_batch']
            token_index = int(token_index_per_job[batch_idx].item())
            veh_id_to_idx = prepared.get('veh_id_to_idx', {})
            new_agent_idx_dict = fb.get('new_agent_idx_dict', {})
            data_veh_ids = fb.get('data_veh_ids', [])
            sampling = prepared.get('sampling', {})
            generator = self._get_generator(env_idx)

            action_results: Dict[int, Tuple[float, float]] = {}
            for veh_id in data_veh_ids:
                if veh_id not in veh_id_to_idx:
                    continue
                agent_key = veh_id_to_idx[veh_id]
                if agent_key not in new_agent_idx_dict:
                    continue
                idx_in_model = int(new_agent_idx_dict[agent_key])
                logits_1d = action_logits[batch_idx, idx_in_model, token_index]

                accel, steer = decode_predicted_action(
                    logits_1d,
                    sampling.get('action_temperature', 1.0),
                    sampling.get('nucleus_sampling', False),
                    sampling.get('nucleus_threshold', 1.0),
                    self.accel_discretization, self.steer_discretization,
                    self.min_accel, self.max_accel,
                    self.min_steer, self.max_steer,
                    generator=generator,
                )
                action_results[veh_id] = (accel, steer)

            action_results_by_job.append(action_results)

        return action_results_by_job

    def _forward_chunk_batched(self, chunk: List[Dict]) -> List[Dict]:
        """chunk 级 batched forward：collate -> RTG -> action -> per-job 结果。"""
        if not chunk:
            return []

        try:
            batched_data, batch_meta = self._collate_chunk_with_padding(chunk)
            rtg_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
            batched_data, rtg_results_by_job, processed_rtg_veh_ids_by_job = (
                self._decode_rtg_stage_batched(
                    batched_data=batched_data,
                    batch_meta=batch_meta,
                    rtg_cache=rtg_cache,
                )
            )
            action_results_by_job = self._decode_action_stage_batched(
                batched_data=batched_data,
                batch_meta=batch_meta,
            )

            per_job_results: List[Dict] = []
            for idx, job in enumerate(chunk):
                per_job_results.append({
                    'env_idx': job['env_idx'],
                    'prepared': job['prepared'],
                    'action_results': action_results_by_job[idx],
                    'rtg_results': rtg_results_by_job[idx],
                    'processed_rtg_veh_ids': processed_rtg_veh_ids_by_job[idx],
                    'error': None,
                })
            return per_job_results

        except Exception as exc:
            warnings.warn(f"[ExternalTeacher] batched chunk forward error: {exc}")
            fallback_results = []
            for job in chunk:
                fallback_results.append({
                    'env_idx': job['env_idx'],
                    'prepared': job['prepared'],
                    'action_results': {},
                    'rtg_results': {},
                    'processed_rtg_veh_ids': [],
                    'error': str(exc),
                })
            return fallback_results

    def _get_generator(self, env_idx: int) -> torch.Generator:
        if env_idx not in self._generators:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.base_seed + env_idx * 100003)
            self._generators[env_idx] = gen
        return self._generators[env_idx]

    def _forward_chunk(self, chunk: List[Dict]) -> List[Dict]:
        """兼容入口：统一走 batched chunk forward。"""
        return self._forward_chunk_batched(chunk)
