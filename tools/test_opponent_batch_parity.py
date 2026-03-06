from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from batch_inference import ExternalTeacher
from batch_inference.ipc_codec import unpack_prepared
from envs.nocturne_ctrlsim import ScenarioLevel
from tools.test_ctrlsim_policy_solving_rate import CtrlSimEgoWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.ckpt"
DEFAULT_VEHICLE_MAP_PATH = PROJECT_ROOT / "data" / "vehicle_map_filtered_train.json"
DEFAULT_SCENARIO_INDEX_PATH = PROJECT_ROOT / "data" / "scenarios_index_filtered_train.json"
DEFAULT_SCENARIO_DATA_DIR = Path("/home/chen/data/nocturne_waymo/formatted_json_v2_no_tl_train")
DEFAULT_PREPROCESS_DIR = Path("/home/chen/data/preprocess/train")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pytest-opponent-batch-parity"

BASE_SEED = 12345
TRACE_STEPS = 40
FIRST_MODEL_STEP = 9
NUM_OPPONENTS = 7

FIXED_LEVEL = ScenarioLevel(
    scenario_id="tfrecord-00540-of-01000_44",
    seed=1845187043,
    goal_tilt=0,
    veh_veh_tilt=0,
    veh_edge_tilt=0,
    per_vehicle_tilting=(
        20,
        -18,
        -23,
        16,
        2,
        23,
        -2,
        21,
        -6,
        -17,
        20,
        -24,
        -10,
        -5,
        15,
        6,
        20,
        3,
        17,
        -22,
        15,
    ),
)


def _get_env_path(env_name: str, default_path: Path) -> str:
    return os.getenv(env_name, str(default_path))


def _ensure_runtime_resources() -> Dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用，跳过 opponent batch parity 集成测试。")

    output_dir = _get_env_path("CTRLSIM_PARITY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    resources = {
        "checkpoint_path": _get_env_path("CTRLSIM_PARITY_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH),
        "vehicle_map_path": _get_env_path("CTRLSIM_PARITY_VEHICLE_MAP_PATH", DEFAULT_VEHICLE_MAP_PATH),
        "scenario_index_path": _get_env_path("CTRLSIM_PARITY_SCENARIO_INDEX_PATH", DEFAULT_SCENARIO_INDEX_PATH),
        "scenario_data_dir": _get_env_path("CTRLSIM_PARITY_SCENARIO_DATA_DIR", DEFAULT_SCENARIO_DATA_DIR),
        "preprocess_dir": _get_env_path("CTRLSIM_PARITY_PREPROCESS_DIR", DEFAULT_PREPROCESS_DIR),
        "output_dir": output_dir,
    }

    missing = [
        f"{name}={path}"
        for name, path in resources.items()
        if name != "output_dir" and not os.path.exists(path)
    ]
    if missing:
        pytest.skip("测试资源缺失: " + ", ".join(missing))
    return resources


def _wrapper_kwargs(resources: Dict[str, str], batch_inference: bool) -> Dict[str, Any]:
    return {
        "scenario_index_path": resources["scenario_index_path"],
        "opponent_checkpoint": resources["checkpoint_path"],
        "scenario_data_dir": resources["scenario_data_dir"],
        "preprocess_dir": resources["preprocess_dir"],
        "vehicle_map_path": resources["vehicle_map_path"],
        "opponent_k": NUM_OPPONENTS,
        "max_episode_steps": 90,
        "tilting_mode": "per_vehicle",
        "tilt_range": [-25.0, 25.0],
        "show_level_log": False,
        "record_video": False,
        "show_vehicle_ids": False,
        "output_dir": resources["output_dir"],
        "xpid": "pytest-opponent-batch-parity",
        "device": "cuda",
        "seed": BASE_SEED,
        "batch_inference": batch_inference,
        "opponent_sparse_inference_enabled": False,
        "opponent_sparse_inference_interval": 2,
        "sparse_inference_action_repeat": False,
    }


def _reset_wrapper_to_fixed_level(wrapper: CtrlSimEgoWrapper) -> None:
    wrapper.env.seed(BASE_SEED)
    wrapper.env.reset_to_level(FIXED_LEVEL)
    wrapper._reset_episode_state()
    wrapper._maybe_disable_opponent_tilting()
    wrapper._reset_ego_adapter()


def _clear_teacher_rng(teacher: ExternalTeacher) -> None:
    teacher._generators.clear()


def _step_nobatch_and_capture_opponent(wrapper: CtrlSimEgoWrapper) -> Dict[int, Tuple[float, float]]:
    t = wrapper.env.current_step
    ego_actions = wrapper.ego_adapter.step(t, wrapper.env.vehicles)
    if wrapper.ego_id is not None and wrapper.ego_id in ego_actions:
        accel, steer = ego_actions[wrapper.ego_id]
    else:
        accel, steer = 0.0, 0.0
    opponent_actions = wrapper.env.opponent.step(t, wrapper.env.vehicles)
    wrapper._step_with_ego_action(accel, steer, opponent_actions)
    return {
        int(veh_id): (float(action[0]), float(action[1]))
        for veh_id, action in opponent_actions.items()
    }


def _step_batch_and_capture_opponent(
    wrapper: CtrlSimEgoWrapper,
    teacher: ExternalTeacher,
) -> Dict[int, Tuple[float, float]]:
    prepared = wrapper.step_prepare(None)
    ego_outputs = teacher.batched_forward([prepared["ego"]])[0]
    opponent_outputs = teacher.batched_forward([prepared["opponent"]])[0]

    ego_actions = wrapper.ego_adapter.apply_predictions(ego_outputs)
    if wrapper.ego_id is not None and wrapper.ego_id in ego_actions:
        accel, steer = ego_actions[wrapper.ego_id]
    else:
        accel, steer = 0.0, 0.0

    opponent_actions = wrapper.env.opponent.apply_predictions(opponent_outputs)
    wrapper._step_with_ego_action(accel, steer, opponent_actions)
    return {
        int(veh_id): (float(action[0]), float(action[1]))
        for veh_id, action in opponent_actions.items()
    }


def _compare_action_dicts(
    actual: Dict[int, Tuple[float, float]],
    expected: Dict[int, Tuple[float, float]],
    atol: float = 1e-6,
) -> None:
    assert sorted(actual.keys()) == sorted(expected.keys())
    for veh_id in sorted(expected.keys()):
        np.testing.assert_allclose(
            np.asarray(actual[veh_id], dtype=np.float32),
            np.asarray(expected[veh_id], dtype=np.float32),
            atol=atol,
            rtol=0.0,
            err_msg=f"veh_id={veh_id} 动作不一致",
        )


def _motion_data_to_numpy(motion_data: Any) -> Dict[str, np.ndarray]:
    return {
        "agent_states": motion_data["agent"].agent_states.detach().cpu().numpy()[0].copy(),
        "agent_types": motion_data["agent"].agent_types.detach().cpu().numpy()[0].copy(),
        "goals": motion_data["agent"].goals.detach().cpu().numpy()[0].copy(),
        "actions": motion_data["agent"].actions.detach().cpu().numpy()[0].copy(),
        "rtgs": motion_data["agent"].rtgs.detach().cpu().numpy()[0].copy(),
        "timesteps": motion_data["agent"].timesteps.detach().cpu().numpy()[0].copy(),
        "moving_agent_mask": motion_data["agent"].moving_agent_mask.detach().cpu().numpy()[0].copy(),
        "road_points": motion_data["map"].road_points.detach().cpu().numpy()[0].copy(),
        "road_types": motion_data["map"].road_types.detach().cpu().numpy()[0].copy(),
    }


def _collect_nobatch_snapshot(wrapper: CtrlSimEgoWrapper) -> Dict[str, Any]:
    opponent = wrapper.env.opponent
    t = wrapper.env.current_step
    opponent._vehicle_data_dict = opponent._update_vehicle_data_dict(
        t,
        wrapper.env.vehicles,
        opponent._vehicle_data_dict,
    )
    opponent.update_policy_state(t)

    policy = opponent._policy
    assert policy is not None

    motion_datas, dead_ids, new_agent_idx_dicts, data_veh_ids = policy.get_data(
        opponent._gt_data_dict,
        opponent._preproc_data,
        opponent.dataset,
        opponent._vehicles_to_control,
        t,
    )
    token_index = t if t < policy.cfg_rl_waymo.train_context_length else -1

    per_focal: Dict[int, Dict[str, Any]] = {}
    for focal_id, motion_data in motion_datas.items():
        focal_data_veh_ids = list(data_veh_ids[focal_id])
        data = motion_data.to(wrapper.device)
        focal_idx_in_model = new_agent_idx_dicts[focal_id][policy.veh_id_to_idx[focal_data_veh_ids[0]]]
        data["focal_idx_in_model"] = focal_idx_in_model
        with torch.no_grad():
            preds = policy.model(data, eval=True)
        action_logits = preds["action_preds"].float().detach().cpu().numpy()[0]

        per_focal[int(focal_id)] = {
            "token_index": int(token_index),
            "motion_data": _motion_data_to_numpy(motion_data),
            "data_veh_ids": focal_data_veh_ids,
            "veh_ids_in_context": [
                policy.idx_to_veh_id[idx]
                for idx in policy.relevant_agent_idxs[focal_id]
            ],
            "new_agent_idx_dict": dict(new_agent_idx_dicts[focal_id]),
            "action_logits": action_logits,
            "argmax_by_vehicle": {
                int(veh_id): int(
                    np.argmax(
                        action_logits[
                            new_agent_idx_dicts[focal_id][policy.veh_id_to_idx[veh_id]],
                            token_index,
                        ]
                    )
                )
                for veh_id in focal_data_veh_ids
            },
        }

    return {
        "dead_ids": sorted(int(veh_id) for veh_id in dead_ids),
        "per_focal": per_focal,
    }


def _collect_batch_snapshot(
    wrapper: CtrlSimEgoWrapper,
    teacher: ExternalTeacher,
) -> Dict[str, Any]:
    prepared = unpack_prepared(wrapper.env.opponent.prepare_step(wrapper.env.current_step, wrapper.env.vehicles))
    per_focal: Dict[int, Dict[str, Any]] = {}

    for focal_batch in prepared["focal_batches"]:
        job = {
            "env_idx": 0,
            "prepared": prepared,
            "focal_batch": focal_batch,
        }
        batched_data, batch_meta = teacher._collate_chunk_with_padding([job])
        token_index = int(batch_meta["token_index_per_job"][0].item())
        with teacher.model_forward_context():
            preds = teacher.model(batched_data, eval=True)
        action_logits = preds["action_preds"].float().detach().cpu().numpy()[0]

        focal_id = int(focal_batch["focal_id"])
        data_veh_ids = [int(veh_id) for veh_id in focal_batch["data_veh_ids"]]
        new_agent_idx_dict = {
            int(key): int(value)
            for key, value in focal_batch["new_agent_idx_dict"].items()
        }
        veh_id_to_idx = prepared["veh_id_to_idx"]

        per_focal[focal_id] = {
            "token_index": token_index,
            "motion_data": {
                key: np.asarray(value).copy()
                for key, value in focal_batch["motion_data_np"].items()
            },
            "data_veh_ids": data_veh_ids,
            "veh_ids_in_context": [int(veh_id) for veh_id in focal_batch["veh_ids_in_context"]],
            "new_agent_idx_dict": new_agent_idx_dict,
            "action_logits": action_logits,
            "argmax_by_vehicle": {
                veh_id: int(
                    np.argmax(
                        action_logits[
                            new_agent_idx_dict[veh_id_to_idx[veh_id]],
                            token_index,
                        ]
                    )
                )
                for veh_id in data_veh_ids
            },
        }

    return {
        "dead_ids": sorted(int(veh_id) for veh_id in prepared["dead_ids"]),
        "per_focal": per_focal,
    }


def _assert_numpy_fields_match(
    actual: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray],
) -> None:
    assert set(actual.keys()) == set(expected.keys())
    for field in sorted(actual.keys()):
        actual_value = np.asarray(actual[field])
        expected_value = np.asarray(expected[field])
        if field in {"agent_states", "goals"}:
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                atol=3e-2,
                rtol=1e-6,
                err_msg=f"{field} 不一致",
            )
        elif field == "road_points":
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                atol=2e-4,
                rtol=0.0,
                err_msg=f"{field} 不一致",
            )
        else:
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                atol=1e-6,
                rtol=0.0,
                err_msg=f"{field} 不一致",
            )


def _assert_snapshots_match(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    assert actual["dead_ids"] == expected["dead_ids"]
    assert sorted(actual["per_focal"].keys()) == sorted(expected["per_focal"].keys())

    for focal_id in sorted(expected["per_focal"].keys()):
        actual_focal = actual["per_focal"][focal_id]
        expected_focal = expected["per_focal"][focal_id]
        assert actual_focal["token_index"] == expected_focal["token_index"]
        assert actual_focal["data_veh_ids"] == expected_focal["data_veh_ids"]
        assert actual_focal["veh_ids_in_context"] == expected_focal["veh_ids_in_context"]
        assert actual_focal["new_agent_idx_dict"] == expected_focal["new_agent_idx_dict"]
        assert actual_focal["argmax_by_vehicle"] == expected_focal["argmax_by_vehicle"]
        _assert_numpy_fields_match(actual_focal["motion_data"], expected_focal["motion_data"])
        np.testing.assert_allclose(
            actual_focal["action_logits"],
            expected_focal["action_logits"],
            atol=1.5e-4,
            rtol=0.0,
            err_msg=f"focal_id={focal_id} action_logits 不一致",
        )


@pytest.fixture(scope="module")
def parity_resources() -> Dict[str, str]:
    return _ensure_runtime_resources()


def _collect_nobatch_snapshot_from_fresh_runtime(resources: Dict[str, str]) -> Dict[str, Any]:
    wrapper = CtrlSimEgoWrapper(**_wrapper_kwargs(resources, batch_inference=False))
    try:
        _reset_wrapper_to_fixed_level(wrapper)
        for _ in range(FIRST_MODEL_STEP):
            _step_nobatch_and_capture_opponent(wrapper)
        return _collect_nobatch_snapshot(wrapper)
    finally:
        wrapper.close()


def _collect_batch_snapshot_from_fresh_runtime(resources: Dict[str, str]) -> Dict[str, Any]:
    teacher = ExternalTeacher(
        checkpoint_path=resources["checkpoint_path"],
        device="cuda",
        base_seed=BASE_SEED,
    )
    wrapper = CtrlSimEgoWrapper(**_wrapper_kwargs(resources, batch_inference=True))
    try:
        _clear_teacher_rng(teacher)
        _reset_wrapper_to_fixed_level(wrapper)
        for _ in range(FIRST_MODEL_STEP):
            _step_batch_and_capture_opponent(wrapper, teacher)
        return _collect_batch_snapshot(wrapper, teacher)
    finally:
        wrapper.close()


def _collect_nobatch_trace_from_fresh_runtime(resources: Dict[str, str]) -> list[Dict[int, Tuple[float, float]]]:
    wrapper = CtrlSimEgoWrapper(**_wrapper_kwargs(resources, batch_inference=False))
    try:
        _reset_wrapper_to_fixed_level(wrapper)
        trace = []
        for _ in range(TRACE_STEPS):
            trace.append(_step_nobatch_and_capture_opponent(wrapper))
        return trace
    finally:
        wrapper.close()


def _collect_batch_trace_from_fresh_runtime(resources: Dict[str, str]) -> list[Dict[int, Tuple[float, float]]]:
    teacher = ExternalTeacher(
        checkpoint_path=resources["checkpoint_path"],
        device="cuda",
        base_seed=BASE_SEED,
    )
    wrapper = CtrlSimEgoWrapper(**_wrapper_kwargs(resources, batch_inference=True))
    try:
        _clear_teacher_rng(teacher)
        _reset_wrapper_to_fixed_level(wrapper)
        trace = []
        for _ in range(TRACE_STEPS):
            trace.append(_step_batch_and_capture_opponent(wrapper, teacher))
        return trace
    finally:
        wrapper.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.requires_checkpoint
@pytest.mark.requires_real_data
def test_fixed_level_opponent_inputs_logits_and_argmax_match(parity_resources) -> None:
    no_batch_snapshot = _collect_nobatch_snapshot_from_fresh_runtime(parity_resources)
    batch_snapshot = _collect_batch_snapshot_from_fresh_runtime(parity_resources)
    _assert_snapshots_match(batch_snapshot, no_batch_snapshot)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.requires_checkpoint
@pytest.mark.requires_real_data
def test_fixed_level_opponent_action_trace_matches_for_40_steps(parity_resources) -> None:
    no_batch_trace = _collect_nobatch_trace_from_fresh_runtime(parity_resources)
    batch_trace = _collect_batch_trace_from_fresh_runtime(parity_resources)

    assert len(batch_trace) == len(no_batch_trace) == TRACE_STEPS
    for step_idx, (no_batch_actions, batch_actions) in enumerate(zip(no_batch_trace, batch_trace), start=1):
        _compare_action_dicts(batch_actions, no_batch_actions)
        assert step_idx <= TRACE_STEPS
