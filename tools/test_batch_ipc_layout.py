from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_batch_ipc_directory_replaces_batch_protocol() -> None:
    batch_inference_dir = PROJECT_ROOT / "batch_inference"

    assert (batch_inference_dir / "batch_ipc" / "__init__.py").exists()
    assert not (batch_inference_dir / "batch_protocol").exists()


def test_batch_ipc_import_references_are_updated() -> None:
    expected_files = [
        PROJECT_ROOT / "batch_inference" / "external_teacher.py",
        PROJECT_ROOT
        / "ctrlsim_adapter"
        / "opponent_vehicle"
        / "opponent_inference_io"
        / "prepare_inference_payload.py",
        PROJECT_ROOT
        / "ctrlsim_adapter"
        / "opponent_vehicle"
        / "opponent_inference_io"
        / "apply_outputs.py",
        PROJECT_ROOT / "tools" / "test_opponent_batch_parity.py",
    ]

    for path in expected_files:
        content = path.read_text(encoding="utf-8")
        assert "batch_protocol" not in content
        assert "batch_ipc" in content
