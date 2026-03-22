from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_opponent_inference_io_directory_replaces_inference_bridge() -> None:
    opponent_dir = PROJECT_ROOT / "ctrlsim_adapter" / "opponent_vehicle"

    assert (opponent_dir / "opponent_inference_io" / "__init__.py").exists()
    assert not (opponent_dir / "inference_bridge").exists()


def test_opponent_inference_io_references_are_updated() -> None:
    expected_files = [
        PROJECT_ROOT / "ctrlsim_adapter" / "opponent_vehicle" / "__init__.py",
        PROJECT_ROOT / "ctrlsim_adapter" / "opponent_vehicle" / "services" / "state_service.py",
        PROJECT_ROOT / "ctrlsim_adapter" / "opponent_vehicle" / "_opponent_state" / "reset.py",
    ]

    for path in expected_files:
        content = path.read_text(encoding="utf-8")
        assert "inference_bridge" not in content
        assert "opponent_inference_io" in content
