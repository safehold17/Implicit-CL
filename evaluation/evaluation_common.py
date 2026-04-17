"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import csv
import os
from typing import Any, Sequence

import numpy as np

EPISODE_METRIC_FIELDS = (
    "number",
    "scenario_id",
    "collision",
    "offroad",
    "position_reached",
    "progress",
    "total_episode_reward",
)


def extract_episode_metrics(info: dict[str, Any]) -> dict[str, float | str]:
    """Extract one completed Nocturne episode's metrics from env info."""
    episode_info = info.get("episode", {})
    total_episode_reward = episode_info.get("r", info.get("episode_reward", 0.0))
    return {
        "scenario_id": info.get("scenario_id", ""),
        "collision": float(info.get("collision_occurred", info.get("collision", 0.0))),
        "offroad": float(info.get("offroad_occurred", info.get("offroad", 0.0))),
        "position_reached": float(
            info.get("position_reached_occurred", info.get("position_reached", 0.0))
        ),
        "progress": float(info.get("max_progress", info.get("progress", 0.0))),
        "total_episode_reward": float(total_episode_reward),
    }


def build_episode_metrics_mean_row(
    episode_metrics: Sequence[dict[str, float | str]],
) -> dict[str, float | str]:
    """Build the final mean row for per-episode metrics CSVs."""
    mean_row: dict[str, float | str] = {"number": "mean", "scenario_id": ""}
    for field in EPISODE_METRIC_FIELDS[2:]:
        values = [float(row[field]) for row in episode_metrics]
        mean_row[field] = float(np.mean(values)) if values else 0.0
    return mean_row


def write_episode_metrics_csv(
    output_path: str,
    episode_metrics: Sequence[dict[str, float | str]],
) -> None:
    """Write per-episode metrics and the final mean row to CSV."""
    with open(output_path, "w", newline="") as csvout:
        csvwriter = csv.writer(csvout)
        csvwriter.writerow(list(EPISODE_METRIC_FIELDS))
        for idx, row in enumerate(episode_metrics, start=1):
            csvwriter.writerow(
                [idx, *[row[field] for field in EPISODE_METRIC_FIELDS[1:]]]
            )
        mean_row = build_episode_metrics_mean_row(episode_metrics)
        csvwriter.writerow([mean_row[field] for field in EPISODE_METRIC_FIELDS])


def resolve_csv_output_path(output_dir: str, file_stem: str) -> str:
    """Return a CSV output path and append _redo when the target exists."""
    output_path = f"{output_dir}/{file_stem}.csv"
    if os.path.exists(output_path):
        output_path = f"{output_dir}/{file_stem}_redo.csv"
    return output_path


def validate_nocturne_env_names(env_names: Sequence[str]) -> None:
    """Validate that all requested environments are Nocturne variants."""
    invalid = [name for name in env_names if not name.startswith("Nocturne")]
    if invalid:
        raise ValueError(
            "This evaluation entry only supports Nocturne envs, got: "
            + ", ".join(invalid)
        )


def collect_replay_nocturne_args(
    flags: dict[str, Any],
    cli_args: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Collect replay-only Nocturne args from flags, CLI args, and defaults."""
    required = dict(defaults)
    keys = [
        "opponent_checkpoint",
        "max_episode_steps",
        "student_accel_discretization",
        "student_steer_discretization",
        "done_on_position_reached_only",
        "goal_pos_tolerance",
        "use_speed_heading_target",
    ]
    for key in keys:
        if key in flags:
            required[key] = flags[key]
        elif key in cli_args:
            required[key] = cli_args[key]
    return required


def build_replay_nocturne_env(
    env_ctor: Any,
    *,
    record_video: bool = False,
    process_idx: int | None = None,
    video_dir: str = "videos/",
    **kwargs: Any,
) -> Any:
    """Build one Nocturne env fixed to replay-mode opponents."""
    allowed_nocturne_keys = {
        "scenario_index_path",
        "opponent_checkpoint",
        "scenario_data_dir",
        "preprocess_dir",
        "vehicle_map_path",
        "max_episode_steps",
        "student_accel_discretization",
        "student_steer_discretization",
        "done_on_position_reached_only",
        "goal_pos_tolerance",
        "device",
        "tilting_mode",
        "use_speed_heading_target",
        "opponent_runtime_mode",
    }
    nocturne_kwargs = {k: v for k, v in kwargs.items() if k in allowed_nocturne_keys}
    nocturne_kwargs.setdefault("tilting_mode", "per_vehicle")
    nocturne_kwargs["opponent_runtime_mode"] = "replay"
    env = env_ctor(**nocturne_kwargs)
    if not record_video:
        return env
    return wrap_nocturne_video_env(
        env,
        video_dir=video_dir,
        process_idx=process_idx,
    )


def wrap_nocturne_video_env(
    env: Any,
    video_dir: str,
    process_idx: int | None = None,
    fps: int = 10,
    dpi: int = 100,
) -> Any:
    """Wrap a Nocturne env with start/stop recording management."""

    class VideoWrapper:
        """Delegate env calls while managing Nocturne video recording."""

        def __init__(self) -> None:
            """Store wrapped env metadata for recording."""
            self.env = env
            self.video_dir = video_dir
            self.episode_count = 0
            self.recording_started = False
            self.process_idx = process_idx
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        def _episode_name(self) -> str:
            """Return the recording file prefix for the current episode."""
            if self.process_idx is None:
                return f"episode_{self.episode_count:04d}"
            return f"process{self.process_idx}_episode_{self.episode_count:04d}"

        def _start_if_needed(self) -> None:
            """Start recording once per episode."""
            if self.recording_started:
                return
            self.env.start_recording(
                self.video_dir,
                self._episode_name(),
                fps=fps,
                dpi=dpi,
            )
            self.recording_started = True

        def _stop_if_recording(self) -> None:
            """Stop recording and advance episode counter when needed."""
            if not self.recording_started:
                return
            if getattr(self.env, "recording_video", False):
                self.env.stop_recording(self._episode_name())
            self.episode_count += 1
            self.recording_started = False

        def reset(self, **kwargs: Any) -> Any:
            """Reset the wrapped env, preferring Nocturne random reset."""
            if hasattr(self.env, "reset_random") and not kwargs:
                return self.env.reset_random()
            return self.env.reset(**kwargs)

        def reset_random(self, **kwargs: Any) -> Any:
            """Reset the wrapped env to a random level."""
            return self.env.reset_random(**kwargs)

        def reset_agent(self, **kwargs: Any) -> Any:
            """Reset only the agent-facing state in the wrapped env."""
            if kwargs:
                return self.env.reset_agent(**kwargs)
            return self.env.reset_agent()

        def step(self, action: Any) -> Any:
            """Forward a one-stage env step and stop recording on done."""
            self._start_if_needed()
            obs, reward, done, info = self.env.step(action)
            if done:
                self._stop_if_recording()
            return obs, reward, done, info

        def step_prepare(self, action: Any) -> Any:
            """Forward phase-1 step call and ensure recording has started."""
            self._start_if_needed()
            return self.env.step_prepare(action)

        def step_complete(self, model_output: Any) -> Any:
            """Forward phase-2 step call and stop recording on done."""
            self._start_if_needed()
            obs, reward, done, info = self.env.step_complete(model_output)
            if done:
                self._stop_if_recording()
            return obs, reward, done, info

        def close(self) -> None:
            """Close the wrapped env and flush any active recording."""
            self._stop_if_recording()
            self.env.close()

        def __getattr__(self, name: str) -> Any:
            """Delegate unknown attributes to the wrapped env."""
            return getattr(self.env, name)

    return VideoWrapper()


def start_headless_display() -> Any | None:
    """Start a Linux headless display when pyvirtualdisplay is available."""
    import sys

    if not sys.platform.startswith("linux"):
        return None

    import pyvirtualdisplay

    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
    display.start()
    return display


def stop_headless_display(display: Any | None) -> None:
    """Stop a previously started headless display."""
    if display is not None:
        display.stop()
