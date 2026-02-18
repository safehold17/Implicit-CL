# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import datetime
import json
import logging
import os
import time
from typing import Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = True,
        seeds=None,
    ):
        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
        self.xpid = xpid
        self._tick = 0
        self._wrote_log_header = False
        self._avg_metric_fields = [
            "collision",
            "offroad",
            "position_reached",
            "goal_reached",
            "progress",
            "episode_reward",
            "mean_agent_return",
            "agent_value_loss",
            "agent_pg_loss",
            "agent_dist_entropy",
            "plr_episode_reward",
        ]

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["xpid"] = self.xpid

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")

        train_full_distribution = xp_args.get('train_full_distribution', False)
        seed_buffer_size = xp_args.get('level_replay_seed_buffer_size', 0)
        self.record_seed_diffs = \
            train_full_distribution and seed_buffer_size > 0

        self.seeds = None
        if not self.record_seed_diffs and seeds:
            self.seeds = [str(seed) for seed in seeds]

        # To stdout handler.
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        self.basepath = os.path.join(rootdir, self.xpid)
        if not os.path.exists(self.basepath):
            self._logger.info("Creating log directory: %s", self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        if symlink_to_latest:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
            level_weights="{base}/level_weights.csv".format(base=self.basepath),
            level_seeds="{base}/level_seeds.csv".format(base=self.basepath),
            final_test_eval="{base}/final_test_eval.csv".format(base=self.basepath)
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning(
                "Path to meta file already exists. " "Not overriding meta."
            )
        else:
            self._save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning(
                "Path to message file already exists. " "New data will be appended."
            )

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info("Saving logs data to %s", self.paths["logs"])
        self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        self.final_test_eval_fieldnames = [
            'env_name',
            'agent_name',
            'num_test_seeds',
            'mean_episode_return',
            'median_episode_return',
        ]
        self.level_seeds_fieldnames = ['new_seeds', 'new_seed_indices']
        if os.path.exists(self.paths["logs"]):
            self._logger.warning(
                "Path to log file already exists. " "New data will be appended."
            )
            self._wrote_log_header = True
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Skip non-numeric rows (for example trailing avg rows) when
                # recovering ticks from previous logs.
                for row in reversed(lines):
                    if not row:
                        continue
                    tick = self._safe_float(row[0])
                    if tick is None:
                        continue
                    self._tick = int(tick) + 1
                    break

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)
        self._levelweightsfile = open(self.paths["level_weights"], "a")
        self._levelweightswriter = csv.writer(self._levelweightsfile)
        self._levelseedsfile = open(self.paths["level_seeds"], "a")
        self._levelseedswriter = csv.DictWriter(self._levelseedsfile, fieldnames=self.level_seeds_fieldnames)
        self._finaltestfile = open(self.paths["final_test_eval"], "a")
        self._finaltestwriter = csv.DictWriter(self._finaltestfile, fieldnames=self.final_test_eval_fieldnames)
        self.tensor_board_writer: SummaryWriter = SummaryWriter(
            log_dir=os.path.join(self.basepath, "tb")
        )
        self._tb_single_write_metric_map = {
            "train/mean_agent_return": "mean_agent_return",
            "train/agent_value_loss": "agent_value_loss",
            "train/update_reward": "update_reward",
            "train/plr_update_reward": "plr_update_reward",
        }

        self._tb_process_avg_metric_map = {
            "train/progress": "progress",
            "train/plr_progress": "plr_progress",
            "env/veh_goal_avg": "veh_goal_avg",
            "env/plr_veh_goal_avg": "plr_veh_goal_avg",
            "env/veh_veh_avg": "veh_veh_avg",
            "env/plr_veh_veh_avg": "plr_veh_veh_avg",
            "env/veh_edge_avg": "veh_edge_avg",
            "env/plr_veh_edge_avg": "plr_veh_edge_avg",
        }
        self._tb_expected_num_processes = int(xp_args["num_processes"])
        self._tb_process_avg_buffer = {tag: {} for tag in self._tb_process_avg_metric_map}

        if self.seeds and not self.record_seed_diffs:
            self._levelweightsfile.write("# %s\n" % ",".join(self.seeds))
            self._levelweightsfile.flush()

        self._finaltestwriter.writeheader()
        self._finaltestfile.flush()

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        prioritized_fields = [
            '_tick',
            'process_idx',
            '_time',
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

        if tick is None:
            to_log["_tick"] = self._tick
            self._tick += 1
        else:
            tick = int(tick)
            to_log["_tick"] = tick
            # Keep implicit ticks monotonic even when caller explicitly sets ticks.
            if tick >= self._tick:
                self._tick = tick + 1
        to_log["_time"] = time.time()

        old_fieldnames = list(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        # Keep avg-only metric column schema stable during normal logging,
        # rather than adding a new column at close() time.
        if "plr_episode_reward" not in self.fieldnames:
            self.fieldnames.append("plr_episode_reward")

        reordered = []
        for k in prioritized_fields:
            if k in self.fieldnames:
                reordered.append(k)
        for k in self.fieldnames:
            if k not in reordered:
                reordered.append(k)
        self.fieldnames = self._move_fields_before(
            reordered,
            anchor="plr_opp0_goal_tilt",
            fields=[
                "plr_veh_goal_avg",
                "veh_goal_avg",
                "plr_veh_veh_avg",
                "veh_veh_avg",
                "plr_veh_edge_avg",
                "veh_edge_avg",
            ],
        )

        if old_fieldnames != self.fieldnames:
            self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)
            self._fieldwriter.writerow(self.fieldnames)
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if not self._wrote_log_header:
            self._logwriter.writeheader()
            self._wrote_log_header = True

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()
        self.log_to_tensorboard(to_log)

    def log_level_weights(self, weights, seeds=None):
        if self.record_seed_diffs:
            if self.seeds is None:
                self.seeds = seeds.copy()
                level_seed_log = {
                    'new_seeds': " ".join([str(s) for s in self.seeds]),
                    'new_seed_indices': " ".join([str(i) for i in range(len(self.seeds))]),
                }
            else:
                new_seed_indices = np.nonzero(self.seeds - seeds)[0]
                new_seeds = seeds[new_seed_indices]
                self.seeds = seeds.copy()
                level_seed_log = {
                    'new_seeds': " ".join([str(s) for s in new_seeds]),
                    'new_seed_indices': " ".join([str(i) for i in new_seed_indices]),
                }
            self._levelseedswriter.writerow(level_seed_log)
            self._levelseedsfile.flush()

        self._levelweightswriter.writerow(weights)
        self._levelweightsfile.flush()

    def log_final_test_eval(self, to_log):
        self._finaltestwriter.writerow(to_log)
        self._finaltestfile.flush()

    def log_to_tensorboard(self, stats: Dict) -> None:
        process_idx = int(stats.get("process_idx", 0))
        global_step = int(stats["total_student_grad_updates"])
        if process_idx == 0:
            for tag, key in self._tb_single_write_metric_map.items():
                value = self._get_float_stat(stats, key)
                if value is not None:
                    self.tensor_board_writer.add_scalar(tag, value, global_step)
        for tag, key in self._tb_process_avg_metric_map.items():
            value = self._get_float_stat(stats, key)
            if value is not None:
                self._collect_tb_process_avg(tag, global_step, process_idx, value)
        self.tensor_board_writer.flush()

    def close(self, successful: bool = True) -> None:
        if successful:
            self._append_avg_row()

        self.metadata["date_end"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()
        self._flush_tb_process_avg_buffers()
        self.tensor_board_writer.close()

    def _save_metadata(self) -> None:
        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)

    def latest_tick(self):
        with open(self.paths["logs"], "r") as logsfile:
            csvreader = csv.reader(logsfile)
            latest = None
            for row in csvreader:
                if not row:
                    continue
                tick = self._safe_float(row[0])
                if tick is None:
                    continue
                latest = int(tick)
            return latest if latest is not None else 0

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _move_fields_before(fieldnames, anchor, fields):
        if anchor not in fieldnames:
            return fieldnames

        result = [f for f in fieldnames if f not in fields]
        anchor_idx = result.index(anchor)
        to_insert = [f for f in fields if f in fieldnames]
        if not to_insert:
            return result
        return result[:anchor_idx] + to_insert + result[anchor_idx:]

    def _collect_tb_process_avg(self, tag, step, process_idx, value):
        step_buffer = self._tb_process_avg_buffer[tag]
        stale_steps = [stale_step for stale_step in step_buffer if stale_step < step]
        for stale_step in sorted(stale_steps):
            self._write_tb_process_avg(tag, stale_step, step_buffer.pop(stale_step))

        process_values = step_buffer.setdefault(step, {})
        process_values[process_idx] = value

        if len(process_values) >= self._tb_expected_num_processes:
            self._write_tb_process_avg(tag, step, process_values)
            step_buffer.pop(step, None)

    def _write_tb_process_avg(self, tag, step, process_values):
        if not process_values:
            return
        mean_value = float(np.mean(list(process_values.values())))
        self.tensor_board_writer.add_scalar(tag, mean_value, step)

    def _flush_tb_process_avg_buffers(self):
        for tag, step_buffer in self._tb_process_avg_buffer.items():
            for step in sorted(step_buffer):
                self._write_tb_process_avg(tag, step, step_buffer[step])
            step_buffer.clear()

    @staticmethod
    def _get_float_stat(stats: Dict, key: str):
        value = stats.get(key)
        if value is None:
            return None
        return float(value)

    def _append_avg_row(self):
        if not os.path.exists(self.paths["logs"]):
            return

        with open(self.paths["logs"], "r", newline="") as csvfile:
            rows = list(csv.reader(csvfile))

        if not rows:
            return

        process_idx = self.fieldnames.index("process_idx") if "process_idx" in self.fieldnames else None
        plr_marker_idx = self.fieldnames.index("plr_update_reward") \
            if "plr_update_reward" in self.fieldnames else None
        episode_reward_idx = self.fieldnames.index("episode_reward") if "episode_reward" in self.fieldnames else None

        metric_values = {k: [] for k in self._avg_metric_fields if k != "plr_episode_reward"}
        plr_episode_rewards = []

        for row in rows:
            if not row:
                continue
            if row[0].startswith("#"):
                continue

            # Only aggregate true per-process data rows.
            if process_idx is None or process_idx >= len(row):
                continue
            process_value = self._safe_float(row[process_idx])
            if process_value is None:
                continue

            for metric in metric_values:
                if metric not in self.fieldnames:
                    continue
                metric_idx = self.fieldnames.index(metric)
                if metric_idx >= len(row):
                    continue
                value = self._safe_float(row[metric_idx])
                if value is not None:
                    metric_values[metric].append(value)

            # PLR row criterion: non-empty plr_update_reward.
            if plr_marker_idx is not None and plr_marker_idx < len(row):
                plr_marker = str(row[plr_marker_idx]).strip()
                if plr_marker != "" and episode_reward_idx is not None and episode_reward_idx < len(row):
                    ep_reward = self._safe_float(row[episode_reward_idx])
                    if ep_reward is not None:
                        plr_episode_rewards.append(ep_reward)

        if not any(metric_values.values()) and not plr_episode_rewards:
            return

        avg_row = {k: "" for k in self.fieldnames}
        if "_tick" in avg_row:
            avg_row["_tick"] = "avg"
        if "process_idx" in avg_row:
            avg_row["process_idx"] = "avg"
        if "_time" in avg_row:
            avg_row["_time"] = time.time()

        for metric, values in metric_values.items():
            if values:
                avg_row[metric] = float(np.mean(values))
        if plr_episode_rewards and "plr_episode_reward" in avg_row:
            avg_row["plr_episode_reward"] = float(np.mean(plr_episode_rewards))

        self._logwriter.writerow(avg_row)
        self._logfile.flush()
