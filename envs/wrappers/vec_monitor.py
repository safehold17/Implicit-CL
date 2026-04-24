# Copyright (c) 2019 Antonin Raffin
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is an extended version of
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_monitor.py

from collections import deque
import time

import numpy as np
from stable_baselines3.common.monitor import ResultsWriter

from .vec_env import VecEnvWrapper

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def _reset_episode_tracking(self, indices=None) -> None:
        """Reset episode trackers for all envs or a selected subset."""
        if self.eprets is None or self.eplens is None:
            self.eprets = np.zeros(self.num_envs, 'f')
            self.eplens = np.zeros(self.num_envs, 'i')
        if indices is None:
            self.eprets.fill(0)
            self.eplens.fill(0)
            return
        self.eprets[list(indices)] = 0
        self.eplens[list(indices)] = 0

    def reset(self):
        obs = self.venv.reset()
        self._reset_episode_tracking()
        return obs

    def reset_agent(self):
        obs = self.venv.reset_agent()
        self._reset_episode_tracking()
        return obs

    def reset_random(self):
        obs = self.venv.reset_random()
        self._reset_episode_tracking()
        return obs

    def reset_alp_gmm(self, level):
        obs = self.venv.reset_alp_gmm(level)
        self._reset_episode_tracking()
        return obs

    def reset_to_level(self, level, index):
        obs = self.venv.reset_to_level(level, index)
        self._reset_episode_tracking(indices=[index])
        return obs

    def reset_to_level_batch(self, level):
        obs = self.venv.reset_to_level_batch(level)
        self._reset_episode_tracking()
        return obs

    def reset_to_level_indices(self, levels, indices):
        obs = self.venv.reset_to_level_indices(levels, indices)
        self._reset_episode_tracking(indices=indices)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = self._track_episodes(rews, dones, infos)
        return obs, rews, dones, newinfos

    def _track_episodes(self, rews, dones, infos):
        """Shared episode tracking logic for step_env / step_complete."""
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info

        return newinfos

    def step_env(self, actions, reset_random=False, auto_reset_on_done=True):
        obs, rews, dones, infos = self.venv.step_env(
            actions,
            reset_random=reset_random,
            auto_reset_on_done=auto_reset_on_done,
        )
        newinfos = self._track_episodes(rews, dones, infos)
        return obs, rews, dones, newinfos

    def step_prepare(self, action):
        return self.venv.step_prepare(action)

    def step_complete(
        self,
        model_outputs,
        reset_random=False,
        auto_reset_on_done=True,
    ):
        obs, rews, dones, infos = self.venv.step_complete(
            model_outputs,
            reset_random=reset_random,
            auto_reset_on_done=auto_reset_on_done,
        )
        newinfos = self._track_episodes(rews, dones, infos)
        return obs, rews, dones, newinfos
