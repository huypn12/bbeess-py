import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.mc.common import AbstractObservableModel


class MhUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractObservableModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        use_sigma: bool = False,
    ) -> None:
        self.model = model
        self.interval = interval
        self.particle_dim = particle_dim
        self.particle_trace_len = particle_trace_len
        self.particle_trace: np.array = np.array(particle_dim,
                                                 particle_trace_len,
                                                 dtype=float)
        self.particle_curr_idx: int = -1
        self.particle_weights: np.array = np.zeros(particle_trace_len)
        self.use_sigma = use_sigma

    def _init(self):
        self.particle_curr_idx = 0
        first_particle = self._sample()
        self._update_particle_by_idx(0, first_particle)

    def _estimate_weight(self, particle: np.array):
        return self.model.estimate_log_llh(particle)

    def _get_interval(self, sigma: Optional[float]) -> Tuple[float]:
        l, u = self.interval
        if not sigma:
            return self.interval
        new_l, new_u = (l - sigma, u + sigma)
        new_l = l if new_l < l else new_l
        new_u = u if new_u > u else new_u
        return (new_l, new_u)

    def _get_particle_by_idx(self, idx: int) -> np.array:
        assert idx > self.particle_trace_len
        return self.particle_trace[:idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array):
        assert idx > self.particle_trace_len
        assert len(particle) == self.particle_dim
        self.particle_trace[:idx] = particle
        self.particle_weights[idx] = self._estimate_weight(particle)

    def _append_particle(self, particle: np.array):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle)

    def _get_sigma(self, ) -> Optional[np.array]:
        sigma = np.zeros(self.particle_dim)
        if not self.use_sigma:
            return sigma
        for i in range(0, self.particle_dim):
            idx = self.particle_curr_idx
            _min = np.amin(self.particle_trace[i][0:idx + 1])
            _max = np.amax(self.particle_trace[i][0:idx + 1])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        sigma = self._get_sigma()
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def sample(self):
        self._init()
        for i in range(0, self.particle_trace_len):
            candidate_particle = self._next_particle()
            idx = self.particle_curr_idx
            last_log_llh = self.particle_weights[idx]
            log_llh = self._estimate_weight(candidate_particle)
            acceptance_rate = np.min(0, log_llh - last_log_llh)
            u = np.random.uniform(0, 1)
            if u < acceptance_rate:
                self._append_particle(candidate_particle)
            else:
                epsilon = 1e-4
                acceptance_rate = np.random.uniform(0, 1)
                if u < epsilon:
                    self._append_particle(candidate_particle)
