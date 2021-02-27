import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.mc.common import AbstractObservableModel
from scripts.mc.mh_uniform_kernel import MhUniformKernel


class SmcUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractObservableModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_count: int,
        kernel_count: int,
    ) -> None:
        self.model = model
        self.interval = interval
        self.particle_dim = particle_dim
        self.particle_count = particle_count
        self.particle_weights: np.array = np.zeros(particle_count)
        self.particle_mh_trace_len: int = 1000
        self.kernel_count: int = kernel_count
        self.kernel_params: np.array = np.zeros(self.particle_dim,
                                                self.kernel_count)

    def _init(self):
        for i in range(0, self.particle_count):
            particle, weight = self._draw_particle_from_kernel_idx(0)
            self._update_particle_by_idx(idx=i,
                                         particle=particle,
                                         weight=weight)

    def _get_interval(self, sigma: Optional[float]) -> Tuple[float]:
        l, u = self.interval
        if not sigma:
            return self.interval
        new_l, new_u = (l - sigma, u + sigma)
        new_l = l if new_l < l else new_l
        new_u = u if new_u > u else new_u
        return (new_l, new_u)

    def _draw_particle_from_kernel_idx(
        self,
        idx: int,
    ) -> Tuple[np.array, float]:
        interval = self._get_interval(self.kernel_params[idx])
        sigma = self.kernel_params[:, idx]
        particle = np.zeros(self.particle_dim)
        weight = 0
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
            weight = 1.0 / (2 * sigma[i])
        return (particle, weight)

    def _get_particle_by_idx(self, idx: int) -> Tuple[np.array, float]:
        assert idx > self.particle_count
        return self.particle_trace[:idx], self.particle_weights[idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array,
                                weight: float):
        assert idx > self.particle_count
        assert len(particle) == self.particle_dim
        self.particle_trace[:idx] = particle
        self.particle_weights[idx] = weight

    def _append_particle(self, particle: np.array, weight: float):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle, weight)

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

    def _mh_next_particle(self, ):
        pass

    def _mh_transition(
        self,
        particle: np.array,
        weight: int,
    ) -> Tuple[np.array, float]:
        mh_particle_trace: np.array = np.zeros(self.particle_dim,
                                               self.particle_mh_trace_len)
        for i in range(0, self.particle_mh_trace_len):
            candidate_particle = self._next_particle()
            idx = self.particle_curr_idx
            last_log_llh = self.particle_weights[self, idx]
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

    def _pertubate(self):
        for i in range(0, self.particle_count):
            particle, weight = self._get_particle_by_idx(i)
            new_particle, new_weight = self._mh_transition(particle, weight)

    def sample(self):
        self._init()
        for i in range(0, self.kernel_count):

            pass

    def _pertubate(self):
        pass