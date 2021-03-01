import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.mc.common import AbstractObservableModel, AbstractRationalModel, AbstractSimulationModel
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
        self.particle_trace: np.array = np.zeros(particle_dim, particle_count)
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

    def _get_smc_sigma(self, ) -> Optional[np.array]:
        return self._get_sigma(self, self.particle_trace)

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        sigma = self._get_smc_sigma()
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def _get_sigma(self, idx: int, particle_trace: np.array):
        sigma = np.zeros(self.particle_dim)
        if not self.use_sigma:
            return sigma
        for i in range(0, self.particle_dim):
            _min = np.amin(particle_trace[0:idx + 1, i])
            _max = np.amax(particle_trace[0:idx + 1, i])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _mh_next_particle(self, idx: int):
        particle = np.zeros(self.particle_dim)
        sigma = self._get_sigma(idx)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def _mh_transition(
        self,
        particle: np.array,
        weight: int,
    ) -> Tuple[np.array, float]:
        mh_particle_trace: np.array = np.zeros(self.particle_mh_trace_len,
                                               self.particle_dim)
        mh_particle_weights: np.array = np.zeros(self.particle_mh_trace_len)
        mh_particle_trace[0] = particle
        mh_particle_weights[0] = weight
        for idx in range(0, self.particle_mh_trace_len - 1):
            last_log_llh = mh_particle_weights[idx]
            candidate_particle = self._mh_next_particle()
            candidate_log_llh = self._estimate_weight(candidate_particle)
            acceptance_rate = np.min(0, candidate_log_llh - last_log_llh)
            u = np.random.uniform(0, 1)
            if u < acceptance_rate:
                mh_particle_trace[idx + 1] = particle
                mh_particle_weights[idx + 1] = weight
            else:
                epsilon = 1e-4
                acceptance_rate = np.random.uniform(0, 1)
                if u < epsilon:
                    mh_particle_trace[idx + 1] = particle
                    mh_particle_weights[idx + 1] = weight

    def _correct(self, kernel_idx: int):
        pass

    def _select(self, kernel_idx: int):
        weights = np.copy(self.particle_weights)
        weights = np.exp(weights)
        weights = weights / sum(weights)

    def _pertubate(self, kernel_idx: int):
        for idx in range(0, self.particle_count):
            particle, weight = self._get_particle_by_idx(idx)
            new_particle, new_weight = self._mh_transition(particle, weight)
            self._update_particle_by_idx(idx,
                                         particle=new_particle,
                                         weight=new_weight)

    def _update_kernels(self):
        pass

    def sample(self):
        self._init()
        for t in range(1, self.kernel_count):
            if t == 0:
                pass
            else:
                if self.llh_free:
                    # estimate distance, abc method
                    pass
                else:
                    # MH-step
                    pass
                pass
            # estimate distacce

    def _pertubate(self):
        pass