import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.model.abstract_model import AbstractRationalModel


class SmcRfUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractRationalModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        kernel_count: int,
        observed_data: List[int],
    ) -> None:
        self.model = model
        self.interval = interval
        self.particle_dim = particle_dim
        self.particle_trace_len = particle_trace_len
        self.particle_trace: np.array = np.zeros(
            (particle_dim, particle_trace_len), dtype=float
        )
        self.particle_weights: np.array = np.zeros(particle_trace_len, dtype=float)
        self.particle_mh_trace_len: int = 1000
        self.kernel_count: int = kernel_count
        self.kernel_params: np.array = np.zeros(
            (self.particle_dim, self.kernel_count), dtype=float
        )
        self.observed_data = observed_data

    def _init(self):
        for i in range(0, self.particle_trace_len):
            particle, weight = self._draw_particle_from_kernel_idx(0)
            self._update_particle_by_idx(idx=i, particle=particle, weight=weight)

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
        sigma = self.kernel_params[:, idx]
        particle = np.zeros(self.particle_dim)
        weight = 0
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
            weight = 1.0 / (2 * sigma[i])
        return (particle, weight)

    def _get_particle_by_idx(self, idx: int) -> Tuple[np.array, float]:
        assert idx > self.particle_trace_len
        return self.particle_trace[:idx], self.particle_weights[idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array, weight: float):
        assert len(particle) == self.particle_dim
        self.particle_trace[:idx] = particle
        self.particle_weights[idx] = weight

    def _append_particle(self, particle: np.array, weight: float):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle, weight)

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        sigma = self._get_sigma(self, self.particle_trace)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def _get_sigma(self, particle_idx: int, particle_trace: np.array):
        sigma = np.zeros(self.particle_dim)
        if not self.use_sigma:
            return sigma
        for i in range(0, self.particle_dim):
            _min = np.amin(particle_trace[0 : particle_idx + 1, i])
            _max = np.amax(particle_trace[0 : particle_idx + 1, i])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _mh_init(
        self,
        particle: np.array,
        weight: np.array,
        mh_particle_trace: np.array,
        mh_particle_weights: np.array,
    ):
        mh_particle_trace[0] = particle
        mh_particle_weights[0] = weight

    def _mh_next_particle(self, particle_idx: int, particle_trace: np.array):
        particle = np.zeros(self.particle_dim)
        sigma = self._get_sigma(particle_idx, particle_trace)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def _mh_transition(
        self,
        particle: np.array,
        weight: int,
    ) -> Tuple[np.array, float]:
        mh_particle_trace: np.array = np.zeros(
            self.particle_mh_trace_len, self.particle_dim
        )
        mh_particle_weights: np.array = np.zeros(self.particle_mh_trace_len)
        self._mh_init(particle, weight, mh_particle_trace, mh_particle_weights)
        mh_particle_idx = 1
        while mh_particle_idx < self.particle_mh_trace_len - 1:
            last_log_llh = mh_particle_weights[mh_particle_idx]
            candidate_particle = self._mh_next_particle(
                mh_particle_idx, mh_particle_trace
            )
            candidate_log_llh = self._estimate_weight(candidate_particle)
            acceptance_rate = np.min(0, candidate_log_llh - last_log_llh)
            acceptance_rate = np.exp(acceptance_rate)
            u = np.random.uniform(0, 1)
            if u < acceptance_rate:
                mh_particle_trace[mh_particle_idx] = particle
                mh_particle_weights[mh_particle_idx] = weight
                mh_particle_idx += 1
            else:
                acceptance_rate = 1e-1
                u = np.random.uniform(0, 1)
                if u < acceptance_rate:
                    mh_particle_trace[mh_particle_idx] = particle
                    mh_particle_weights[mh_particle_idx] = weight
                    mh_particle_idx += 1

    def _correct(self, kernel_idx: int):
        sigma = self._get_sigma(kernel_idx, self.particle_trace)
        last_sigma = self.kernel_params(kernel_idx - 1)
        w = [last_sigma[i] / sigma[i] for i in range(0, len(sigma))]
        return w

    def _select(self, weights: np.array):
        w = np.copy(weights)
        w = w / sum(w)
        new_particles_idx = np.random.choice(
            self.particle_trace_len, self.particle_dim, replace=True, p=w
        )
        for idx in range(0, len(new_particles_idx)):
            self._update_particle_by_idx(
                idx,
                particle=self.particle_trace[idx],
                weight=1,
            )
        return new_particles_idx

    def _pertubate(self, kernel_idx: int):
        for idx in range(0, self.particle_trace_len):
            particle, weight = self._get_particle_by_idx(idx)
            new_particle, new_weight = self._mh_transition(particle, weight)
            self._update_particle_by_idx(idx, particle=new_particle, weight=new_weight)

    def run(self):
        self._init()
        for t in range(1, self.kernel_count):
            # Correct
            weight = self._correct(t)
            # Select
            self._select(weight)
            # Mutation
            self._pertubate(t)
        return self._get_result()

    def _get_result(self):
        return (self.particle_trace, self.particle_weights)