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
            (particle_trace_len, particle_dim), dtype=float
        )
        self.particle_weights: np.array = np.zeros(particle_trace_len, dtype=float)
        self.particle_llh: np.array = np.zeros(particle_trace_len, dtype=float)
        self.particle_mh_trace_len: int = particle_trace_len
        self.particle_mean: np.array = np.zeros(particle_dim, dtype=float)
        self.kernel_count: int = kernel_count
        self.kernel_params: np.array = np.zeros(
            (kernel_count, particle_dim), dtype=float
        )
        self.observed_data = observed_data

    def _init(self):
        sigma = self._get_sigma()
        self.kernel_params[0] = sigma
        for idx in range(0, self.particle_trace_len):
            particle = self._next_particle(sigma)
            llh = self.model.estimate_log_llh(particle, self.observed_data)
            weight = 1
            self.particle_trace[idx] = particle
            self.particle_weights[idx] = weight
            self.particle_llh[idx] = llh

    def _get_interval(self, sigma: Optional[float]) -> Tuple[float]:
        l, u = self.interval
        if sigma is not None:
            return self.interval
        new_l, new_u = (l - sigma, u + sigma)
        new_l = l if new_l < l else new_l
        new_u = u if new_u > u else new_u
        return (new_l, new_u)

    def _get_sigma(self):
        sigma = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            _min = np.amin(self.particle_trace[:, i])
            _max = np.amax(self.particle_trace[:, i])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _update_particle_by_idx(self, idx: int, particle: np.array, weight: float):
        assert len(particle) == self.particle_dim
        self.particle_trace[idx] = particle
        self.particle_weights[idx] = weight

    def _normalize_weight(self) -> np.array:
        return self.particle_weights / np.sum(self.particle_weights)

    def _next_particle(self, sigma: Optional[np.array]) -> np.array:
        assert len(sigma) == self.particle_dim
        particle = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def _correct(self, kernel_idx: int):
        sigma = self._get_sigma()
        last_sigma = self.kernel_params[kernel_idx]
        new_sigma = np.zeros(len(sigma), dtype=float)
        for i in range(0, len(sigma)):
            interval = self._get_interval(sigma[i])
            last_interval = self._get_interval(last_sigma[i])
            new_sigma[i] = np.abs(last_interval[0] - last_interval[1]) / np.abs(
                interval[0] - interval[1]
            )
        return new_sigma

    def _select(self):
        new_particles_idx = np.random.choice(
            self.particle_trace_len, self.particle_trace_len, replace=True
        )
        new_trace = np.copy(self.particle_trace)
        for idx in range(0, self.particle_trace_len):
            new_trace[idx] = self.particle_trace[new_particles_idx[idx]]
            self.particle_weights[idx] = 1
        return new_particles_idx

    def _pertubate(self):
        for idx in range(0, self.particle_trace_len):
            particle = self.particle_trace[idx]
            weight = self.particle_llh[idx]
            new_particle, _ = self._mh_transition(particle, weight)
            self._update_particle_by_idx(idx, particle=new_particle, weight=1)

    def _estimate_point(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        normalized_weight = self._normalize_weight()
        for i in range(0, self.particle_trace_len):
            particle += self.particle_trace[i] * normalized_weight[i]
        self.particle_mean = particle

    def run(self):
        self._init()
        print(self.particle_trace)
        for t in range(1, self.kernel_count):
            # Correct
            self._correct(t)
            # Select
            self._select()
            # Mutation
            self._pertubate()
        self._estimate_point()

    def _mh_init(
        self,
        particle: np.array,
        weight: float,
    ):
        mh_particle_trace: np.array = np.zeros(
            (self.particle_mh_trace_len, self.particle_dim), dtype=float
        )
        mh_particle_weights: np.array = np.zeros(self.particle_mh_trace_len)
        mh_particle_trace[0] = particle
        mh_particle_weights[0] = weight
        return mh_particle_trace, mh_particle_weights

    def _mh_get_sigma(self, particle_idx: int, particle_trace: np.array):
        sigma = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            _min = np.amin(particle_trace[0 : particle_idx + 1, i])
            _max = np.amax(particle_trace[0 : particle_idx + 1, i])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _mh_next_particle(
        self, mh_particle_idx: int, mh_particle_trace: np.array
    ) -> np.array:
        sigma = self._mh_get_sigma(mh_particle_idx, mh_particle_trace)
        particle = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma)
            particle[i] = np.random.uniform(*interval)
        return particle

    def _mh_transition(
        self,
        particle: np.array,
        weight: float,
    ) -> Tuple[np.array, float]:
        mh_particle_trace, mh_particle_weights = self._mh_init(particle, weight)
        mh_particle_idx = 1
        while mh_particle_idx < self.particle_mh_trace_len:
            last_log_llh = mh_particle_weights[mh_particle_idx]
            candidate_particle = self._mh_next_particle(
                mh_particle_idx, mh_particle_trace
            )
            candidate_sat = self.model.check_bounded(candidate_particle)
            if not candidate_sat:
                continue
            y_sim = self.model.simulate(
                candidate_particle, 100 * len(self.observed_data)
            )
            candidate_distance = self.model.estimate_distance(
                self._to_stats_summary(y_sim),
                self._to_stats_summary(self.observed_data),
            )
            acceptance_rate = np.min([0, candidate_log_llh - last_log_llh])
            acceptance_rate = np.exp(acceptance_rate)
            u = np.random.uniform(0, 1)
            if u < acceptance_rate:
                mh_particle_trace[mh_particle_idx] = candidate_particle
                mh_particle_weights[mh_particle_idx] = candidate_log_llh
                mh_particle_idx += 1
            else:
                acceptance_rate = 1e-1
                u = np.random.uniform(0, 1)
                if u < acceptance_rate:
                    mh_particle_trace[mh_particle_idx] = candidate_particle
                    mh_particle_weights[mh_particle_idx] = candidate_log_llh
                    mh_particle_idx += 1
        mh_particle_idx -= 1
        return mh_particle_trace[mh_particle_idx], mh_particle_weights[mh_particle_idx]

    def get_result(self):
        return (self.particle_mean, self.particle_trace, self.particle_weights)