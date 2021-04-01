import abc
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Type
import logging

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.model.abstract_model import AbstractSimulationModel


class AbcSmcSmcUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractSimulationModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        kernel_count: int,
        abc_threshold: float,
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
        self.particle_mean: np.array = np.zeros(particle_dim, dtype=float)
        self.kernel_count: int = kernel_count
        self.kernel_params: np.array = np.zeros(
            (kernel_count, particle_dim), dtype=float
        )
        self.abc_threshold: float = abc_threshold
        self.observed_data = observed_data

    def _init(self):
        sigma = self._get_sigma()
        self.kernel_params[0] = sigma
        for idx in range(0, self.particle_trace_len):
            particle = self._next_particle(sigma)
            y_sim = self.model.simulate(particle, 1000 * len(self.observed_data))
            distance = self.model.estimate_distance(
                self._average(y_sim),
                self._average(self.observed_data),
            )
            self.particle_trace[idx] = particle
            self.particle_weights[idx] = distance

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

    def _normalize_weight(self) -> np.array:
        epsilon = 1e-6
        for i in range(0, len(self.particle_weights)):
            self.particle_weights[i] = (
                1 / self.particle_weights[i]
                if self.particle_weights[i] > epsilon
                else 1 / (self.particle_weights[i] + epsilon)
            )
        return self._average(self.particle_weights)

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
            sigma = self._get_sigma()
            candidate_found = False
            while not candidate_found:
                candidate_particle = self._next_particle(sigma)
                candidate_sat = self.model.check_bounded(candidate_particle)
                if not candidate_sat:
                    continue
                y_sim = self.model.simulate(
                    candidate_particle, 1000 * len(self.observed_data)
                )
                candidate_distance = self.model.estimate_distance(
                    self._average(y_sim),
                    self._average(self.observed_data),
                )
                if candidate_distance < self.abc_threshold:
                    candidate_found = True
                    self.particle_trace[idx] = candidate_particle
                    self.particle_weights[idx] = candidate_distance

    def _estimate_point(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        normalized_weight = self._normalize_weight()
        for i in range(0, self.particle_trace_len):
            particle += self.particle_trace[i] * normalized_weight[i]
        self.particle_mean = particle

    def run(self):
        self._init()
        for t in range(1, self.kernel_count):
            logging.info(
                f"{str(datetime.now())} Start kernel {t} threshold={self.abc_threshold}"
            )
            # Correct
            self._correct(t)
            # Select
            self._select()
            # Mutation
            self._pertubate()
            # Logging
            logging.info(
                f"{str(datetime.now())} Finish kernel {t} threshold={self.abc_threshold}"
            )
        self._estimate_point()

    def _average(self, cat: np.array) -> np.array:
        return cat / np.sum(cat)

    def get_result(self):
        return (self.particle_mean, self.particle_trace, self.particle_weights)