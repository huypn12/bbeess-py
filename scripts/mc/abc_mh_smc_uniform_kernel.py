import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.model.abstract_model import AbstractSimulationModel


class AbcMhSimUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractSimulationModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        observed_data: List[int],
        abc_threshold: float,
        use_sigma: bool = True,
    ) -> None:
        self.model = model
        self.interval = interval
        self.particle_dim = particle_dim
        self.particle_trace_len = particle_trace_len
        self.particle_trace: np.array = np.zeros(
            (particle_trace_len, particle_dim), dtype=float
        )
        self.particle_curr_idx: int = -1
        self.particle_weights: np.array = np.zeros(particle_trace_len)
        self.particle_mean: np.array = np.zeros(particle_dim)
        self.observed_data = observed_data
        self.abc_threshold = abc_threshold
        self.use_sigma = use_sigma

    def _init(self):
        self.particle_curr_idx = 0
        first_particle = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            first_particle[i] = np.random.uniform(*self.interval)
        y_sim = self.model.simulate(first_particle, 100 * len(self.observed_data))
        distance = self.model.estimate_distance(
            self._to_stats_summary(y_sim),
            self._to_stats_summary(self.observed_data),
        )
        self._update_particle_by_idx(0, first_particle, distance)
        self.particle_curr_idx = 1

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
        return self.particle_trace[idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array, weight: float):
        assert idx < self.particle_trace_len
        assert len(particle) == self.particle_dim
        self.particle_trace[idx] = particle
        self.particle_weights[idx] = weight

    def _to_stats_summary(self, cat: np.array) -> np.array:
        return cat / np.sum(cat)

    def _append_particle(self, particle: np.array, weight: float):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle, weight)

    def _get_sigma(self, particle_idx: int) -> Optional[np.array]:
        sigma = np.zeros(self.particle_dim)
        if not self.use_sigma:
            return sigma
        for i in range(0, self.particle_dim):
            _min = np.amin(self.particle_trace[0 : particle_idx + 1, i])
            _max = np.amax(self.particle_trace[0 : particle_idx + 1, i])
            sigma[i] = 0.5 * (_max - _min)
        return sigma

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        sigma = self._get_sigma(self.particle_curr_idx)
        for i in range(0, self.particle_dim):
            interval = self._get_interval(sigma[i])
            particle[i] = np.random.uniform(*interval)
        return particle

    def run(self):
        self._init()
        for _ in range(1, self.particle_trace_len - 1):
            candidate_sat = False
            candidate_dist_sat = False
            candidate_distance = self.abc_threshold + 1
            while not (candidate_sat and candidate_dist_sat):
                candidate_particle = self._next_particle()
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
                print(
                    "Particle {} sat {} , distance {}".format(
                        candidate_particle, candidate_sat, candidate_distance
                    )
                )
                candidate_dist_sat = candidate_distance < self.abc_threshold
            candidate_q = np.prod(self._get_sigma(self.particle_curr_idx))
            last_q = np.prod(self._get_sigma(self.particle_curr_idx - 1))
            print("q1={} q2={}".format(last_q, candidate_q))
            acceptance_rate = np.min(np.array([1, last_q / candidate_q]))
            print("Acceptance rate {}".format(acceptance_rate))
            u = np.random.uniform(0, 1)
            if u < acceptance_rate:
                print(
                    "Accepted: Particle {}, distance {}".format(
                        candidate_particle, candidate_distance
                    )
                )
                self._append_particle(candidate_particle, candidate_distance)
            else:
                acceptance_rate = 1e-1
                u = np.random.uniform(0, 1)
                if u < acceptance_rate:
                    self._append_particle(candidate_particle, candidate_distance)
        self._estimate_point()

    def get_result(self):
        return (self.particle_mean, self.particle_trace, self.particle_weights)

    def _normalize_weight(self) -> np.array:
        return self.particle_weights / np.sum(self.particle_weights)

    def _estimate_point(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        normalized_weight = self._normalize_weight()
        for i in range(0, self.particle_trace_len):
            particle += self.particle_trace[i] * normalized_weight[i]
        self.particle_mean = particle