from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.model.abstract_model import AbstractSimulationModel


class AbcMcSmcUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractSimulationModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        observed_data: List[int],
        abc_threshold: float,
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

    def _init(self):
        self.particle_curr_idx = 0
        first_particle = self._next_particle()
        y_sim = self.model.simulate(first_particle, 100 * len(self.observed_data))
        distance = self.model.estimate_distance(
            self._to_stats_summary(y_sim),
            self._to_stats_summary(self.observed_data),
        )
        self._update_particle_by_idx(0, first_particle, distance)

    def _get_particle_by_idx(self, idx: int) -> np.array:
        return self.particle_trace[idx]

    def _to_stats_summary(self, cat: np.array) -> np.array:
        return cat / np.sum(cat)

    def _update_particle_by_idx(self, idx: int, particle: np.array, weight: float):
        self.particle_trace[idx] = particle
        self.particle_weights[idx] = weight

    def _append_particle(self, particle: np.array, weight: float):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle, weight)

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            particle[i] = np.random.uniform(*self.interval)
        return particle

    def run(self):
        self._init()
        for _ in range(1, self.particle_trace_len - 1):
            candidate_sat = False
            candidate_dist_sat = False
            distance = self.abc_threshold + 1
            while not (candidate_sat and candidate_dist_sat):
                candidate_particle = self._next_particle()
                candidate_sat = self.model.check_bounded(candidate_particle)
                if not candidate_sat:
                    continue
                y_sim = self.model.simulate(
                    candidate_particle, 100 * len(self.observed_data)
                )
                distance = self.model.estimate_distance(
                    self._to_stats_summary(y_sim),
                    self._to_stats_summary(self.observed_data),
                )
                candidate_dist_sat = distance < self.abc_threshold
                print(
                    "Particle {} sat {} , distance {}".format(
                        candidate_particle, candidate_sat, distance
                    )
                )
            self._append_particle(candidate_particle, distance)
            print(
                "Accepted: Particle {}, distance {}".format(
                    candidate_particle, distance
                )
            )
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