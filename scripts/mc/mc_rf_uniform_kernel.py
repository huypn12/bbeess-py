import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp

from scripts.model.abstract_model import AbstractRationalModel


class McRfUniformKernel(object):
    def __init__(
        self,
        model: Type[AbstractRationalModel],
        interval: Tuple[float],
        particle_dim: int,
        particle_trace_len: int,
        observed_data: List[int],
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
        self.observed_data = observed_data

    def _init(self):
        self.particle_curr_idx = 0
        first_particle = self._next_particle()
        self._update_particle_by_idx(0, first_particle)

    def _get_particle_by_idx(self, idx: int) -> np.array:
        return self.particle_trace[idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array):
        self.particle_trace[idx] = particle
        self.particle_weights[idx] = self.model.estimate_log_llh(
            particle, self.observed_data
        )

    def _append_particle(self, particle: np.array):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle)

    def _next_particle(self) -> np.array:
        particle = np.zeros(self.particle_dim)
        for i in range(0, self.particle_dim):
            particle[i] = np.random.uniform(*self.interval)
        return particle

    def run(self):
        self._init()
        for _ in range(1, self.particle_trace_len - 1):
            candidate_sat = False
            while not candidate_sat:
                candidate_particle = self._next_particle()
                candidate_sat = self.model.check_bounded(candidate_particle)
            self._append_particle(candidate_particle)

    def get_result(self):
        return (self.particle_trace, self.particle_weights)