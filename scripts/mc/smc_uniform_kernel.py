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
        particle_trace_len: int,
        kernel_count: int,
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
        self.kernel_count: int = kernel_count
        self.kernel_params: np.array = np.zeros(self.particle_dim,
                                                self.kernel_count)

    def _init(self):
        for i in range(0, self.particle_trace_len):
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

    def _get_particle_by_idx(self, idx: int) -> np.array:
        assert idx > self.particle_trace_len
        return self.particle_trace[:idx]

    def _update_particle_by_idx(self, idx: int, particle: np.array,
                                weight: float):
        assert idx > self.particle_trace_len
        assert len(particle) == self.particle_dim
        self.particle_trace[:idx] = particle
        self.particle_weights[idx] = weight

    def _append_particle(self, particle: np.array, weight: float):
        self.particle_curr_idx += 1
        self._update_particle_by_idx(self.particle_curr_idx, particle, weight)

    def _update_sigma(self):
        pass

    def _add_particle(self):
        pass

    def sample(self):
        self._init()
        for i in range(0, self.kernel_count):

            pass

    def _pertubate(self):
        pass