from scripts.my_prism_stat_mc import PrismSmcWrapper
import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class AbcSmc2(object):
    """ABC-SMC scheme with statistical model checking

    Args:
        object ([type]): [description]
    """

    def __init__(self) -> None:
        super().__init__()
        self._my_prism_stat_mc = PrismSmcWrapper()

    def _simulate(self, simulation_count: int):
        # implement ABC using L2-Distance
        return

    def _do_statmc(self):
        self._do_seqmc()

    def _do_seqmc(self):
        # Accepted point: points which satisfy the properties
        # Weight: likelihood to generate the observed data
        for m in range(0, self.particle_count):
            for _ in range(0, self.perturbation_count):
                candidate_param: np.array = None
                if m == 0:
                    param_dim = len(self.current_param)
                    candidate_param = np.random.uniform(0, 1, param_dim)
                else:
                    candidate_param = self._perturbate(self.current_param)
                if not self._is_candidate_params_valid(candidate_param):
                    continue
                llh_obs = self._estimate_obs_llh(candidate_param)
                llh_prop = self._estimate_check_llh(candidate_param)
                if llh_prop < self.check_threshold:
                    continue
                self.param_space_sample.append((candidate_param, llh_obs))