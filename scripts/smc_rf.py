from typing import List
import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp
from scipy.stats import multinomial as sp_multinomial


class MyPrismProgram(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__()
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)
        props_str = self._load_props_file()
        self.prism_props = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )

    def _load_props_file(self):
        lines = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        return ";".join(lines)


class SmcRf(object):
    """In case of likelihood can be cheaply calculated (closed form solution of the property is known)"""

    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        obs_data: List[int],
        particle_count: int,
        perturbation_count: int,
        check_threshold: float,
    ) -> None:
        super().__init__()
        self.rng: np.random.Generator = np.random.default_rng()
        # PRISM model configuration
        self.my_prism_program = MyPrismProgram(prism_model_file, prism_props_file)
        self.model = stormpy.build_parametric_model(
            self.my_prism_program.prism_program,
            self.my_prism_program.prism_props,
        )
        self.model_parameters = self.model.collect_probability_parameters()
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None
        # Properties for checking and observing
        self.check_prop = self.my_prism_program.prism_props[0]
        self.check_rf = stormpy.model_checking(self.model, self.check_prop).at(
            self.model.initial_states[0]
        )
        self.obs_props = self.my_prism_program.prism_props[1:]
        self.obs_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.obs_props
        ]
        self.obs_data = obs_data
        assert len(self.obs_data) == len(self.obs_rf)
        # SMC configuration
        self.param_space_sample: List = []
        self.particle_count: int = particle_count
        self.perturbation_count: int = perturbation_count
        self.check_threshold: int = check_threshold
        self.current_param = np.zeros(len(self.model_parameters), dtype=np.float)

    def _instantiate_pmodel(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        self.instantiated_model = self.instantiator.instantiate(point)

    def _init(self):
        sample_size = len(self.current_param)
        sampled_param = np.random.uniform(0, 1, sample_size)
        self.current_param = sampled_param
        self._instantiate_pmodel(sampled_param)

    def _perturbate(self, param) -> np.array:
        # TODO: properly designed perturbation function; proof of convergence/KL distance
        new_param = np.zeros(len(param))
        for i, p_i in enumerate(param):
            alpha = beta = 1 / p_i
            new_param[i] = self.rng.beta(alpha, beta)
        return new_param

    def _smc(self):
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

    def _compute_llh_multinomial(self, P, data):
        N = sum(data)
        return np.sum(np.log(sp_multinomial(N, P).pmf(data)))

    def _estimate_obs_llh(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        P = [float(rf.evaluate(point)) for rf in self.obs_rf]
        return self._compute_llh_multinomial(P, self.obs_data)

    def _estimate_check_llh(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        return float(self.check_rf.evaluate(point))

    def _is_candidate_params_valid(self, p: np.array):
        for _p in p:
            if _p < 0 or _p > 1:
                return False
        return True

    def run(self):
        self._init()
        self._smc()
        print(self.param_space_sample)


def main():
    prism_model_file = (
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/zeroconf.pm"
    )
    prism_props_file = (
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/zeroconf.pctl"
    )
    smc_rf = SmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_data=[30, 60],
        particle_count=100,
        perturbation_count=10,
        check_threshold=0.0,
    )
    smc_rf.run()


if __name__ == "__main__":
    main()