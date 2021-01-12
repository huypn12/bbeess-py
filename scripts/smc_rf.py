from typing import List
import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp


class SmcRf(object):
    """In case of likelihood can be cheaply calculated (closed form solution of the property is known)"""

    def __init__(
        self,
        prism_model_file_path: str,
        verify_property: str,
        observe_properties: List[str],
        particle_count: int,
        pertubation_count: int,
        verify_threshold: float,
    ) -> None:
        super().__init__()
        # PRISM model and one PCTL property
        self.prism_program = stormpy.parse_prism_program(prism_model_file_path)
        props_str = ";".join([verify_property] + observe_properties)
        props = stormpy.parse_properties_for_prism_program(
            props_str,
            self.prism_program,
        )
        self.model = stormpy.build_parametric_model(self.prism_program, props)
        self.model_parameters = self.model.collect_probability_parameters()
        self.verify_property = props[0]
        self.verify_rf = stormpy.model_checking(self.model, self.verify_property).at(
            self.model.initial_states[0]
        )
        self.obs_props = props[1:]
        self.obs_props_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.observe_properties
        ]
        self.current_param_values = np.array(len(self.model_parameters), dtype=np.float)
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None

        # SMC configuration
        self.param_space_sample: List = []
        self.particle_count: int = particle_count
        self.pertubation_count: int = pertubation_count
        self.verify_threshold: int = verify_threshold

    def _instantiate_pmodel(self, params: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = params[i]
        self.instantiated_model = self.instantiator.instantiate(point)

    def _init(self):
        sample_size = len(self.current_param_values)
        sampled_params = np.random.uniform(0, 1, sample_size)
        self.current_param_values = sampled_params
        self._instantiate_pmodel(sampled_params)

    def _perturbate(self, param) -> np.array:
        """Draw new parameter from Normal distribution"""
        # TODO: properly designed perturbation function; proof of convergence/KL distance
        for i, p_i in enumerate(param):
            alpha = beta = 1 / p_i
            param[i] = self.rng.beta(alpha, beta)
        return param

    def _smc(self):
        # Accepted point: points which satisfy the properties
        # Weight: likelihood to generate the observed data
        for m in range(0, self.particle_count):
            for _ in range(0, self.pertubation_count):
                candidate_params: np.array = None
                if m == 0:
                    param_dim = len(self.current_param_values)
                    candidate_params = np.random.uniform(0, 1, param_dim)
                else:
                    candidate_params = self._perturbate(self.current_param_values)
                if not self._is_candidate_params_valid(candidate_params):
                    continue
                llh_obs = self._estimate_obs_llh(candidate_params)
                llh_prop = self._estimate_prop_llh(candidate_params)
                if llh_prop < self.threshold:
                    continue
                self.trace.append({candidate_params: llh_obs})

    def _estimate_obs_llh(self, params: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = params[i]
        return [float(rf.evaluate(point)) for rf in self.obs_props_rf]

    def _estimate_sat_prob(self, params: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = params[i]
        return float(self.verify_rf.evaluate(point))

    def _is_candidate_params_valid(self, p: np.array):
        for _p in p:
            if _p < 0 or _p > 1:
                return False
        return True

    def run(self):
        self._init()
        self._smc()