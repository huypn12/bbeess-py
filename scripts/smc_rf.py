from examples.storm_pars_api import example_parametric_models_01

from typing import List
import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class SmcRf(object):
    """In case of likelihood can be cheaply calculated (closed form solution of the property is known)"""

    def __init__(
        self,
        prism_model_file_path: str,
        verify_property: str,
        observe_properties: List[str],
        particle_count: int,
        pertubation_count: int,
    ) -> None:
        super().__init__()
        # PRISM model and one PCTL property
        self.prism_program = stormpy.parse_prism_program(prism_model_file_path)
        properties_str = ";".join([verify_property] + observe_properties)
        properties = stormpy.parse_properties_for_prism_program(
            properties_str,
            self.prism_program,
        )
        self.model = stormpy.build_parametric_model(self.prism_program, properties)
        self.model_parameters = self.model.collect_probability_parameters()
        self.verify_property = properties[0]
        self.verify_rf = stormpy.model_checking(self.model, self.verify_property).at(
            self.model.initial_states[0]
        )
        self.observe_properties = properties[1:]
        self.observe_properties_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.observe_properties
        ]
        self.current_param_values = np.array(len(self.model_parameters), dtype=np.float)
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None

        # ABC-SMC configuration
        self.param_space_sample: List = []
        self.particle_count: int = particle_count
        self.pertubation_count: int = pertubation_count

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

    @staticmethod
    def _distance(v1: np.array, v2: np.array) -> float:
        """Distance between two vector"""
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def _kernel(self, s1, s2, threshold):
        return 1 if (SmcRf._distance(s1, s2) > threshold) else 0

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
            for i in range(0, self.pertubation_count):
                candidate_params: np.array = None
                if m == 0:
                    param_dim = len(self.current_param_values)
                    candidate_params = np.random.uniform(0, 1, param_dim)
                else:
                    candidate_params = self._perturbate(self.current_param_values)
                # Reject candidate if prior(candidate) == 0
                if not self._is_candidate_params_valid(candidate_params):
                    continue

    def _estimate_likelihood(self, params: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = params[i]
        return float(self.verify_rf.evaluate(point))

    def _step(self):
        self._init()

    def _is_candidate_params_valid(self, p: np.array):
        for _p in p:
            if p < 0 or 1 < p:
                return False
        return True

    def run(self):
        pass