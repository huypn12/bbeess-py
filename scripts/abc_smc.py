import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class AbcSmc(object):
    def __init__(
        self,
        prism_model_file_path: str,
        verify_property_str: str,
        observe_property_str: str,
        particle_count: int,
        pertubation_count: int,
    ) -> None:
        super().__init__()
        # PRISM model and one PCTL property
        self.prism_program = stormpy.parse_prism_program(prism_model_file_path)
        properties = stormpy.parse_properties_for_prism_program(
            verify_property_str + ";" + observe_property_str,
            self.prism_program,
        )
        self.verify_property = properties[0]
        self.observe_property = properties[1]
        self.model = stormpy.build_parametric_model(self.prism_program, properties)
        self.model_parameters = self.model.collect_probability_parameters()
        self.current_param_values = np.array(len(self.model_parameters), dtype=np.float)
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None
        # ABC-SMC configuration
        self.param_space_sample = []
        self.particle_count = 1000
        self.pertubation_count = 1000

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

    def _perturbate(self):
        # Draw new parameter from Normal distribution
        pass

    def _distance(self):
        pass

    def _step(self):
        self._init()

    def _abc(self):
        pass

    def _smc(self):
        pass

    def _estimate_likelihood(self):
        pass

    def run(self):
        self._init()
