import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class AbcSmc(object):
    def __init__(self) -> None:
        super().__init__()
        self.mc_trace = []

        path = stormpy.examples.files.prism_pdtmc_die
        prism_program = stormpy.parse_prism_program(path)
        formula_str = "P=? [F s=7 & d=2]"
        properties = stormpy.parse_properties_for_prism_program(
            formula_str, prism_program
        )
        property = properties[0]
        model = stormpy.build_parametric_model(prism_program, properties)
        print("Model supports parameters: {}".format(model.supports_parameters))
        parameters = model.collect_probability_parameters()
        assert len(parameters) == 2

        instantiator = stormpy.pars.PDtmcInstantiator(model)
        point = dict()
        for x in parameters:
            print(x.name)
            point[x] = stormpy.RationalRF(0.4)
        instantiated_model = instantiator.instantiate(point)
        result = stormpy.model_checking(instantiated_model, property)
        print(result)

    def _instantiate_pmodel(self):
        pass

    def _init(self):
        # Sample model parameters from Uniform(0,1)
        pass

    def _perturbate(self):
        # Draw new parameter from Normal distribution
        pass

    def _transition(self):
        pass

    def _abc(self):
        pass

    def _smc(self):
        pass

    def _estimate_likelihood(self):
        pass

    def run(self):
        pass