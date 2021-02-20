import unittest

import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import stormpy.examples
import stormpy.examples.files
import stormpy._config as config


class TestStormPars(unittest):
    def setUp(self):
        if not config.storm_with_pars:
            raise AssertionError("Support parameters is missing. Try building storm-pars.")

        path = stormpy.examples.files.prism_pdtmc_die
        self.prism_program = stormpy.parse_prism_program(path)
        formula_str = "P=? [F s=7 & d=2]"
        properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
        model = stormpy.build_parametric_model(prism_program, properties)



def example_parametric_models_01():
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return

    

    
    print("Model initial states: {}".format(model.initial_states))
    print("Model supports parameters: {}".format(model.supports_parameters))
    parameters = model.collect_probability_parameters()
    assert len(parameters) == 2
    point = dict()
    for x in parameters:
        print(x.name)
        point[x] = pycarl.cln.cln.Rational(0.4)
    initial_state = model.initial_states[0]
    result = stormpy.model_checking(model, properties[0])
    rf = result.at(initial_state)
    print("Rational function of the desired property {}".format(rf))
    print(float(rf.evaluate(point)))

    instantiator = stormpy.pars.PDtmcInstantiator(model)
    instantiated_model = instantiator.instantiate(point)
    result = stormpy.model_checking(instantiated_model, properties[0])
    print(result.at(0))


if __name__ == "__main__":
    example_parametric_models_01()