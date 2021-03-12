import unittest

import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import stormpy.examples
import stormpy.examples.files
import stormpy._config as config


class TestStormPars(unittest.TestCase):
    def setUp(self):
        if not config.storm_with_pars:
            raise AssertionError(
                "Support parameters is missing. Try building storm-pars."
            )

        path = stormpy.examples.files.prism_pdtmc_die
        prism_program = stormpy.parse_prism_program(path)
        unbounded_formula_str = "P=? [F s=7 & d=2]"
        bounded_formula_str = "P>=0.7 [F s=7 & d=2]"
        self.properties = stormpy.parse_properties_for_prism_program(
            ";".join([unbounded_formula_str, bounded_formula_str]), prism_program
        )
        self.model = stormpy.build_parametric_model(prism_program, self.properties)
        parameters = self.model.collect_probability_parameters()
        self.assertEqual(len(parameters), 2)

    def test_rf_evaluation(self):
        point = dict()
        for x in self.model.collect_probability_parameters():
            point[x] = pycarl.cln.cln.Rational(0.4)
        initial_state = self.model.initial_states[0]
        result = stormpy.model_checking(self.model, self.properties[0])
        rf = result.at(initial_state)
        self.assertEqual(str(rf), "((p)^2 * (q+(-1)))/(p*q+(-1))")
        self.assertEqual(float(rf.evaluate(point)), 0.1142857142857143)

    def test_unbounded_parametric_checking(self):
        point = dict()
        for x in self.model.collect_probability_parameters():
            point[x] = pycarl.cln.cln.Rational(0.4)
        instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        instantiated_model = instantiator.instantiate(point)
        result = stormpy.model_checking(instantiated_model, self.properties[0])
        initial_state = self.model.initial_states[0]
        self.assertEqual(result.at(initial_state), 0.11428571428571428)

    def test_bounded_parametric_checking(self):
        point = dict()
        for x in self.model.collect_probability_parameters():
            point[x] = pycarl.cln.cln.Rational(0.4)
        instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        instantiated_model = instantiator.instantiate(point)
        result = stormpy.model_checking(instantiated_model, self.properties[1])
        initial_state = self.model.initial_states[0]
        self.assertFalse(result.at(initial_state))