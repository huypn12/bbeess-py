import unittest

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel


class TestSimpleRfModel(unittest.TestCase):
    def setUp(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_310.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_310.pctl"
        )
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )

    def test_check_bounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertFalse(result)

    def test_check_unbounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_unbounded(true_params)
        self.assertEqual(result, 0.530119534444794)

    def test_estimate_log_llh(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.estimate_log_llh(true_params, [20, 20, 20, 20])
        self.assertEqual(result, -90.64469718700241)

    def test_simulate(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.simulate(true_params, 100)
        self.assertEqual(sum(result), 100)
