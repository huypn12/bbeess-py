import unittest

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel


class TestSimpleRfModel(unittest.TestCase):
    def setUp(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl"
        )
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )

    def test_check_bounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertTrue(result)

    def test_check_unbounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_unbounded(true_params)
        self.assertEqual(result, 0.3000818123078822)

    def test_estimate_log_llh(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.estimate_log_llh(true_params, [0, 25, 13, 62])
        self.assertEqual(result, -49.23976481740532)

    def test_simulate(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.simulate(true_params, 100)
        print(result)
        self.assertEqual(sum(result), 100)
