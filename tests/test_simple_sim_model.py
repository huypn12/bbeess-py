import unittest

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel


class TestSimpleSimModel(unittest.TestCase):
    def setUp(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_310.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_310.pctl"
        )
        self.model = SimpleSimModel(
            self.prism_model_file,
            self.prism_props_file,
            ["bscc_300", "bscc_202", "bscc_103", "bscc_013"],
        )

    def test_check_bounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertFalse(result)

        true_params = np.array([0.002, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertTrue(result)

    def test_check_unbounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_unbounded(true_params)
        self.assertEqual(result, 0.530119534444794)

    def test_estimate_distance(self):
        true_params = np.array([0.005, 0.007])
        y_obs = np.array([20, 20, 20, 20])
        s_obs = y_obs * 1.0 / np.sum(y_obs)
        y_sim = self.model.simulate(true_params, 1000)
        s_sim = y_sim * 1.0 / np.sum(y_sim)
        result = self.model.estimate_distance(s_sim, s_obs)
        self.assertLess(result, 0.3)

    def test_simulate(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.simulate(true_params, 100)
        print(result)
        self.assertEqual(sum(result), 100)

        true_params = np.array([0.002, 0.007])
        result = self.model.simulate(true_params, 100)
        print(result)
        self.assertEqual(sum(result), 100)
