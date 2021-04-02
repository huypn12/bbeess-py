import unittest

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel


class TestSimpleSimModel(unittest.TestCase):
    def setUp(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl"
        )
        self.model = SimpleSimModel(
            self.prism_model_file,
            self.prism_props_file,
            ["bscc_0_0_4", "bscc_1_0_3", "bscc_2_0_2", "bscc_3_0_1"],
        )

    def test_check_bounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertTrue(result)

        true_params = np.array([0.002, 0.007])
        result = self.model.check_bounded(true_params)
        self.assertTrue(result)

    def test_check_unbounded(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.check_unbounded(true_params)
        self.assertEqual(result, 0.3000818123078822)

    def test_estimate_distance(self):
        true_params = np.array([0.005, 0.007])
        y_obs = np.array([3181, 1990, 1602, 3227])
        s_obs = y_obs * 1.0 / np.sum(y_obs)
        y_sim = self.model.simulate(true_params, 10000)
        s_sim = y_sim * 1.0 / np.sum(y_sim)
        result = self.model.estimate_distance(s_sim, s_obs)
        self.assertLess(result, 0.1)

    def test_simulate(self):
        true_params = np.array([0.005, 0.007])
        result = self.model.simulate(true_params, 10000)
        print(result)
        self.assertEqual(sum(result), 10000)

        true_params = np.array([0.002, 0.007])
        result = self.model.simulate(true_params, 10000)
        print(result)
        self.assertEqual(sum(result), 10000)
