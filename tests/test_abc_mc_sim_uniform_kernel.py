import abc
import unittest

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_mc_smc_uniform_kernel import AbcMcSmcUniformKernel


class TestAbcMcSimUniformKernel(unittest.TestCase):
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
            obs_labels=["bscc_300", "bscc_202", "bscc_103", "bscc_013"],
        )
        self.mc = AbcMcSmcUniformKernel(
            model=self.model,
            interval=[0, 0.01],
            particle_dim=2,
            particle_trace_len=100,
            observed_data=[28, 12, 17, 43],
            abc_threshold=0.4,
        )

    def test_exec(self):
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        print("Inference result: {}".format(particle_mean))
