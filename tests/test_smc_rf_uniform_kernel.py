import unittest

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel


class TestMhRfUniformKernel(unittest.TestCase):
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
        self.mc = SmcRfUniformKernel(
            model=self.model,
            interval=[0, 0.01],
            particle_dim=2,
            particle_trace_len=10,
            kernel_count=100,
            observed_data=[28, 12, 17, 43],
        )

    def test_exec(self):
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        print("Inference result: {}".format(particle_mean))
