import unittest

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.mc_rf_uniform_kernel import McRfUniformKernel


class TestMcRfUniformKernel(unittest.TestCase):
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
        self.mc = McRfUniformKernel(
            model=self.model,
            interval=[0, 0.01],
            particle_dim=2,
            particle_trace_len=1000,
            observed_data=[28, 12, 17, 43],
        )

    def test_exec(self):
        self.mc.run()
        trace, weights = self.mc.get_result()
        pass
