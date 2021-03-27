import unittest

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel


class TestSmcSimUniformKernel(unittest.TestCase):
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
            obs_labels=["bscc_300", "bscc_202", "bscc_103", "bscc_013"],
        )
        self.mc = AbcSmcSmcUniformKernel(
            model=self.model,
            interval=[0, 0.01],
            particle_dim=2,
            particle_trace_len=10,
            kernel_count=10,
            abc_threshold=0.1,
            observed_data=[51, 20, 17, 12],
        )

    def test_exec(self):
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        print("Inference result: {}".format(particle_mean))
