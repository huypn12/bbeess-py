import unittest

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_mh_smc_uniform_kernel import AbcMhSimUniformKernel


class TestAbcMhSimUniformKernel(unittest.TestCase):
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
            obs_labels=["bscc_0_0_4", "bscc_1_0_3", "bscc_2_0_2", "bscc_3_0_1"],
        )
        self.mc = AbcMhSimUniformKernel(
            model=self.model,
            interval=[0, 0.01],
            particle_dim=2,
            particle_trace_len=20,
            observed_data=[28, 12, 17, 43],
            abc_threshold=0.4,
        )

    def test_exec(self):
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        print("Inference result: {}".format(particle_mean))
