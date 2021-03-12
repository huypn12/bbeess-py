import unittest

from scripts.mc.mh_rf_uniform_kernel import MhRfUniformKernel


class TestMhRfUniformKernel(unittest.TestCase):
    def setUp(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/experiments/data/sir_310.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/experiments/data/sir_310.pctl"
        )