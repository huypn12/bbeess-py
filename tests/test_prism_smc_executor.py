import unittest

from scripts.prism.prism_smc_executor import PrismSmcSprtExecutor


class TestPrismSmcSprtExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/experiments/data/sir_310.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/experiments/data/sir_310.pctl"
        )
        self.prism_sprt_executor = PrismSmcSprtExecutor(
            prism_model_file=self.prism_model_file,
            prism_props_file=self.prism_props_file,
        )
        self.prism_apmc_executor = PrismSmcSprtExecutor(
            prism_model_file=self.prism_model_file,
            prism_props_file=self.prism_props_file,
        )

    def test_sprt_sir310(self):
        result = self.prism_sprt_executor.exec(model_consts_str="pa=1,pr=2")
        self.assertFalse(result)

    def test_apmc_sir310(self):
        result = self.prism_apmc_executor.exec(model_consts_str="pa=1,pr=2")
        self.assertFalse(result)