import unittest

from scripts.prism.prism_smc_executor import PrismSmcExecutor


class PrismSmcExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = PrismSmcExecutor()
