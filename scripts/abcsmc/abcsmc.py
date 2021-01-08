import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class DtmcModel(object):
    def __init__(self) -> None:
        super().__init__()


class AbcSmc(object):
    def __init__(self) -> None:
        super().__init__()
        self.model_params = np.array(5, dtype=np.float64)
        self.mc_trace = []

    def _init(self):
        # Sample model parameters from Uniform(0,1)
        pass

    def _perturbate(self):
        # Draw new parameter from Normal distribution
        pass

    def _transition(self):
        pass

    def _abc(self):
        pass

    def _smc(self):
        pass

    def _estimate_likelihood(self):
        pass

    def run(self):
        pass