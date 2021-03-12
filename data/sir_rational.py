from typing import Tuple
import numpy as np

from scripts.prism.prism_model import AbstractPrismModelRational
from scripts.mc.smc_uniform_kernel import SmcUniformKernel


class SirRationalModel(AbstractPrismModelRational):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        super().__init__(prism_model_file, prism_props_file)


class SirRationalSmc(object):
    def __init__(
        self,
        sir_model: SirRationalModel,
        interval: Tuple[float, float],
        particle_dim: int,
        particle_count: int,
        kernel_count: int,
    ) -> None:
        self.sir_model = sir_model
        self.smc_uniform = SmcUniformKernel(
            sir_model,
            interval,
            particle_dim,
            particle_count,
            kernel_count,
        )

    def run(self):
        self.smc_uniform.sample()
