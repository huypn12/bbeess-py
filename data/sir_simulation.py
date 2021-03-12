from typing import Tuple
import numpy as np

from scripts.prism.prism_model import AbstractModelRational
from scripts.mc.abc_smc2_uniform_kernel import AbcSmc2UniformKernel


class SirSimulationModel(AbstractModelRational):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        super().__init__(prism_model_file, prism_props_file)


class SirRationalSmc(object):
    def __init__(
        self,
        sir_model: SirSimulationModel,
        interval: Tuple[float, float],
        particle_dim: int,
        particle_count: int,
        kernel_count: int,
    ) -> None:
        self.sir_model = sir_model
        self.smc_uniform = AbcSmc2UniformKernel(
            sir_model,
            interval,
            particle_dim,
            particle_count,
            kernel_count,
        )

    def run(self):
        result = self.smc_uniform.sample()
