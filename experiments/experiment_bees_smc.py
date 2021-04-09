from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel

import numpy as np

from datetime import datetime
from timeit import default_timer as timer
from typing import Tuple, List
import sys
import logging

logging.basicConfig(
    filename=f"experiment_sir_abc_smc2_{str(datetime.now())}.log",
    level=logging.DEBUG,
)


class ExperimentBeesSmc(object):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    m = sys.argv[1]
    logging.shutdown()