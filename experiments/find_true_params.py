from datetime import datetime
import logging
from timeit import default_timer as timer
from typing import List, Tuple

logging.basicConfig(
    filename=f"temp-files/log/find_true_params_{str(datetime.now())}.log",
    level=logging.DEBUG,
)

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.model.simple_prism_rf_model import SimpleRfModel


class FindTrueParams(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        interval: Tuple[float, float],
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        self.interval = interval
        self.true_params: List[np.array] = []

    def search(self):
        while len(self.true_params) < 10:
            alpha = np.random.uniform(*self.interval)
            beta = np.random.uniform(*self.interval)
            sat = self.model.check_bounded(np.array([alpha, beta]))
            if sat:
                self.true_params.append(np.array([alpha, beta]))
                logging.info(
                    f"{str(datetime.now())} True parameter (SAT): {[alpha, beta]}"
                )

    def simulate(self):
        for p in self.true_params:
            logging.info(f"{str(datetime.now())} Synthesizing data")
            self.synthetic_data = self.model.simulate(p, 10000)
            logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.search()
        self.simulate()


def main():
    prism_model_file: str = "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm"
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl"
    )
    experiment = FindTrueParams(prism_model_file, prism_props_file, (0, 0.1))
    experiment.exec()

    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl"
    )
    experiment = FindTrueParams(prism_model_file, prism_props_file, (0, 0.1))
    experiment.exec()

    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl"
    )
    experiment = FindTrueParams(prism_model_file, prism_props_file, (0, 0.1))
    experiment.exec()


if __name__ == "__main__":
    main()