from datetime import datetime
import logging
from timeit import default_timer as timer
from typing import List, Tuple

logging.basicConfig(
    filename=f"temp-files/log/bees_true_params_{str(datetime.now())}.log",
    level=logging.DEBUG,
)
import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel


class FindTrueParams(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        dim: int,
        interval: Tuple[float, float],
        simulation_count: int,
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        logging.info(f"{str(datetime.now())} PRISM model: {prism_model_file}")
        logging.info(f"{str(datetime.now())} PRISM props: {prism_props_file}")
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        for i, rf in enumerate(self.model.obs_rf):
            print(f"{i}: {rf}")
        self.dim = dim
        self.interval = interval
        self.simulation_count = simulation_count
        self.true_params: List[np.array] = []

    def search(self):
        i: int = 0
        while i < 1000:
            theta = np.zeros(self.dim)
            for ii in range(0, self.dim):
                theta[ii] = np.random.uniform(*self.interval)
            theta = np.sort(theta)
            sat = self.model.check_bounded(theta)
            if sat:
                logging.info(f"{str(datetime.now())} True parameter (SAT): {theta}")
                self.simulate(theta)
                self.true_params.append(theta)
                i += 1

    def simulate(self, p: np.array):
        self.synthetic_data = self.model.simulate(p, self.simulation_count)
        logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.search()


def main():
    configs = [
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pctl",
            3,
            (0, 1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pctl",
            5,
            (0, 1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pctl",
            10,
            (0, 1),
            10000,
        ),
    ]

    for cfg in configs:
        prism_model_file, prism_props_file, dim, interval, simulation_count = cfg
        experiment = FindTrueParams(
            prism_model_file, prism_props_file, dim, interval, simulation_count
        )
        experiment.exec()


if __name__ == "__main__":
    main()
    # manual()
    logging.shutdown()