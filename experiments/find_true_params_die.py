from datetime import datetime
import logging
from timeit import default_timer as timer
from typing import List, Tuple

logging.basicConfig(
    filename=f"temp-files/log/die_true_params_{str(datetime.now())}.log",
    level=logging.DEBUG,
)
import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel


class FindTrueParams(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
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
        self.interval = interval
        self.simulation_count = simulation_count
        self.true_params: List[np.array] = []

    def search(self):
        i: int = 0
        while i < 200:
            p = np.random.uniform(*self.interval)
            q = np.random.uniform(*self.interval)
            sat = self.model.check_bounded(np.array([p, q]))
            if sat:
                logging.info(f"{str(datetime.now())} True parameter (SAT): {[p, q]}")
                self.simulate(np.array([p, q]))
                self.true_params.append(np.array([p, q]))
                i += 1

    def simulate(self, p: np.array):
        self.synthetic_data = self.model.simulate(p, self.simulation_count)
        logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.search()
        logging.info(f"{self.true_params}")


def main():
    configs = [
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/die.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/die.pctl",
            (0, 1),
            10000,
        ),
    ]

    for cfg in configs:
        prism_model_file, prism_props_file, interval, simulation_count = cfg
        experiment = FindTrueParams(
            prism_model_file, prism_props_file, interval, simulation_count
        )
        experiment.exec()


def manual():
    cfg = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/die.pm",
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/die.pctl",
        (0, 1),
        10000,
    )
    prism_model_file, prism_props_file, interval, simulation_count = cfg
    experiment = FindTrueParams(
        prism_model_file, prism_props_file, interval, simulation_count
    )
    for rf in experiment.model.obs_rf:
        logging.info(rf)


if __name__ == "__main__":
    # main()
    manual()
    logging.shutdown()