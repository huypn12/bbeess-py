from datetime import datetime
import logging
from timeit import default_timer as timer
from typing import List, Tuple

logging.basicConfig(
    filename=f"temp-files/log/jchain_true_params_{str(datetime.now())}.log",
    level=logging.DEBUG,
)
import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
import scripts.config as gCfg


class FindTrueParams(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        param_dim: int,
        interval: Tuple[float, float],
        simulation_counts: List[int],
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        logging.info(f"{str(datetime.now())} PRISM model: {prism_model_file}")
        logging.info(f"{str(datetime.now())} PRISM props: {prism_props_file}")
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        self.param_dim = param_dim
        self.interval = interval
        self.simulation_counts = simulation_counts

    def search(self):
        p = np.zeros(self.param_dim)
        i: int = 0
        while i < self.param_dim:
            for i in range(0, self.param_dim):
                p[i] = np.random.uniform(*self.interval)
            sat = self.model.check_bounded(p)
            if sat:
                logging.info(f"{str(datetime.now())} True parameter (SAT): {p}")
                for count in self.simulation_counts:
                    self.simulate(np.array(p), count)
                i += 1

    def simulate(self, p: np.array, simulation_count):
        self.synthetic_data = self.model.simulate(p, simulation_count)
        logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.search()


def main():
    gCfg.set_has_synthesis(False)
    configs = [
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/jchain.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/jchain.pctl",
            10,
            (0, 1),
            [10000, 200],
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/jchain2.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/jchain2.pctl",
            2,
            (0, 1),
            [10000, 200],
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pctl",
            2,
            (0, 1),
            [10000, 200],
        ),
    ]

    for cfg in configs:
        prism_model_file, prism_props_file, param_dim, interval, simulation_counts = cfg
        experiment = FindTrueParams(
            prism_model_file, prism_props_file, param_dim, interval, simulation_counts
        )
        for _ in range(0, 100):
            experiment.exec()


if __name__ == "__main__":
    main()
    # manual()
    logging.shutdown()