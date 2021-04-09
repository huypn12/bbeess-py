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

    def search(self):
        i: int = 0
        while i < 10:
            alpha = np.random.uniform(*self.interval)
            beta = np.random.uniform(*self.interval)
            sat = self.model.check_bounded(np.array([alpha, beta]))
            if sat:
                logging.info(
                    f"{str(datetime.now())} True parameter (SAT): {[alpha, beta]}"
                )
                self.simulate(np.array([alpha, beta]))
                i += 1

    def simulate(self, p: np.array):
        self.synthetic_data = self.model.simulate(p, self.simulation_count)
        logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.search()


def manual():
    configs = [
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pctl",
            (0, 0.1),
            10000,
            np.array([0.03405521326944327, 0.08773454035144489]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pctl",
            (0, 0.1),
            200,
            np.array([0.03405521326944327, 0.08773454035144489]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pctl",
            (0, 0.1),
            10000,
            np.array([0.025490115891226895, 0.06929809986640066]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pctl",
            (0, 0.1),
            200,
            np.array([0.025490115891226895, 0.06929809986640066]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
            (0, 0.1),
            10000,
            np.array([0.011499276183591657, 0.06211051606863456]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
            (0, 0.1),
            200,
            np.array([0.011499276183591657, 0.06211051606863456]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0_a.pctl",
            (0, 0.1),
            10000,
            np.array([0.01061723587232546, 0.06099531949307238]),
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0_a.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0_a.pctl",
            (0, 0.1),
            200,
            np.array([0.01061723587232546, 0.06099531949307238]),
        ),
    ]
    for cfg in configs:
        prism_model_file, prism_props_file, interval, simulation_count, true_param = cfg
        experiment = FindTrueParams(
            prism_model_file, prism_props_file, interval, simulation_count
        )
        experiment.simulate(true_param)


def main():
    configs = [
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl",
            (0, 0.1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl",
            (0, 0.1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl",
            (0, 0.1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl",
            (0, 0.1),
            10000,
        ),
        (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0.pm",
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_20_1_0.pctl",
            (0, 0.1),
            10000,
        ),
    ]

    for cfg in configs:
        prism_model_file, prism_props_file, interval, simulation_count = cfg
        experiment = FindTrueParams(
            prism_model_file, prism_props_file, interval, simulation_count
        )
        experiment.exec()


if __name__ == "__main__":
    # main()
    manual()
    logging.shutdown()