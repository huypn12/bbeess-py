from typing import Tuple, List

import sys
import logging
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel


logging.basicConfig(
    filename=f"experiment_sir_smc_rf_{str(datetime.time(datetime.now()))}.log",
    level=logging.DEBUG,
)


class ExperimentSirSmcRf:
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        interval: Tuple[float, float],
        true_param: np.array,
        observed_data: np.array,
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        self.interval = interval
        self.true_param = true_param
        self.observed_data = observed_data
        logging.info(f"{str(datetime.now())} Start experiments")
        print("interval: ", interval)
        is_sat = self.model.check_bounded(self.true_param)
        assert is_sat == True
        logging.info(f"{str(datetime.now())} True parameter (SAT): {self.true_param}")
        logging.info(f"{str(datetime.now())} Synthetic data: {self.observed_data}")
        self.summary = {
            "prism_model_file": self.prism_model_file,
            "prism_props_file": self.prism_props_file,
            "interval": self.interval,
            "true_param": self.true_param,
            "observed_data": self.observed_data,
        }

    def exec(self):
        start_time = datetime.now()
        self.mc = SmcRfUniformKernel(
            model=self.model,
            interval=self.interval,
            particle_dim=2,
            particle_trace_len=200,
            kernel_count=20,
            observed_data=self.observed_data,
        )
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        end_time = datetime.now()
        logging.info(f"{str(datetime.now())} Particle trace")
        logging.info(trace)
        logging.info(f"{str(datetime.now())} Particle weights")
        logging.info(weights)
        logging.info(
            f"{str(datetime.now())} Particle mean, distance={np.linalg.norm(particle_mean - self.true_param)}"
        )
        logging.info(particle_mean)
        logging.info(f"{str(datetime.now())} Time elapsed {end_time - start_time}")
        logging.info(f"{str(datetime.now())} End experiments.")


def do_experiment_sir5():
    prism_model_file: str = "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm"
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl"
    )
    experiment = ExperimentSirSmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.017246491978609703, 0.067786043574277]),
        observed_data=np.array([421, 834, 1126, 1362, 1851, 4406]),
    )
    experiment.exec()


def do_experiment_sir10():
    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl"
    )
    experiment = ExperimentSirSmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.01099297054879006, 0.035355703902286616]),
        observed_data=np.array(
            [563, 976, 1016, 909, 764, 696, 606, 565, 603, 855, 2447]
        ),
    )
    experiment.exec()


def do_experiment_sir15():
    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl"
    )
    experiment = ExperimentSirSmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.004132720013578173, 0.07217656035559976]),
        observed_data=np.array(
            [0, 0, 0, 1, 4, 12, 29, 37, 96, 177, 283, 459, 619, 1078, 1845, 5360]
        ),
    )
    experiment.exec()


def do_all_experiments():
    do_experiment_sir5()
    do_experiment_sir10()
    do_experiment_sir15()


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "5":
        do_experiment_sir5()

    elif m == "10":
        do_experiment_sir10()
    elif m == "15":
        do_experiment_sir15()
    else:
        raise ValueError(f"Unsupported option {m}")
    logging.shutdown()