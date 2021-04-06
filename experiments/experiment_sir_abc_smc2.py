from datetime import datetime
import logging
from timeit import default_timer as timer
from typing import Tuple, List
import sys

logging.basicConfig(
    filename=f"experiment_sir_abc_smc2_{str(datetime.now())}.log",
    level=logging.DEBUG,
)

import numpy as np

from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel


class ExperimentSirAbcSmc2:
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        interval: Tuple[float],
        true_param: np.array,
        observed_data: np.array,
        obs_labels: List[str],
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        self.model = SimpleSimModel(
            self.prism_model_file, self.prism_props_file, obs_labels=obs_labels
        )
        self.interval = interval
        self.true_param = true_param
        self.observed_data = observed_data
        logging.info(f"{str(datetime.now())} Start experiments")
        self.model.check_bounded(self.true_param)
        logging.info(f"{str(datetime.now())} True parameter (SAT): {self.true_param}")
        logging.info(f"{str(datetime.now())} Synthetic data: {self.observed_data}")

    def simulate(self):
        self.synthetic_data = self.model.simulate(self.true_param, 10000)
        logging.info(f"{str(datetime.now())} Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.simulate()
        start = timer()
        self.mc = AbcSmcSmcUniformKernel(
            model=self.model,
            interval=self.interval,
            particle_dim=2,
            particle_trace_len=200,
            kernel_count=20,
            observed_data=self.observed_data,
            abc_threshold=0.2,
        )
        self.mc.run()
        end = timer()
        particle_mean, trace, weights = self.mc.get_result()
        logging.info(f"{str(datetime.now())} Particle trace")
        logging.info(trace)
        logging.info(f"{str(datetime.now())} Particle weights")
        logging.info(weights)
        logging.info(
            f"{str(datetime.now())} Particle mean, distance={np.linalg.norm(particle_mean - self.true_param)}"
        )
        logging.info(particle_mean)
        logging.info(f"{str(datetime.now())} Time elapsed {end - start}")
        logging.info(f"{str(datetime.now())} End experiments")


def do_experiment_sir5():
    prism_model_file: str = "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm"
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl"
    )
    experiment = ExperimentSirAbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.017246491978609703, 0.067786043574277]),
        observed_data=np.array([421, 834, 1126, 1362, 1851, 4406]),
        obs_labels=[
            "bscc_0_0_6",
            "bscc_1_0_5",
            "bscc_2_0_4",
            "bscc_3_0_3",
            "bscc_4_0_2",
            "bscc_5_0_1",
        ],
    )
    experiment.exec()


def do_experiment_sir10():
    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl"
    )
    experiment = ExperimentSirAbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.01099297054879006, 0.035355703902286616]),
        observed_data=np.array(
            [563, 976, 1016, 909, 764, 696, 606, 565, 603, 855, 2447]
        ),
        obs_labels=[
            "bscc_0_0_11",
            "bscc_1_0_10",
            "bscc_2_0_9",
            "bscc_3_0_8",
            "bscc_4_0_7",
            "bscc_5_0_6",
            "bscc_6_0_5",
            "bscc_7_0_4",
            "bscc_8_0_3",
            "bscc_9_0_2",
            "bscc_10_0_1",
        ],
    )
    experiment.exec()


def do_experiment_sir15():
    prism_model_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm"
    )
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl"
    )
    experiment = ExperimentSirAbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=(0, 0.1),
        true_param=np.array([0.004132720013578173, 0.07217656035559976]),
        observed_data=np.array(
            [0, 0, 0, 1, 4, 12, 29, 37, 96, 177, 283, 459, 619, 1078, 1845, 5360]
        ),
        obs_labels=[
            "bscc_0_0_16",
            "bscc_1_0_15",
            "bscc_2_0_14",
            "bscc_3_0_13",
            "bscc_4_0_12",
            "bscc_5_0_11",
            "bscc_6_0_10",
            "bscc_7_0_9",
            "bscc_8_0_8",
            "bscc_9_0_7",
            "bscc_10_0_6",
            "bscc_11_0_5",
            "bscc_12_0_4",
            "bscc_13_0_3",
            "bscc_14_0_2",
            "bscc_15_0_1",
        ],
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