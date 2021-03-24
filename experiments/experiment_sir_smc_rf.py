import datetime
import logging

logging.basicConfig(
    filename=f"experiment_sir_smc_rf_{str(datetime.time(datetime.now()))}.log",
    level=logging.DEBUG,
)

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel


class ExperimentSirSmcRf:
    def __init__(self, prism_model_file: str, prism_props_file: str):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        logging.info(">>>>> Start experiments")
        self.true_param = np.array([0.002, 0.007])
        result = self.model.check_bounded(self.true_param)
        assert result is True
        logging.info(f"True parameter (SAT): {self.true_param}")

    def simulate(self):
        self.synthetic_data = self.model.simulate(self.true_param, 2000)
        logging.info(f"Synthetic data: {self.synthetic_data}")

    def exec(self):
        self.simulate()
        self.mc = SmcRfUniformKernel(
            model=self.model,
            interval=[0, 0.1],
            particle_dim=2,
            particle_trace_len=200,
            kernel_count=25,
            observed_data=self.synthetic_data,
        )
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        logging.info("PARTICLE MEAN")
        logging.info(particle_mean)
        logging.info("PARTICLE TRACE")
        logging.info(trace)
        logging.info("PARTICLE WEIGHTS")
        logging.info(weights)
        logging.info(">>>>> End experiments")


def do_experiment_sir5():
    prism_model_file: str = "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm"
    prism_props_file: str = (
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl"
    )
    experiment = ExperimentSirSmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
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
    )
    experiment.exec()


def do_all_experiments():
    do_experiment_sir5()
    do_experiment_sir10()
    do_experiment_sir15()


if __name__ == "__main__":
    do_all_experiments()
    logging.shutdown()