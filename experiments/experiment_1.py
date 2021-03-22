import logging

logging.basicConfig(filename="experiment_1.log", level=logging.DEBUG)

import numpy as np

from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel


class Experiment01:
    def __init__(self):
        self.prism_model_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm"
        )
        self.prism_props_file = (
            "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl"
        )
        self.model = SimpleRfModel(
            self.prism_model_file,
            self.prism_props_file,
        )
        self.true_param = np.array([0.002, 0.007])
        result = self.model.check_bounded(self.true_param)
        assert result is True
        print("DONE INIT")

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


if __name__ == "__main__":
    experiment = Experiment01()
    experiment.exec()
    logging.shutdown()