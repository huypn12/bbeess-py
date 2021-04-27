from experiments.experiment_sir_config import ExperimentSirConfig
from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel
from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel
import scripts.config as gCfg

import sys
from typing import Tuple, List
from enum import IntEnum
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(
    filename=f"temp-files/log/{str(datetime.time(datetime.now()))}_experiment_sir_smc.log",
    level=logging.DEBUG,
)


class EvaluationMode(IntEnum):
    RationalFunction = 1
    Simulation = 2


class ExperimentSirSmc(object):
    def __init__(
        self,
        mode: EvaluationMode,
        smc_trace_len: int,
        smc_pertubation_len: int,
        smc_mh_trace_len: int,
        abc_threshold: float,
        prism_model_file: str,
        prism_props_file: str,
        interval: Tuple[float, float],
        true_param: np.array,
        observed_data: np.array,
        observed_labels: List[str],
    ):
        self.mode = mode
        self.smc_trace_len = smc_trace_len
        self.smc_pertubation_len = smc_pertubation_len
        self.smc_mh_trace_len = smc_mh_trace_len
        self.abc_threshold = abc_threshold
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file
        self.interval = interval
        self.true_param = true_param
        self.observed_data = observed_data
        self.observed_labels = observed_labels

    def _init(self):
        if self.mode == EvaluationMode.RationalFunction:
            self.model = SimpleRfModel(
                self.prism_model_file,
                self.prism_props_file,
            )
            self.mc = SmcRfUniformKernel(
                model=self.model,
                interval=self.interval,
                particle_dim=2,
                particle_trace_len=self.smc_trace_len,
                kernel_count=self.smc_pertubation_len,
                mh_trace_len=self.smc_mh_trace_len,
                observed_data=self.observed_data,
            )
        elif self.mode == EvaluationMode.Simulation:
            self.model = SimpleSimModel(
                self.prism_model_file,
                self.prism_props_file,
                obs_labels=self.observed_labels,
            )
            self.mc = AbcSmcSmcUniformKernel(
                model=self.model,
                interval=self.interval,
                particle_dim=2,
                particle_trace_len=self.smc_trace_len,
                kernel_count=self.smc_pertubation_len,
                observed_data=self.observed_data,
                abc_threshold=self.abc_threshold,
            )
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
        self.model.check_bounded(self.true_param)
        logging.info(f"{str(datetime.now())} Mode {self.mode}")
        logging.info(f"{str(datetime.now())} PRISM model: {self.prism_model_file}")
        logging.info(f"{str(datetime.now())} PRISM props: {self.prism_props_file}")
        logging.info(f"{str(datetime.now())} True parameter (SAT): {self.true_param}")
        logging.info(f"{str(datetime.now())} Synthetic data: {self.observed_data}")

    def run(self):
        self._init()
        start_time = datetime.now()
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        end_time = datetime.now()
        logging.info(f"{str(datetime.now())} Particle trace")
        logging.info(trace)
        logging.info(f"{str(datetime.now())} Particle weights")
        logging.info(weights)
        logging.info(f"{str(datetime.now())} Particle mean")
        logging.info(particle_mean)
        distance = np.linalg.norm(particle_mean - self.true_param)
        logging.info(f"{str(datetime.now())} L2_Distance(true_p, est_p) = {distance}")
        logging.info(f"{str(datetime.now())} Time elapsed {end_time - start_time}")
        logging.info(f"{str(datetime.now())} End experiments.")


def main(cfg_name: str, mode: EvaluationMode):
    cfg = ExperimentSirConfig.get_config(cfg_name)
    smc_trace_len = cfg["smc_trace_len"]
    smc_pertubation_len = cfg["smc_pertubation_len"]
    smc_mh_trace_len = cfg["smc_mh_trace_len"]
    abc_threshold = cfg["abc_threshold"]
    prism_model_file = cfg["prism_model_file"]
    prism_props_file = cfg["prism_props_file"]
    interval = cfg["interval"]
    true_param = cfg["true_param"]
    observed_data = cfg["observed_data"]
    observed_labels = cfg["observed_labels"]
    experiment = ExperimentSirSmc(
        mode=mode,
        smc_trace_len=smc_trace_len,
        smc_pertubation_len=smc_pertubation_len,
        smc_mh_trace_len=smc_mh_trace_len,
        abc_threshold=abc_threshold,
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=interval,
        true_param=true_param,
        observed_data=observed_data,
        observed_labels=observed_labels,
    )
    experiment.run()


def manual():
    cfg = ExperimentSirConfig.get_config("sir_15_1_0_a_few")
    gCfg.set_abc_threshold_decreasing_factor(0.95)
    gCfg.set_per_bscc_sampling(10000)
    smc_trace_len = cfg["smc_trace_len"]
    smc_pertubation_len = 10
    smc_mh_trace_len = cfg["smc_mh_trace_len"]
    abc_threshold = cfg["abc_threshold"]
    prism_model_file = cfg["prism_model_file"]
    prism_props_file = cfg["prism_props_file"]
    interval = cfg["interval"]
    true_param = cfg["true_param"]
    observed_data = cfg["observed_data"]
    observed_labels = cfg["observed_labels"]
    experiment = ExperimentSirSmc(
        mode=EvaluationMode.Simulation,
        smc_trace_len=smc_trace_len,
        smc_pertubation_len=smc_pertubation_len,
        smc_mh_trace_len=smc_mh_trace_len,
        abc_threshold=abc_threshold,
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        interval=interval,
        true_param=true_param,
        observed_data=observed_data,
        observed_labels=observed_labels,
    )
    experiment.run()


if __name__ == "__main__":
    print("Supported config entries: ")
    for cfg in ExperimentSirConfig.get_all_config_names():
        print(cfg)

    if len(sys.argv) == 2:
        manual()

    if len(sys.argv) > 4:
        raise ValueError(f"Invalid number of arguments {len(sys.argv)}")

    mode = EvaluationMode.Simulation
    mode_str = sys.argv[1]
    if mode_str == "sim":
        mode = EvaluationMode.Simulation
        if len(sys.argv) == 4:
            sim_count = int(sys.argv[3])
            logging.info(
                f"{str(datetime.now())} Manually set simulation count: {sim_count}"
            )
            gCfg.set_per_bscc_sampling(sim_count)
    elif mode_str == "rf":
        mode = EvaluationMode.RationalFunction
    else:
        raise ValueError(f"Invalid evaluation mode {sys.argv[1]}")

    cfg_name = sys.argv[2]
    main(cfg_name=cfg_name, mode=mode)

    logging.shutdown()