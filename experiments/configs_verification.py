from experiments.experiment_config import ExperimentConfig
from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel
from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel

from typing import Tuple, List
from enum import IntEnum
import numpy as np
from datetime import datetime


class EvaluationMode(IntEnum):
    RationalFunction = 1
    Simulation = 2


class ConfigVerification(object):
    def __init__(
        self,
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

    def _verify(self):
        sat = self.model.check_bounded(self.true_param)
        print(f"{str(datetime.now())} PRISM model: {self.prism_model_file}")
        print(f"{str(datetime.now())} PRISM props: {self.prism_props_file}")
        print(f"{str(datetime.now())} RF props: {self.model.check_rf_unbounded}")
        for p in self.model.prism_props:
            print(p)
        print(f"{str(datetime.now())} True parameter (SAT={sat}): {self.true_param}")
        print(f"{str(datetime.now())} Synthetic data: {self.observed_data}")

    def run(self):
        self._init()
        self._verify()


def main():
    all_cfg_names = ExperimentConfig.get_all_config_names()
    for cfg_name in all_cfg_names:
        cfg = ExperimentConfig.get_config(cfg_name)
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
    main()
