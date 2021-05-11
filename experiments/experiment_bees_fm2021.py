import scripts.config as gCfg

from experiments.experiment_zeroconf_config import ExperimentZeroconfConfig
from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.mc.smc_rf_uniform_kernel import SmcRfUniformKernel
from scripts.model.simple_prism_sim_model import SimpleSimModel
from scripts.mc.abc_smc_smc_uniform_kernel import AbcSmcSmcUniformKernel

from scripts.utils.hpd import compute_hpd_univariate, compute_hpd_multivariate


from experiments.experiment_config import ExperimentConfig

import os
from datetime import datetime
import logging
from enum import IntEnum
from typing import Tuple, List

logging.basicConfig(
    filename=f"temp-files/log/{str(datetime.time(datetime.now()))}_experiment_bees_fm2021_smc.log",
    level=logging.DEBUG,
)
import numpy as np


class ExperimentFm2021Config(ExperimentConfig):
    _config = {
        "bees_10": {
            "smc_trace_len": 500,
            "smc_pertubation_len": 5,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pctl",
            "interval": (0, 1),
            "true_param": np.array(
                [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
            ),
            "observed_data": np.array(
                [513, 75, 296, 707, 1214, 1698, 1837, 168, 1154, 643, 183]
            ),
            "observed_labels": [
                "bscc_0",
                "bscc_1",
                "bscc_2",
                "bscc_3",
                "bscc_4",
                "bscc_5",
                "bscc_6",
                "bscc_7",
                "bscc_8",
                "bscc_9",
                "bscc_10",
            ],
        },
    }

    @staticmethod
    def _get_config():
        return ExperimentFm2021Config._config


class EvaluationMode(IntEnum):
    RationalFunction = 1
    Simulation = 2


class ExperimentFm2021(object):
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
        self.particle_dim = len(self.true_param)

    def _init(self):
        if self.mode == EvaluationMode.RationalFunction:
            self.model = SimpleRfModel(
                self.prism_model_file,
                self.prism_props_file,
            )
            self.mc = SmcRfUniformKernel(
                model=self.model,
                interval=self.interval,
                particle_dim=self.particle_dim,
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
                particle_dim=self.particle_dim,
                particle_trace_len=self.smc_trace_len,
                kernel_count=self.smc_pertubation_len,
                observed_data=self.observed_data,
                abc_threshold=self.abc_threshold,
            )
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
        # self.model.check_bounded(self.true_param)
        logging.info(f"{str(datetime.now())} Mode {self.mode}")
        logging.info(f"{str(datetime.now())} PRISM model: {self.prism_model_file}")
        logging.info(f"{str(datetime.now())} PRISM props: {self.prism_props_file}")
        logging.info(f"{str(datetime.now())} True parameter: {self.true_param}")
        logging.info(f"{str(datetime.now())} Synthetic data: {self.observed_data}")

    def run(self):
        self._init()
        start_time = datetime.now()
        self.mc.run()
        particle_mean, trace, weights = self.mc.get_result()
        end_time = datetime.now()
        logging.info(f"{str(datetime.now())} Particle trace")
        logging.info(trace)
        model_file = os.path.basename(self.prism_model_file)
        with open(
            f"temp-files/log/{str(datetime.time(datetime.now()))}_{model_file}_{self.mode.name}_trace.npy",
            "wb",
        ) as fptr:
            np.save(fptr, trace)
        logging.info(f"{str(datetime.now())} Particle weights")
        logging.info(weights)
        with open(
            f"temp-files/log/{str(datetime.time(datetime.now()))}_{model_file}_{self.mode.name}_weight.npy",
            "wb",
        ) as fptr:
            np.save(fptr, weights)
        logging.info(f"{str(datetime.now())} HPDs")
        hpd = compute_hpd_multivariate(trace, self.particle_dim)
        logging.info(hpd)
        logging.info(f"{str(datetime.now())} Particle mean")
        logging.info(particle_mean)
        distance = np.linalg.norm(particle_mean - self.true_param)
        logging.info(f"{str(datetime.now())} L2_Distance(true_p, est_p) = {distance}")
        logging.info(f"{str(datetime.now())} Time elapsed {end_time - start_time}")
        logging.info(f"{str(datetime.now())} End experiments.")


def main(cfg_name: str, mode: EvaluationMode):
    gCfg.set_bee_model(False)
    gCfg.set_has_synthesis(False)
    gCfg.set_abc_threshold_decreasing_factor(0.99)
    cfg = ExperimentFm2021Config.get_config(cfg_name)
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
    experiment = ExperimentFm2021(
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


def main_all():
    for eval_mode in [EvaluationMode.Simulation]:
        for cfg in ExperimentFm2021Config.get_all_config_names():
            main(cfg, eval_mode)


if __name__ == "__main__":
    main_all()

    logging.shutdown()