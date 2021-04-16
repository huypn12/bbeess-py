from experiments.experiment_config import ExperimentConfig

import numpy as np


class ExperimentZeroconfConfig(ExperimentConfig):
    _config = {
        "zeroconf_4": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_4.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_4.pctl",
            "interval": (0, 1),
            "true_param": np.array([0.10554747457679226, 0.4496587423249129]),
            "observed_data": np.array([41, 9959]),
            "observed_labels": ["err", "ok"],
        },
        "zeroconf_10": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pctl",
            "interval": (0, 1),
            "true_param": np.array([0.6508177164148675, 0.7072758508868217]),
            "observed_data": np.array([524, 9476]),
            "observed_labels": ["err", "ok"],
        },
    }

    @staticmethod
    def _get_config():
        return ExperimentZeroconfConfig._config