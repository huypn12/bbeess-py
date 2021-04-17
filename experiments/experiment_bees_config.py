from experiments.experiment_config import ExperimentConfig

import numpy as np


class ExperimentBeesConfig(ExperimentConfig):
    _config = {
        "bee_3": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pctl",
            "interval": (0, 1),
            "true_param": np.array([0.66562362, 0.83040077, 0.83977757]),
            "observed_data": np.array([344, 54, 1390, 8212]),
            "observed_labels": ["bscc_0", "bscc_1", "bscc_2", "bscc_3"],
        },
        "bee_5": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pctl",
            "interval": (0, 1),
            "true_param": np.array(
                [0.2783698, 0.30599383, 0.4897924, 0.73725233, 0.76658066]
            ),
            "observed_data": np.array([1940, 11, 216, 2682, 4200, 951]),
            "observed_labels": [
                "bscc_0",
                "bscc_1",
                "bscc_2",
                "bscc_3",
                "bscc_4",
                "bscc_5",
            ],
        },
        "bee_10": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pctl",
            "interval": (0, 1),
            "true_param": np.array(
                [
                    0.2221692,
                    0.24699272,
                    0.28193407,
                    0.44638416,
                    0.49161226,
                    0.53461125,
                    0.56940931,
                    0.68465134,
                    0.7171388,
                    0.80098673,
                ]
            ),
            "observed_data": np.array(
                [769, 0, 1, 10, 187, 972, 2494, 2982, 2133, 419, 33]
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
        return ExperimentBeesConfig._config