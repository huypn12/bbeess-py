from experiments.experiment_config import ExperimentConfig

import numpy as np


class BeesExperimentConfig(ExperimentConfig):
    _config = {
        "bees_3": {
            "smc_trace_len": 10,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 5,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025877526857399316, 0.04726738301090197]),
            "observed_data": np.array([2337, 1967, 1952, 3744]),
            "observed_labels": ["bscc_0_0_4", "bscc_1_0_3", "bscc_2_0_2", "bscc_3_0_1"],
        },
        "bees_5": {
            "smc_trace_len": 10,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 5,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025877526857399316, 0.04726738301090197]),
            "observed_data": np.array([2337, 1967, 1952, 3744]),
            "observed_labels": ["bscc_0_0_4", "bscc_1_0_3", "bscc_2_0_2", "bscc_3_0_1"],
        },
        "bees_10": {
            "smc_trace_len": 10,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 5,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_3_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025877526857399316, 0.04726738301090197]),
            "observed_data": np.array([2337, 1967, 1952, 3744]),
            "observed_labels": ["bscc_0_0_4", "bscc_1_0_3", "bscc_2_0_2", "bscc_3_0_1"],
        },
    }
