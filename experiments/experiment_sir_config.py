import numpy as np

from experiments.experiment_config import ExperimentConfig


class ExperimentSirConfig(ExperimentConfig):
    _config = {
        "sir_3_1_0": {
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
        "sir_5_1_0": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.03405521326944327, 0.08773454035144489]),
            "observed_data": np.array([1098, 1377, 1296, 1312, 1466, 3451]),
            "observed_labels": [
                "bscc_0_0_6",
                "bscc_1_0_5",
                "bscc_2_0_4",
                "bscc_3_0_3",
                "bscc_4_0_2",
                "bscc_5_0_1",
            ],
        },
        "sir_5_1_0_a": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.03405521326944327, 0.08773454035144489]),
            "observed_data": np.array([3964, 6036]),
            "observed_labels": ["bscc_k_0_geq4", "bscc_k_0_leq3"],
        },
        "sir_5_1_0_a_few": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.03405521326944327, 0.08773454035144489]),
            "observed_data": np.array([76, 124]),
            "observed_labels": ["bscc_k_0_geq4", "bscc_k_0_leq3"],
        },
        "sir_10_1_0": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025490115891226895, 0.06929809986640066]),
            "observed_data": np.array(
                [1002, 1258, 1123, 902, 770, 651, 497, 420, 496, 685, 2196]
            ),
            "observed_labels": [
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
        },
        "sir_10_1_0_a": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025490115891226895, 0.06929809986640066]),
            "observed_data": np.array([9607, 393]),
            "observed_labels": ["bscc_k_0_gt6", "bscc_k_0_leq6"],
        },
        "sir_10_1_0_a_few": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.025490115891226895, 0.06929809986640066]),
            "observed_data": np.array([87, 113]),
            "observed_labels": ["bscc_k_0_gt6", "bscc_k_0_leq6"],
        },
        "sir_15_1_0": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.011499276183591657, 0.06211051606863456]),
            "observed_data": np.array(
                [
                    50,
                    181,
                    302,
                    455,
                    539,
                    567,
                    582,
                    566,
                    541,
                    553,
                    574,
                    528,
                    512,
                    586,
                    875,
                    2589,
                ]
            ),
            "observed_labels": [
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
        },
        "sir_15_1_0_a": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.011499276183591657, 0.06211051606863456]),
            "observed_data": np.array([3227, 6773]),
            "observed_labels": ["bscc_k_0_gt8", "bscc_k_0_leq8"],
        },
        "sir_15_1_0_a_few": {
            "smc_trace_len": 200,
            "smc_pertubation_len": 20,
            "smc_mh_trace_len": 50,
            "abc_threshold": 0.25,
            "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
            "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
            "interval": (0, 0.1),
            "true_param": np.array([0.011499276183591657, 0.06211051606863456]),
            "observed_data": np.array([54, 146]),
            "observed_labels": ["bscc_k_0_gt8", "bscc_k_0_leq8"],
        },
    }

    @staticmethod
    def _get_config():
        return ExperimentSirConfig._config