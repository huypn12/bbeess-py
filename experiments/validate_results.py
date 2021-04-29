from scripts.model.simple_prism_rf_model import SimpleRfModel
from scripts.model.simple_prism_sim_model import SimpleSimModel

from typing import Any, Dict
import numpy as np

from datetime import datetime
import logging

logging.basicConfig(
    filename=f"temp-files/log/validation_{str(datetime.now())}.log",
    level=logging.DEBUG,
)


class ResultValidation:
    def __init__(self) -> None:
        self.result = self._result()

    def validate(self) -> None:
        for k, v in self.result.items():
            logging.info(f"{str(datetime.now())} Config: {k}")
            self._validate_config(v)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        prism_model_file = config["prism_model_file"]
        prism_props_file = config["prism_props_file"]
        logging.info(f"{str(datetime.now())} PRISM model: {prism_model_file}")
        logging.info(f"{str(datetime.now())} PRISM props: {prism_props_file}")
        true_param = config["true_param"]
        logging.info(f"{str(datetime.now())} True param: {true_param}")
        rf_model = SimpleRfModel(
            prism_model_file,
            prism_props_file,
        )
        logging.info(
            f"{str(datetime.now())} Target property: {rf_model.check_prop_bounded}"
        )
        dobs = config["observed_data"]
        logging.info(f"{str(datetime.now())} Observed data: { dobs}")
        sat_p = rf_model.check_unbounded(true_param)
        logging.info(
            f"{str(datetime.now())} True param satisfaction probability: {sat_p}"
        )
        estimated_param_rf = config["estimated"][0]
        sat_p = rf_model.check_unbounded(estimated_param_rf)
        logging.info(f"{str(datetime.now())} RF satisfaction probability: {sat_p}")
        sim_model = SimpleSimModel(
            prism_model_file, prism_props_file, config["observed_labels"]
        )
        estimated_param_sim = config["estimated"][1]
        sat_p = sim_model.check_unbounded(true_param)
        logging.info(
            f"{str(datetime.now())} True param Eval satisfaction probability: {sat_p}"
        )
        sat_p = sim_model.check_unbounded(estimated_param_rf)
        logging.info(f"{str(datetime.now())} RF Eval satisfaction probability: {sat_p}")
        sat_p = sim_model.check_unbounded_smc(estimated_param_rf)
        logging.info(f"{str(datetime.now())} RF Sim satisfaction probability: {sat_p}")
        sat_p = sim_model.check_unbounded(estimated_param_sim)
        logging.info(f"{str(datetime.now())} Sim satisfaction probability: {sat_p}")
        sat_p = sim_model.check_unbounded_smc(estimated_param_sim)
        logging.info(
            f"{str(datetime.now())} Sim, APMC satisfaction probability: {sat_p}"
        )

    def _result(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "zeroconf_4": {
                "estimated": np.array([[0.188956, 0.460554], [0.176469, 0.355322]]),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_4.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_4.pctl",
                "true_param": np.array([0.105547, 0.449658]),
                "observed_data": np.array([41, 9959]),
                "observed_labels": ["err", "ok"],
            },
            "zeroconf_10": {
                "estimated": np.array([[0.301807, 0.457090], [0.378774, 0.405870]]),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/zeroconf_10.pctl",
                "true_param": np.array([0.197779, 0.621824]),
                "observed_data": np.array([22, 9978]),
                "observed_labels": ["err", "ok"],
            },
            "bee_3": {
                "estimated": np.array(
                    [
                        [0.671388, 0.575026, 0.525502],
                        [0.81165139, 0.62107331, 0.5441299],
                    ]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_3.pctl",
                "true_param": np.array([0.66562362, 0.83040077, 0.83977757]),
                "observed_data": np.array([344, 54, 1390, 8212]),
                "observed_labels": ["bscc_0", "bscc_1", "bscc_2", "bscc_3"],
            },
            "bee_5": {
                "estimated": np.array(
                    [
                        [0.576565, 0.589724, 0.490334, 0.554397, 0.524433],
                        [0.36121979, 0.31600669, 0.5456908, 0.64396223, 0.59120587],
                    ]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_5.pctl",
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
                "estimated": np.array(
                    [
                        [
                            0.60488117,
                            0.47255664,
                            0.4844413,
                            0.5007056,
                            0.4933995,
                            0.49550837,
                            0.46659561,
                            0.51016748,
                            0.47415344,
                            0.48406133,
                        ],
                        [
                            0.3913126,
                            0.48568797,
                            0.42405577,
                            0.38148878,
                            0.44068071,
                            0.57886481,
                            0.594232,
                            0.56455705,
                            0.5478044,
                            0.52000621,
                        ],
                    ]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/bee_10.pctl",
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
            "sir_5_1_0": {
                "estimated": np.array([[0.025473, 0.067613], [0.02675003, 0.06898774]]),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_5_1_0.pctl",
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
            "sir_10_1_0": {
                "estimated": np.array([[0.01409, 0.066328], [0.022552, 0.066416]]),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl",
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
            "sir_15_1_0": {
                "estimated": np.array(
                    [[0.01002178, 0.06722998], [0.01244393, 0.06586205]]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0.pctl",
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
            "sir_15_1_0_a_few": {
                "estimated": np.array(
                    [[0.00945054, 0.06634182], [0.01669793, 0.08115346]]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
                "true_param": np.array([0.011499276183591657, 0.06211051606863456]),
                "observed_data": np.array([54, 146]),
                "observed_labels": ["bscc_k_0_gt8", "bscc_k_0_leq8"],
            },
            "sir_15_1_0_a_few_mod": {
                "estimated": np.array(
                    [[0.01148671, 0.06624936], [0.01148671, 0.06624936]]
                ),
                "prism_model_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pm",
                "prism_props_file": "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_15_1_0_a.pctl",
                "true_param": np.array([0.011499276183591657, 0.06211051606863456]),
                "observed_data": np.array([54, 146]),
                "observed_labels": ["bscc_k_0_gt8", "bscc_k_0_leq8"],
            },
        }
        return result


def main():
    validation = ResultValidation()
    validation.validate()


if __name__ == "__main__":
    main()
    logging.shutdown()