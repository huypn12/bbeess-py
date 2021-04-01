import numpy as np
from typing import Tuple


class BaseExperiment:
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        interval: Tuple[float, float],
        true_param: np.array,
        observed_data: np.array,
    ):
        self.prism_model_file = prism_model_file
        self.prism_props_file = prism_props_file