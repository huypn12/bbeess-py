import numpy as np
import scipy as sp


class AbcSmc(object):
    def __init__(self) -> None:
        super().__init__()

    def _distance(self, v1: np.array, v2: np.array) -> float:
        """Distance between two vector"""
        return np.linalg.norm(v1 - v2)

    def _kernel(self, s1, s2, threshold):
        return 1 if (AbcSmc._distance(s1, s2) > threshold) else 0

    def _check_model_smc(self):
        pass

    def _estimate_obs_llh(self):
        pass

    def _estimate_check_llh(self):
        pass