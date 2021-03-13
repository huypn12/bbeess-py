import abc

import numpy as np
from typing import List, Dict


class AbstractObservableModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass


class AbstractRationalModel(abc.ABC):
    @abc.abstractmethod
    def check_bounded(self, particle: np.array) -> float:
        return 0.0

    @abc.abstractmethod
    def check_unbounded(self, particle: np.array) -> float:
        return 0.0

    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array, y_obs: np.array) -> float:
        pass


class AbstractSimulationModel(abc.ABC):
    @abc.abstractmethod
    def check_bounded(self, particle: np.array) -> float:
        return 0.0

    @abc.abstractmethod
    def check_unbounded(self, particle: np.array) -> float:
        return 0.0

    @abc.abstractmethod
    def estimate_distance(
        self,
        s_sim: np.array,
        s_obs: np.array,
    ) -> float:
        pass