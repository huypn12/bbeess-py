import abc

import numpy as np
from typing import List, Dict


class AbstractObservableModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass


class AbstractRationalModel(abc.ABC):
    @abc.abstractmethod
    def check_prop(self, particle: np.array) -> float:
        pass

    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass


class AbstractSimulationModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass

    @abc.abstractmethod
    def check_prop(self, particle: np.array) -> float:
        pass

    @abc.abstractmethod
    def estimate_distance(
        self,
        theta: np.array,
        s_sim: np.array,
    ) -> float:
        pass