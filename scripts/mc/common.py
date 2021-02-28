import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import numpy as np


class AbstractObservableModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass


class AbstractRationalModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass


class AbstractSimulationModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, theta: np.array) -> float:
        pass

    @abc.abstractmethod
    def estimate_distance(
        self,
        theta: np.array,
        s_sim: np.array,
    ) -> float:
        pass