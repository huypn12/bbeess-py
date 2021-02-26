import abc
from typing import List, Optional, Dict, Tuple, Any, Type

import numpy as np


class AbstractObservableModel(abc.ABC):
    @abc.abstractmethod
    def estimate_log_llh(self, particle: np.array) -> float:
        pass
