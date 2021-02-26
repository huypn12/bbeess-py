import numpy as np
from typing import List, Dict, Any


class ZeroconfGenerator(object):
    def __init__(self, state_count: int) -> None:
        self.adj_list: List[Any] = []
