import subprocess

from typing import List, Dict, Any
from enum import Enum


class ResultEntry(Enum):
    simulating = "Simulating:"
    simulation_method = "Simulation method:"
    simulation_method_parameters = "Simulation method parameters:"
    simulation_parameters = "Simulation parameters:"


class PrismStatsMc(object):
    def __init__(self) -> None:
        super().__init__()
        self.prism_path = ""
        self.prism_cmd = "prism"
        self.prism_args = []
        self.result = ""

    def run(self):
        return self._simulate()

    def get_result(self):
        return

    def _tokenize_output(self, prism_output: List[str]) -> Dict[str, str]:
        parsed_lines: List[str] = []
        for line in prism_output:
            if line.startswith(ResultEntry.simulating.value):
                self._process_simulating_line()
            elif line.startswith(ResultEntry.simulation_method.value):
                self._process_simulation_result_line()
            elif line.startswith(ResultEntry.simulation_method_parameters):
                pass
            else:
                pass

    def _process_simulation_params(self, sim_result: str) -> Dict[str, float]:
        pass

    def _process_simulation_result(self, sim_result: str) -> Dict[str, float]:
        pass
