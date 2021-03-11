from abc import abstractmethod
import subprocess

from typing import List, Optional
from enum import Enum


class PrismSmcCmdResultEntry(Enum):
    simulating = "Simulating:"
    method = "Simulation method:"
    method_parameters = "Simulation method parameters:"
    parameters = "Simulation parameters:"
    result_details = "Simulation result details:"
    result = "Result:"


class PrismSmcExecutor(object):
    def __init__(
        self,
        model_file: str,
        property_file: str,
    ) -> None:
        super().__init__()
        self.prism_exec = "prism"
        self.model_file: str = model_file
        self.property_file: str = property_file
        self.exec_output: str = ""
        self.result_str: str = ""
        self.resutl_num: float = 0.0
        self.resutl_bool: bool = False

    def _execute_prism(self, model_consts: Optional[str] = None):
        prism_args = self._get_prism_args(model_consts)
        process = subprocess.Popen(
            [self.prism_exec] + prism_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, _ = process.communicate()
        self.exec_output = stdout

    def _get_base_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        cmd_args = [
            self.model_file,
            self.property_file,
            "-prop",
            "1",
            "-sim",
        ]
        if model_consts:
            cmd_args.extend(["-const", model_consts])
        return cmd_args

    @abstractmethod
    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        pass

    def exec(self, model_consts_str: Optional[str] = None):
        self._execute_prism(model_consts_str)
        self._process_output()
        return self.result

    def _process_output(self):
        assert self.exec_output
        parsed_lines: List[str] = self.exec_output.splitlines(keepends=False)
        for line in parsed_lines:
            if line.startswith(PrismSmcCmdResultEntry.simulating.value):
                self._process_simulating(line)
            elif line.startswith(PrismSmcCmdResultEntry.result.value):
                self._process_final_result(line)
            else:
                continue

    def _process_final_result(self, line: str):
        result_str: str = line.replace(PrismSmcCmdResultEntry.result.value, "")
        self.result = float(result_str)


class PrismSmcSprtExecutor(PrismSmcExecutor):
    def __init__(
        self,
        model_file: str,
        property_file: str,
        simwidth: int = 1000,
        simconf: float = 0.95,
        simapprox: int = 1000,
    ) -> None:
        super().__init__(model_file, property_file)
        self.simwidth: int = simwidth
        self.simconf: float = simconf
        self.simapprox: int = simapprox

    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        base_args = self._get_base_prism_args(model_consts)
        sim_args = [
            "-method",
            "sprt",
            "-simwidth",
            str(self.simwidth),
            "-simconf",
            str(self.simconf),
            "-simapprox",
            str(self.simapprox),
        ]
        return base_args + sim_args


class PrismSmcApmcExecutor(PrismSmcExecutor):
    def __init__(
        self,
        model_file: str,
        property_file: str,
        simwidth: int = 1000,
        simconf: float = 0.95,
        simapprox: int = 1000,
    ) -> None:
        super().__init__(model_file, property_file)
        self.simwidth: int = simwidth
        self.simconf: float = simconf
        self.simapprox: int = simapprox

    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        base_args = self._get_base_prism_args(model_consts)
        sim_args = [
            "-method",
            "apmc",
            "-simwidth",
            str(self.simwidth),
            "-simconf",
            str(self.simconf),
            "-simapprox",
            str(self.simapprox),
        ]
        return base_args + sim_args