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


class PrismSmcCmdParams(Enum):
    method = "method"


class PrismSmcExecutor(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__()
        self.prism_exec = "prism"
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prop_str: str = ""
        self.exec_output: str = ""
        self.result_str: str = ""
        self.result = None
        self.simwidth: Optional[float] = None
        self.simsamples: Optional[float] = None
        self.simconf: Optional[float] = None
        self.simapprox: Optional[float] = None

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
            self.prism_model_file,
            self.prism_props_file,
            "-prop",
            "1",
            "-sim",
        ]
        if model_consts:
            cmd_args.extend(["-const", model_consts])
        return cmd_args

    @abstractmethod
    def _get_prism_args(
        self,
        model_consts: Optional[str] = None,
    ) -> List[str]:
        pass

    def _process_output(self):
        assert self.exec_output
        parsed_lines: List[str] = self.exec_output.splitlines(keepends=False)
        for line in parsed_lines:
            if line.startswith(PrismSmcCmdResultEntry.simulating.value):
                self._process_simulating(line)
            elif line.startswith(PrismSmcCmdResultEntry.result_details.value):
                self._process_result_details(line)
            elif line.startswith(PrismSmcCmdResultEntry.result.value):
                self.result_str = line
                self._process_final_result(line)
            else:
                continue

    def _process_simulating(self, line: str):
        prop_str: str = line.replace(
            PrismSmcCmdResultEntry.simulating.value, ""
        ).strip()
        self.prop_str = prop_str

    @abstractmethod
    def _process_result_details(self, line: str):
        pass

    def _process_final_result(self, line: str):
        result_str: str = line.replace(PrismSmcCmdResultEntry.result.value, "").strip()
        self._set_result(result_str)

    @abstractmethod
    def _set_result(self, result_str: str):
        pass

    def set_prism_args(
        self,
        simwidth: Optional[float] = None,
        simsamples: Optional[float] = None,
        simconf: Optional[float] = None,
        simapprox: Optional[float] = None,
    ):
        self.simwidth = simwidth
        self.simsamples = simsamples
        self.simconf = simconf
        self.simapprox = simapprox

    def exec(self, model_consts_str: Optional[str] = None):
        self._execute_prism(model_consts_str)
        self._process_output()
        return self.result


class PrismSmcSprtExecutor(PrismSmcExecutor):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__(prism_model_file, prism_props_file)
        self.simsamples: int = -1
        self.result: bool = False

    def _set_result(self, result_str):
        if "false" in result_str.lower():
            self.result = False
        elif "true" in result_str.lower():
            self.result = True
        else:
            raise ValueError("Unsupported value {}".format(result_str.lower()))

    def _process_result_details(self, line: str):
        details_str: str = line.replace(
            PrismSmcCmdResultEntry.result_details.value, ""
        ).strip()
        self.simsamples = int(details_str.split(" ")[0])

    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        base_args = self._get_base_prism_args(model_consts)
        sim_args = [
            "-simmethod",
            "sprt",
        ]
        if self.simwidth:
            sim_args.extend(["-simwidth", str(self.simwidth)])
        if self.simconf:
            sim_args.extend(["-simconf", str(self.simconf)])
        return base_args + sim_args


class PrismSmcApmcExecutor(PrismSmcExecutor):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__(prism_model_file, prism_props_file)
        self.result: float = -1

    def _set_result(self, result_str):
        if "false" in result_str.lower():
            self.result = False
        elif "true" in result_str.lower():
            self.result = True
        else:
            raise ValueError("Unsupported value {}".format(result_str.lower()))

    def _process_result_details(self, line: str):
        details_str: str = line.replace(
            PrismSmcCmdResultEntry.result_details.value, ""
        ).strip()
        self.simsamples = int(details_str.split(" ")[0])

    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        base_args = self._get_base_prism_args(model_consts)
        sim_args = [
            "-simmethod",
            "apmc",
        ]
        if self.simsamples:
            sim_args.extend(["-simsamples", str(self.simsamples)])
        if self.simconf:
            sim_args.extend(["-simconf", str(self.simconf)])
        if self.simapprox:
            sim_args.extend(["-simapprox", str(self.simapprox)])
        return base_args + sim_args