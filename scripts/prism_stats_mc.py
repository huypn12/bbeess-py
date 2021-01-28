import subprocess

from typing import List, Optional
from enum import Enum


class ResultEntry(Enum):
    simulating = "Simulating:"
    method = "Simulation method:"
    method_parameters = "Simulation method parameters:"
    parameters = "Simulation parameters:"
    result_details = "Simulation result details:"
    result = "Result:"


class PrismStatsMc(object):
    def __init__(
        self,
        prism_exec: str,
        model_file: str,
        property_file: str,
        sim_path_len: int = 1000,
        sim_confidence: float = 0.95,
        sim_samples: int = 1000,
    ) -> None:
        super().__init__()
        self.prism_exec: str = prism_exec
        self.model_file: str = model_file
        self.property_file: str = property_file
        self.exec_output: str = None
        self.is_numeric_result: bool = False
        self.sim_path_len: int = sim_path_len
        self.sim_confidence: float = sim_confidence
        self.sim_samples: int = sim_samples
        self.result: float = 0.0

    def _execute_prism(self, model_consts: Optional[str] = None):
        prism_args = self._get_prism_args(model_consts)
        process = subprocess.Popen(
            [self.prism_exec] + prism_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate()
        assert not stderr
        self.exec_output = stdout

    def _get_prism_args(self, model_consts: Optional[str] = None) -> List[str]:
        cmd_args = [
            self.model_file,
            self.property_file,
            "-prop",
            "1",
            "-sim",
            "-simsamples",
            str(self.sim_samples),
            "-simconf",
            str(self.sim_confidence),
        ]
        if model_consts:
            cmd_args.extend(["-const", model_consts])
        return cmd_args

    def run(self, model_consts_str: Optional[str] = None):
        self._execute_prism(model_consts_str)
        self._process_output()
        return self.result

    def _process_output(self):
        assert self.exec_output
        parsed_lines: List[str] = self.exec_output.splitlines(keepends=False)
        for line in parsed_lines:
            if line.startswith(ResultEntry.simulating.value):
                self._process_simulating(line)
            elif line.startswith(ResultEntry.result.value):
                self._process_final_result(line)
            else:
                continue

    def _process_simulating(self, line: str):
        self.is_numeric_result = False
        if "P=?" in line or "R=?" in line:
            self.is_numeric_result = True

    def _process_final_result(self, line: str):
        result_str: str = line.replace(ResultEntry.result.value, "")
        self.result = float(result_str)


def main():
    stats_mc = PrismStatsMc(
        prism_exec="/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/third_party/prism-4.6-linux64/bin/prism",
        model_file="/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/die.pm",
        property_file="/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/die.pctl",
        sim_path_len=1000,
        sim_confidence=0.95,
        sim_samples=1000,
    )
    print(stats_mc.run("p=0.5"))


if __name__ == "__main__":
    main()
