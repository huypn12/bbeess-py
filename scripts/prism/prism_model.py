import abc
from scripts.prism.prism_smc_executor import PrismSmcExecutor
from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars

import numpy as np


class AbstractPrismModelProps(abc.ABC):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = None
        self.prism_props = None

    def _load_prism_model_props(self):
        self._load_model_file()
        self._load_props_file()
        assert self.prism_program is not None
        assert self.prism_props is not None

    def _load_model_file(self):
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)

    def _load_props_file(self):
        lines: List[str] = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        props_str = ";".join(lines)
        self.prism_props = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )


class AbstractModelRational(AbstractPrismModelProps):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        super().__init__(prism_model_file, prism_props_file)


class AbstractModelSimulation(AbstractPrismModelProps):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        super().__init__(prism_model_file, prism_props_file)
        self.prism_smc_executor = PrismSmcExecutor(
            prism_exec="",
            prism_model_file=prism_model_file,
            prism_props_file=prism_props_file,
        )

    pass