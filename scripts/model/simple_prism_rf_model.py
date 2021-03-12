from scripts.model.abstract_model import AbstractRationalModel

from typing import List, Optional, Dict, Tuple, Any, Type
import stormpy
import stormpy.core
import stormpy.pars

import numpy as np


class SimpleRfModel(AbstractRationalModel):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        s_obs: List[float],
    ) -> None:
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.model = None
        self.prism_program = None
        self.prism_props = None
        self.s_obs = None

    def _load_prism_model_props(self):
        self._load_model_file()
        self._load_props_file()

    def _load_model_file(self):
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)
        assert self.prism_program is not None

    def _load_props_file(self):
        lines: List[str] = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        props_str = ";".join(lines)
        self.prism_props = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )
        assert self.prism_props is not None

    def _load_rf(self):
        self.model = stormpy.build_parametric_model(
            self.my_prism_program.prism_program,
            self.my_prism_program.prism_props,
        )
        self.model_parameters = self.model.collect_probability_parameters()
        # Property for checking
        self.check_prop = self.my_prism_program.prism_props[0]
        self.check_rf = stormpy.model_checking(self.model, self.check_prop).at(
            self.model.initial_states[0]
        )
        # Properties for observing
        self.obs_props = self.my_prism_program.prism_props[1:]
        assert len(self.s_obs) == len(self.obs_props)
        self.obs_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.obs_props
        ]
