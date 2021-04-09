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
    ) -> None:
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = None
        self.prism_props = None
        self.model = None
        # Property for
        self.check_prop = None
        self.check_rf = None
        # Properties fo
        self.obs_props = []
        self.obs_rf = []
        # initiate
        self._load()

    def _load_prism_program(self):
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)
        assert self.prism_program is not None

    def _extract_bounded_op_str(self, prop_str: str):
        prop_str = prop_str.split(" ")[0]
        opstr = "".join([c for c in prop_str if c in "<>="])
        if opstr not in [">", "<", ">=", "<="]:
            raise ValueError(f"Unrecognized bound operation{opstr} in PCTL {prop_str}")
        value_str = prop_str[prop_str.find(opstr) + len(opstr) :]
        value = float(value_str)
        return opstr, value

    def _replace_bounded_op_str(self, prop_str: str):
        prop_str_tokens = prop_str.split(" ")
        prop_str_tokens[0] = "P=?"
        return " ".join(prop_str_tokens)

    def _load_prism_props(self):
        lines: List[str] = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        self.prism_props_str = lines
        opstr, bound_value = self._extract_bounded_op_str(self.prism_props_str[0])
        self.check_prop_bounded_opstr = opstr
        self.check_prop_bounded_value = bound_value
        self.check_prop_unbounded_str = self._replace_bounded_op_str(
            self.prism_props_str[0]
        )
        self.prism_props_str.insert(1, self.check_prop_unbounded_str)
        props_str = ";".join(self.prism_props_str)
        self.prism_props = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )
        assert self.prism_props is not None

    def _load_rf(self):
        self.model = stormpy.build_parametric_model(
            self.prism_program,
            self.prism_props,
        )
        # Property for checking
        self.check_prop_bounded = self.prism_props[0]
        self.check_prop_unbounded = self.prism_props[1]
        self.check_rf_unbounded = stormpy.model_checking(
            self.model, self.check_prop_unbounded
        ).at(self.model.initial_states[0])
        # Properties for observing
        self.obs_props = self.prism_props[2:]
        self.obs_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.obs_props
        ]

    def _load(self):
        self._load_prism_program()
        self._load_prism_props()
        self._load_rf()

    def _instantiate(self, particle: np.array):
        instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        model_parameters = self.model.collect_probability_parameters()
        point = dict()
        for i, p in enumerate(model_parameters):
            point[p] = stormpy.RationalRF(particle[i])
        instantiated_model = instantiator.instantiate(point)
        return instantiated_model

    def check_bounded(self, particle: np.array):
        model_parameters = self.model.collect_probability_parameters()
        point = dict()
        for i, p in enumerate(model_parameters):
            point[p] = stormpy.RationalRF(particle[i])
        result = float(self.check_rf_unbounded.evaluate(point))
        op = self.check_prop_bounded_opstr
        if op == ">=":
            return result >= self.check_prop_bounded_value
        elif op == "<=":
            return result <= self.check_prop_bounded_value
        elif op == ">":
            return result > self.check_prop_bounded_value
        elif op == "<":
            return result < self.check_prop_bounded_value
        else:
            raise ValueError(
                f"Unable to compare boundary op={op} result={result} bound_value={self.check_prop_bounded_value}"
            )

    def check_unbounded(self, particle: np.array):
        model_parameters = self.model.collect_probability_parameters()
        point = dict()
        for i, p in enumerate(model_parameters):
            point[p] = stormpy.RationalRF(particle[i])
        result = float(self.check_rf_unbounded.evaluate(point))
        return result

    def simulate(self, particle: np.array, sample_count: int):
        model_params = self.model.collect_probability_parameters()
        assert len(particle) == len(model_params)
        point = dict()
        for i, p in enumerate(model_params):
            point[p] = stormpy.RationalRF(particle[i])
        P = [float(rf.evaluate(point)) for rf in self.obs_rf]
        sample = np.random.multinomial(sample_count, P)
        return sample

    def estimate_log_llh(self, particle: np.array, y_obs: np.array) -> float:
        model_params = self.model.collect_probability_parameters()
        assert len(particle) == len(model_params)
        point = dict()
        for i, p in enumerate(model_params):
            point[p] = stormpy.RationalRF(particle[i])
        P = [float(rf.evaluate(point)) for rf in self.obs_rf]
        log_llh = 0
        for i in range(0, len(particle)):
            log_llh += y_obs[i] * np.log(P[i])
        return log_llh
