from scripts.model.abstract_model import AbstractSimulationModel
from scripts.prism.prism_smc_executor import PrismSmcSprtExecutor, PrismSmcApmcExecutor

from typing import List, Optional, Dict, Tuple, Any, Type

import stormpy
import stormpy.core
import stormpy.pars
import stormpy.simulator

import numpy as np


class SimpleSimModel(AbstractSimulationModel):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        obs_labels: List[str],
    ) -> None:
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = None
        self.prism_props = None
        self.prism_sprt_executor = PrismSmcSprtExecutor(
            self.prism_model_file, self.prism_props_file
        )
        self.prism_apmc_executor = PrismSmcApmcExecutor(
            self.prism_model_file, self.prism_props_file
        )
        self.model = None
        self.check_prop = None
        self.obs_props = []
        self.obs_labels = obs_labels
        self._load()

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
            self.prism_program,
            self.prism_props,
        )
        # Property for checking
        self.check_prop_bounded = self.prism_props[0]
        self.check_prop_unbounded = self.prism_props[1]
        # Properties for observing
        self.obs_props = self.prism_props[2:]
        # State label mapping
        self.state_mapping: Dict[int, List[str]] = {}
        for state in self.model.states:
            self.state_mapping[state.id] = [label for label in state.labels]

    def _load(self):
        self._load_model_file()
        self._load_props_file()
        self._load_rf()

    def _instantiate(self, particle: np.array):
        instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        model_parameters = self.model.collect_probability_parameters()
        point = dict()
        for i, p in enumerate(model_parameters):
            point[p] = stormpy.RationalRF(particle[i])
        instantiated_model = instantiator.instantiate(point)
        return instantiated_model

    def _is_obs_state(self, state_idx: int) -> Tuple[bool, Optional[str]]:
        state_labels = self.state_mapping[state_idx]
        for label in state_labels:
            if label in self.obs_labels:
                return (True, label)
        return (False, None)

    def _model_params_to_prism_cmd_args(self, particle: np.array):
        params = []
        model_parameters = self.model.collect_probability_parameters()
        for i, p in enumerate(model_parameters):
            param_str = str(p) + "=" + str(particle[i])
            params.append(param_str)
        return ",".join(params)

    def _set_prism_args(
        self,
        simwidth: Optional[float] = None,
        simsamples: Optional[float] = None,
        simconf: Optional[float] = None,
        simapprox: Optional[float] = None,
    ):
        self.prism_sprt_executor.set_prism_args(
            simwidth=simwidth,
            simsamples=simsamples,
            simconf=simconf,
            simapprox=simapprox,
        )
        self.prism_apmc_executor.set_prism_args(
            simwidth=simwidth,
            simsamples=simsamples,
            simconf=simconf,
            simapprox=simapprox,
        )

    def check_bounded(self, particle: np.array):
        self.prism_sprt_executor.set_prism_args(
            simwidth=0.0005,
            simconf=0.025,
        )
        result = self.prism_sprt_executor.exec(
            self._model_params_to_prism_cmd_args(particle)
        )
        return result

    def check_unbounded(self, particle: np.array):
        instantiated_model = self._instantiate(particle)
        initial_state = self.model.initial_states[0]
        result = stormpy.model_checking(
            instantiated_model, self.check_prop_unbounded
        ).at(initial_state)
        return result

    def simulate(self, particle: np.array, sample_count: int):
        model_params = self.model.collect_probability_parameters()
        assert len(particle) == len(model_params)
        point = dict()
        for i, p in enumerate(model_params):
            point[p] = stormpy.RationalRF(particle[i])
        instantiated_model = self._instantiate(particle)
        simulator = stormpy.simulator.create_simulator(
            instantiated_model, seed=np.random.randint(1000)
        )

        final_outcomes = dict()
        for _ in range(sample_count):
            observation = None
            while not simulator.is_done():
                observation, _ = simulator.step()  # reward in place hodler
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()

        summary = np.zeros(len(self.obs_props))
        for k, v in final_outcomes.items():
            is_obs_state, label = self._is_obs_state(k)
            if is_obs_state:
                summary[self.obs_labels.index(label)] = v
        return summary

    def estimate_distance(self, s_sim: np.array, s_obs: np.array) -> float:
        distance = np.linalg.norm(s_sim - s_obs)
        return distance
