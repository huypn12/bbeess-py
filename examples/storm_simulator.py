import stormpy
import stormpy.core
import stormpy.simulator
import stormpy.pars

import stormpy.examples
import stormpy.examples.files

import multiprocessing as mp
import numpy as np
import random

"""
Simulator for deterministic models
"""


def create_model():
    path = "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/multi_sync_3_bees.pm"
    prism_program = stormpy.parse_prism_program(path)
    options = stormpy.BuilderOptions([])
    options.set_build_state_valuations()
    model = stormpy.build_parametric_model(prism_program)
    for state in model.states:
        print(str(state.id) + " : " + str(state.labels if state.labels else ""))
    parameters = model.collect_probability_parameters()
    param = [0.3, 0.4, 0.5]
    point = dict()
    for i, p in enumerate(parameters):
        point[p] = stormpy.RationalRF(param[i])
    instantiator = stormpy.pars.PDtmcInstantiator(model)
    instantiated_model = instantiator.instantiate(point)

    return instantiated_model


def run_simulator():
    simulator = stormpy.simulator.create_simulator(create_model(), seed=42)
    final_outcomes = dict()
    for n in range(10000):
        observation = None
        while not simulator.is_done():
            observation, reward = simulator.step()
        if observation not in final_outcomes:
            final_outcomes[observation] = 1
        else:
            final_outcomes[observation] += 1
        simulator.restart()

    print(", ".join([f"{str(k)}: {v}" for k, v in final_outcomes.items()]))


if __name__ == "__main__":
    run_simulator()