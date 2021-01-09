import stormpy
import stormpy.core
import stormpy.simulator

import stormpy.examples
import stormpy.examples.files

import multiprocessing as mp
import numpy as np
import random

"""
Simulator for deterministic models
"""

g_path = stormpy.examples.files.prism_dtmc_die
g_prism_program = stormpy.parse_prism_program(g_path)
g_model = stormpy.build_model(g_prism_program)


def example_simulator_01():
    simulator = stormpy.simulator.create_simulator(g_model, seed=42)
    final_outcomes = dict()
    for n in range(1000):
        observation = None
        while not simulator.is_done():
            observation, reward = simulator.step()
        if observation not in final_outcomes:
            final_outcomes[observation] = 1
        else:
            final_outcomes[observation] += 1
        simulator.restart()

    options = stormpy.BuilderOptions([])
    options.set_build_state_valuations()
    model = stormpy.build_sparse_model_with_options(g_prism_program, options)
    simulator = stormpy.simulator.create_simulator(model, seed=42)
    simulator.set_observation_mode(
        stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL
    )
    final_outcomes = dict()
    for n in range(1000):
        observation = None
        while not simulator.is_done():
            observation, reward = simulator.step()
        if observation not in final_outcomes:
            final_outcomes[observation] = 1
        else:
            final_outcomes[observation] += 1
        simulator.restart()
    print(", ".join([f"{str(k)}: {v}" for k, v in final_outcomes.items()]))


def _do_simulation_task(rnseed, step_count):
    simulator = stormpy.simulator.create_simulator(g_model, seed=rnseed)
    simulator.set_observation_mode(
        stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL
    )
    final_outcomes = dict()
    for n in range(step_count):
        observation = None
        while not simulator.is_done():
            observation, reward = simulator.step()
        if observation not in final_outcomes:
            final_outcomes[observation] = 1
        else:
            final_outcomes[observation] += 1
        simulator.restart()
    return final_outcomes


def example_simulator_threaded():
    tasks = [1000] * 10
    seeds = [random.randint(1, 10000) for i in range(0, len(tasks))]
    final_outcomes: dict = {}
    with mp.Pool(processes=(mp.cpu_count() + 1)) as ppool:
        results = ppool.starmap(_do_simulation_task, zip(seeds, tasks))
        for (k, v) in results:
            if k not in final_outcomes:
                final_outcomes[k] = 1
            else:
                final_outcomes[k] += 1
    print(", ".join([f"{str(k)}: {v}" for k, v in final_outcomes.items()]))


if __name__ == "__main__":
    example_simulator_01()
    example_simulator_threaded()