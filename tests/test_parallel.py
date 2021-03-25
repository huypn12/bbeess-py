import stormpy
import stormpy.core
import stormpy.pars
import stormpy.simulator
import stormpy.examples.files

import multiprocessing as mp


class TestMultiprocessing(object):
    def __init__(self):
        path = stormpy.examples.files.prism_dtmc_die
        self.prism_program = stormpy.parse_prism_program(path)
        self.model = stormpy.build_model(self.prism_program)
        simulator = stormpy.simulator.create_simulator(self.model, seed=42)
        final_outcomes = dict()
        for n in range(1000):
            while not simulator.is_done():
                observation, reward = simulator.step()
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()

    def _do_simulation_task(self, sample_count: int):
        simulator = stormpy.simulator.create_simulator(self.model, seed=42)
        final_outcomes = dict()
        for _ in range(sample_count):
            while not simulator.is_done():
                observation, _ = simulator.step()
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()
        return final_outcomes

    def _run_parallel_tasks(self):
        sample_count = 1000
        with mp.Pool(processes=mp.cpu_count()) as _pool:
            results = _pool.starmap(self._do_simulation_task, [sample_count] * 16)
        return results

    def run(self):
        results = self._run_parallel_tasks()


if __name__ == "__main__":
    test_mp = TestMultiprocessing()
    test_mp.run()