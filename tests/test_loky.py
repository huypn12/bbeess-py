import loky
import cloudpickle

gExecutor = loky.get_reusable_executor(max_workers=13, timeout=5)

import stormpy
import stormpy.core
import stormpy.pars


class TestLoky(object):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)

    def _task_noop(self, p: int):
        self.val = p

    def run(self):
        results = gExecutor.map(self._task_noop, range(30))
        n_workers = len(set(results))
        print("Results: ", results)
        print("Number of used processes: ", n_workers)


if __name__ == "__main__":
    lk = TestLoky(
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pm",
        "/home/huypn12/Works/mcss/bbeess-py/data/prism/sir_10_1_0.pctl",
    )
    cloudpickle.dumps(lk)
