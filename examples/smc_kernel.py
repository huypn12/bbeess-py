import numpy as np
from typing import List


class LogitNormal:
    pass


class MulivariateLogitNormal:
    pass


class ExampleSmcKernel(object):
    def __init__(self) -> None:
        super().__init__()
        self.rng: np.random.Generator = np.random.default_rng()
        self.particle_count: int = 1000

    def gen_smc_univariate_beta(self) -> List[float]:
        trace = np.zeros(self.particle_count, dtype=float)
        trace[0] = self.rng.uniform(0, 1)
        for i in range(1, self.particle_count):
            alpha = beta = 1 / trace[i - 1]
            trace[i] = self.rng.beta(alpha, beta)
        return trace

    def gen_smc_univariate_logitnormal(self) -> List[float]:
        trace = np.zeros(self.particle_count, dtype=float)
        trace[0] = self.rng.uniform(0, 1)
        for i in range(1, self.particle_count):
            alpha = beta = 1 / trace[i - 1]
            trace[i] = self.rng.beta(alpha, beta)
        return trace

    def gen_smc_multivariate(self, pdim: int) -> List[float]:
        trace = np.zeros((self.particle_count, pdim), dtype=float)
        trace[0] = self.rng.uniform(0, 1, size=pdim)
        for i in range(1, self.particle_count):
            alpha = [(1 / p) for p in trace[i - 1]]
            trace[i] = self.rng.dirichlet(alpha)
        return trace


def main():
    example = ExampleSmcKernel()
    print("Beta kernel smc")
    trace = example.gen_smc_univariate()
    for i in range(0, len(trace)):
        print(trace[i])
    print("Dirichlet kernel smc")
    trace = example.gen_smc_multivariate(2)
    for i in range(0, len(trace)):
        print(trace[i])


if __name__ == "__main__":
    main()