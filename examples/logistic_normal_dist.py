import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import norm, rv_continuous


class LogitNormal(rv_continuous):
    def __init__(self, scale=1, loc=0):
        super().__init__(self)
        self.scale = scale
        self.loc = loc

    def _pdf(self, x):
        return norm.pdf(logit(x), loc=self.loc, scale=self.scale) / (x * (1 - x))


fig, ax = plt.subplots()
values = np.linspace(10e-10, 1 - 10e-10, 1000)
sigma, mu = 1.78, 0
ax.plot(values, LogitNormal(scale=sigma, loc=mu).pdf(values), label="subclassed")
ax.legend()
plt.savefig("logit-normal.png")