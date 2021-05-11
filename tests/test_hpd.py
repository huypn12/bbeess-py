import unittest
import numpy as np
from scipy.stats import norm

from scripts.utils.hpd import compute_hpd_univariate, compute_hpd_multivariate


class TestMhRfUniformKernel(unittest.TestCase):
    def test_hpd_univariate(self):
        trace = norm.rvs(size=100)
        hpd = compute_hpd_univariate(trace, 0.95)
        print(hpd)

    def test_hpd_multivariate(self):
        dim = 5
        trace_len = 100
        trace = np.zeros((trace_len, dim))
        for i in range(0, dim):
            trace[:, i] = norm.rvs(size=100)
        hpd = compute_hpd_multivariate(trace, dim, 0.95)
        print(hpd)