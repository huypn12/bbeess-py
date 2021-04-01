import unittest

from datetime import datetime
import numpy as np


class TestEval(unittest.TestCase):
    def test_eval(self):
        begin_str = "2021-03-27 21:48:02.523528"
        begin_time = datetime.strptime(begin_str, "%Y-%m-%d %H:%M:%S.%f")
        end_str = "2021-03-28 04:20:26.457741"
        end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S.%f")
        print(end_time - begin_time)

    def test_l2dist(self):
        truep = np.array([0.02307652, 0.06481155])
        # p = np.array([0.01724649, 0.06778604])
        p = np.array([0.01758384, 0.06535699])
        d = np.linalg.norm(truep - p)
        print(d)