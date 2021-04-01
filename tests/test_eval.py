import unittest

import os, sys, threading, resource

resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
sys.setrecursionlimit(10 ** 9)
threading.stack_size(2 ** 29)


from asteval import Interpreter


class TestEval(unittest.TestCase):
    def setUp(self) -> None:
        self.x = 0.5
        self.expr_str = "+".join(["x**2 + 4*x**4 + 6*x**6"] * 100000)
        self.aeval = Interpreter()
        self.aeval.symtable["x"] = self.x
        self.ast_expr = self.aeval.parse(self.expr_str)

    def test_eval(self):
        x = self.x
        val = eval(self.expr_str)
        self.assertEqual(val, 59375)

    def test_asteval(self):
        val = self.aeval.run(self.ast_expr)
        self.assertEqual(val, 59375)