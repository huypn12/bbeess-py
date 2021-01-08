import stormpy
import stormpy.core
import stormpy.pars

import pycarl
import pycarl.core

import numpy as np


class AbcSmc2(object):
    """ABC-SMC scheme with statistical model checking

    Args:
        object ([type]): [description]
    """

    def __init__(self) -> None:
        super().__init__()