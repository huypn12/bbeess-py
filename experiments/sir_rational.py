import numpy as np

from scripts.prism.prism_model import AbstractModelRational


class SirRational(AbstractModelRational):
    def __init__(self, prism_model_file: str, prism_props_file: str) -> None:
        super().__init__(prism_model_file, prism_props_file)
        