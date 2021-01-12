import subprocess


class PrismSmcWrapper(object):
    def __init__(self) -> None:
        super().__init__()
        self.prism_cmd = "prism"
        self.prism_args = []
