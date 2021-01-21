import subprocess


class PrismSmcWrapper(object):
    def __init__(self) -> None:
        super().__init__()
        self.prism_path = ""
        self.prism_cmd = "prism"
        self.prism_args = []
        self.result = ""

    def run(self):
        return self._simulate()

    def get_result(self):
        return