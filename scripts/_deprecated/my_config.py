from typing import Dict


class MyConfig(object):
    _prism_exec: str = "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/third_party/prism-4.6-linux64/bin/prism"

    @staticmethod
    def prism_exec() -> str:
        return MyConfig._prism_exec