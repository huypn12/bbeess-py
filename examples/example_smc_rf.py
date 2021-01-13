import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from scripts.smc_rf import SmcRf


def main():
    prism_model_file = "examples/data/zeroconf.pm"
    prism_props_file = "examples/data/zeroconf.pctl"
    smc_rf = SmcRf(
        prism_model_file_path=prism_model_file,
        prism_props_file_path=prism_props_file,
        particle_count=100,
        pertubation_count=10,
        verify_threshold=0.01,
    )
    smc_rf.run()
