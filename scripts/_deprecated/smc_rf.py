from typing import List, Dict, Tuple
import stormpy
import stormpy.core
import stormpy.pars

import numpy as np
import scipy as sp
from scipy.stats import multinomial as sp_multinomial

import matplotlib.pyplot as plt
from matplotlib import cm


class MyPrismProgram(object):
    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
    ) -> None:
        super().__init__()
        self.prism_model_file: str = prism_model_file
        self.prism_props_file: str = prism_props_file
        self.prism_program = stormpy.parse_prism_program(self.prism_model_file)
        props_str = self._load_props_file()
        self.prism_props = stormpy.parse_properties_for_prism_program(
            props_str, self.prism_program
        )

    def _load_props_file(self):
        lines = []
        with open(self.prism_props_file, "r") as fptr:
            lines = fptr.readlines()
        return ";".join(lines)

class McParticle(object):
    def __init__(self, dim: int, theta: np.array = None, weight: float=0) -> None:
        super().__init__()
        self.dim = dim
        self.theta: np.array = np.zeros(dim) if theta is None else theta
        self.weight = weight

class SequentialMc(object):
    def __init__(self,
        particle_dim: int,
        trace_len: int = 1000,
    ) -> None:
        super().__init__()
        self._trace: List[McParticle] = []
        self._trace_len: int = trace_len
        self._particle_dim: int = particle_dim
        self._covar: np.array = np.zeros((particle_dim, particle_dim))

    def _mh(self):
        pass

    def _update_covar(self):
        pass

    def _init_particle(self) -> np.array:
        pass

    def sample(self):
        """[summary]
        """
        pass

    def reset(self):
        """[summary]

        Returns:
            [type]: [description]
        """

class SmcRf(object):
    """In case of likelihood can be cheaply calculated (closed form solution of the property is known)"""

    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        obs_data: List[int]
    ) -> None:
        super().__init__()
        self.rng: np.random.Generator = np.random.default_rng()
        # PRISM model configuration
        self.my_prism_program = MyPrismProgram(prism_model_file, prism_props_file)
        self.model = stormpy.build_parametric_model(
            self.my_prism_program.prism_program,
            self.my_prism_program.prism_props,
        )
        self.model_parameters = self.model.collect_probability_parameters()
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None
        # Properties for checking and observing
        self.check_prop = self.my_prism_program.prism_props[0]
        self.check_rf = stormpy.model_checking(self.model, self.check_prop).at(
            self.model.initial_states[0]
        )
        self.obs_props = self.my_prism_program.prism_props[1:]
        self.obs_rf = [
            stormpy.model_checking(self.model, obs_prop).at(
                self.model.initial_states[0]
            )
            for obs_prop in self.obs_props
        ]
        self.obs_data = obs_data
        assert len(self.obs_data) == len(self.obs_rf)
        

    def _instantiate_pmodel(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        self.instantiated_model = self.instantiator.instantiate(point)

    def _init(self):
        sample_size = len(self.current_param)
        sampled_param = np.random.uniform(0, 1, sample_size)
        self.current_param = sampled_param
        self._instantiate_pmodel(sampled_param)

    def _perturbate(self, param: np.array) -> np.array:
        new_param = self._component_wise_uniform(param)
        return new_param

    def _pertubate_component_wise_uniform(self, particle: McParticle) -> List[McParticle]:

        pass

    def _smc(self):
        # Accepted point: points which satisfy the properties
        # Weight: likelihood to generate the observed data
        for m in range(0, self.particle_trace_len):
            for _ in range(0, self.perturbation_count):
                candidate_param: np.array = None
                if m == 0:
                    param_dim = len(self.current_param)
                    candidate_param = np.random.uniform(0, 1, param_dim)
                else:
                    candidate_param = self._perturbate(self.current_param)
                if not self._is_candidate_params_valid(candidate_param):
                    continue
                llh_obs = self._estimate_obs_llh(candidate_param)
                llh_prop = self._estimate_check_llh(candidate_param)
                if llh_prop < self.check_threshold:
                    continue
                self.param_space_sample.append((candidate_param, llh_obs))

    def _compute_llh_multinomial(self, P, data):
        N = sum(data)
        return np.sum(np.log(sp_multinomial(N, P).pmf(data)))

    def _estimate_obs_llh(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        P = [float(rf.evaluate(point)) for rf in self.obs_rf]
        return self._compute_llh_multinomial(P, self.obs_data)

    def _estimate_check_llh(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        return float(self.check_rf.evaluate(point))

    def _is_candidate_params_valid(self, p: np.array):
        for _p in p:
            if _p < 0 or _p > 1:
                return False
        return True

    def get_result(self):
        return self.param_space_sample

    def get_current_param(self):
        return self.current_param

    def model_params_to_prism_cmd_args(self, param: np.array):
        dims = []
        for i, p in enumerate(self.model_parameters):
            dim_str = str(p) + "=" + str(param[i])
            dims.append(dim_str)
        return ",".join(dims)

    def run(self):
        self._init()
        self._smc()


def main():
    prism_model_file = (
        "/mnt/home/huypn12/repos/bbeess-py/examples/data/zeroconf.pm"
    )
    prism_props_file = (
        "/mnt/home/huypn12/repos/bbeess-py/examples/data/zeroconf.pctl"
    )
    smc_rf = SmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_data=[30, 60],
        particle_count=100,
        perturbation_count=10,
        check_threshold=0.1,
    )
    smc_rf.run()
    res = smc_rf.get_result()
    x = []
    y = []
    z = []
    for point in res:
        print(point)
        pq = point[0]
        x.append(pq[0])
        y.append(pq[1])
        z.append(point[1])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Zeroconf v4, params (p,q)", fontsize=14)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")

    points = ax.scatter(x, y, s=20, c=z, marker="o", cmap=cm.jet)
    plt.colorbar(points)
    plt.savefig("zeroconf4-pq.png")


def main2():
    prism_model_file = (
        "/mnt/home/huypn12/repos/bbeess-py/examples/data/die.pm"
    )
    prism_props_file = (
        "/mnt/home/huypn12/repos/bbeess-py/examples/data/die.pctl"
    )
    smc_rf = SmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_data=[30, 60, 50, 32, 55, 44],
        particle_count=100,
        perturbation_count=10,
        check_threshold=0.0,
    )
    smc_rf.run()
    res = smc_rf.get_result()
    x = []
    z = []
    for point in res:
        print(point)
        x.append(point[0])
        z.append(point[1])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Knuth die, params (p)", fontsize=14)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")

    points = ax.scatter(x, z, s=20, c=z, marker="o", cmap=cm.jet)
    plt.colorbar(points)
    plt.savefig("die-p.png")


def main3():
    prism_model_file = "/mnt/home/huypn12/repos/bbeess-py/examples/data/multi_sync_3_bees.pm"
    prism_props_file = "/mnt/home/huypn12/repos/bbeess-py/examples/data/multi_sync_3_bees.pctl"
    smc_rf = SmcRf(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_data=[30, 60, 50, 44],
        particle_count=100,
        perturbation_count=10,
        check_threshold=0.0,
    )
    smc_rf.run()
    res = smc_rf.get_result()
    theta = []
    llh = []
    for point in res:
        theta.append(point[0])
        llh.append(point[1])
    x = [p[0] for p in theta]
    y = [p[1] for p in theta]
    z = [p[2] for p in theta]
    print(x)
    llh = np.array(llh)
    np.nan_to_num(-99999)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("multi params, 3 bees, sync", fontsize=14)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.set_zlabel("q", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")

    points = ax.scatter(x, y, z, s=20, c=llh, marker="o", cmap=cm.jet)
    plt.colorbar(points)
    plt.savefig("multi-sync-3.png")


if __name__ == "__main__":
    main3()