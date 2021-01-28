from typing import List, Tuple, Dict, Optional
import stormpy
import stormpy.core
import stormpy.pars
import stormpy.simulator

import numpy as np
import scipy as sp
from scipy.stats import multinomial as sp_multinomial

import matplotlib.pyplot as plt
from matplotlib import cm

from scripts.prism_stats_mc import PrismStatsMc
from scripts.my_prism_program import MyPrismProgram
from scripts.my_config import MyConfig


class AbcSmc2(object):
    """In case of likelihood can be cheaply calculated (closed form solution of the property is known)"""

    def __init__(
        self,
        prism_model_file: str,
        prism_props_file: str,
        obs_labels: List[str],
        obs_data: List[int],
        particle_count: int,
        perturbation_count: int,
        check_threshold: float,
    ) -> None:
        super().__init__()
        self.rng: np.random.Generator = np.random.default_rng()
        # PRISM model configuration
        self.my_prism_program = MyPrismProgram(prism_model_file, prism_props_file)
        self.model = stormpy.build_parametric_model(
            self.my_prism_program.prism_program,
            self.my_prism_program.prism_props,
        )
        self.state_mapping: Dict[int, List[str]] = {}
        for state in self.model.states:
            self.state_mapping[state.id] = [label for label in state.labels]
        print(self.state_mapping)
        self.model_parameters = self.model.collect_probability_parameters()
        self.instantiator = stormpy.pars.PDtmcInstantiator(self.model)
        self.instantiated_model = None
        # Properties for checking and observing
        self.check_prop = self.my_prism_program.prism_props[0]
        self.obs_labels = (
            obs_labels  # observational states must have labels for simulation
        )
        self.obs_props = self.my_prism_program.prism_props[1:]
        self.obs_data: np.array = np.array(obs_data)
        self.obs_data_stats: np.array = np.array(obs_data)
        # Sequential Monte Carlo configuration
        self.param_space_sample: List[Tuple[np.array, float]] = []
        self.particle_count: int = particle_count
        self.perturbation_count: int = perturbation_count
        self.check_threshold: float = check_threshold
        self.abs_threshold: float = 0.5
        self.current_param = np.zeros(len(self.model_parameters), dtype=np.float)
        # Statistical model checking configuration
        self.simulator: PrismStatsMc = PrismStatsMc(
            prism_exec=MyConfig.prism_exec(),
            model_file=prism_model_file,
            property_file=prism_props_file,
        )

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
        data_sum: int = sum(self.obs_data)
        self.obs_data_stats: np.array = self.obs_data * 1.0 / data_sum

    def _perturbate(self, param) -> np.array:
        # TODO: properly designed perturbation function; proof of convergence/KL distance
        new_param = np.zeros(len(param))
        for i, p_i in enumerate(param):
            alpha = beta = 1 / p_i
            new_param[i] = self.rng.beta(alpha, beta)
        return new_param

    def _smc(self):
        # Accepted point: points which satisfy the properties
        # Weight: likelihood to generate the observed data
        for m in range(0, self.particle_count):
            for _ in range(0, self.perturbation_count):
                candidate_param: np.array = None
                if m == 0:
                    param_dim = len(self.current_param)
                    candidate_param = np.random.uniform(0, 1, param_dim)
                else:
                    candidate_param = self._perturbate(self.current_param)
                is_valid_candidate, dist = self._is_candidate_params_valid(
                    candidate_param
                )
                if not is_valid_candidate:
                    continue
                llh_prop = self._estimate_check_llh(candidate_param)
                if llh_prop < self.check_threshold:
                    continue
                print("Accept parameter point ", candidate_param)
                self.param_space_sample.append((candidate_param, dist))

    def _is_obs_state(self, state_idx: int) -> Tuple[bool, Optional[str]]:
        state_labels = self.state_mapping[state_idx]
        for label in state_labels:
            if label in self.obs_labels:
                return (True, label)
        return (False, None)

    def _simulate_obs(self, param: np.array):
        point = dict()
        for i, p in enumerate(self.model_parameters):
            point[p] = stormpy.RationalRF(param[i])
        instantiated_model = self.instantiator.instantiate(point)
        simulator = stormpy.simulator.create_simulator(instantiated_model, seed=42)
        final_outcomes = dict()
        for _ in range(1000):
            observation = None
            while not simulator.is_done():
                observation, _ = simulator.step()  # reward in place hodler
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()
        summary_stats = np.zeros(len(self.obs_data))
        for k, v in final_outcomes.items():
            is_obs_state, label = self._is_obs_state(k)
            if is_obs_state:
                summary_stats[self.obs_labels.index(label)] = v

        summary_stats = summary_stats * 1.0 / np.sum(summary_stats)
        return summary_stats

    def _estimate_check_llh(self, param: np.array):
        smc_res = self.simulator.run(self.model_params_to_prism_cmd_args(param))
        print("StatsModelCheck P = ", smc_res)
        return smc_res

    def _is_candidate_params_valid(self, param: np.array):
        for _p in param:
            if _p < 0 or _p > 1:
                return False

        # Storm model simulation
        summary_stats = self._simulate_obs(param)
        dist = np.linalg.norm(summary_stats - self.obs_data_stats)
        print("Dist(s_obs, s_hat) = ", dist)
        return (dist < self.abs_threshold, dist)

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
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/zeroconf.pm"
    )
    prism_props_file = (
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/zeroconf.pctl"
    )
    smc_rf = AbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_labels=["ok", "err"],
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
        print("Sampled param point: ", point)
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
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/die.pm"
    )

    prism_props_file = (
        "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/die.pctl"
    )
    smc_rf = AbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_labels=["one", "two", "three", "four", "five", "six"],
        obs_data=[60, 60, 60, 60, 60, 60],
        particle_count=10,
        perturbation_count=1,
        check_threshold=0.01,
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
    prism_model_file = "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/multi_sync_3_bees.pm"
    prism_props_file = "/home/huypn12/Work/UniKonstanz/MCSS/bbeess-py/examples/data/multi_sync_3_bees.pctl"
    smc_rf = AbcSmc2(
        prism_model_file=prism_model_file,
        prism_props_file=prism_props_file,
        obs_labels=["bscc_1", "bscc_2", "bscc_3", "bscc_4"],
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
    print(res)
    x = [p[0] for p in theta]
    y = [p[1] for p in theta]
    z = [p[2] for p in theta]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("multi params, 3 bees, sync", fontsize=14)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")

    points = ax.scatter(x, y, z, s=20, c=llh, marker="o", cmap=cm.jet)
    plt.colorbar(points)
    plt.savefig("multi-sync-3.png")


if __name__ == "__main__":
    main2()