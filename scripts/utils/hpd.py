import numpy as np


def compute_hpd_univariate(trace, alpha):
    """
    Code was taken from PyMC3 project
    """
    n = len(trace)
    cred_mass = alpha
    x = np.sort(np.array(trace))
    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation")

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return np.array([hdi_min, hdi_max])


def compute_hpd_multivariate(
    particle_trace: np.array, particle_dim: np.array, alpha: float = 0.95
):
    params_hpd = np.zeros((particle_dim, 2))
    for i in range(0, particle_dim):
        trace = particle_trace[:, i]
        h = compute_hpd_univariate(trace, alpha)
        params_hpd[i] = h
    return params_hpd
