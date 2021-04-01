import numpy as np


class ExperimentConfig(object):

    true_params = {
        "sir5_true_param": np.array([0.017246491978609703, 0.067786043574277]),
        "sir10_true_param": np.array([0.01099297054879006, 0.035355703902286616]),
        "sir10_observed_data": np.array(
            [563, 976, 1016, 909, 764, 696, 606, 565, 603, 855, 2447]
        ),
        "sir15_true_param": np.array([0.004132720013578173, 0.07217656035559976]),
        "sir15_observed_data": np.array(
            [0, 0, 0, 1, 4, 12, 29, 37, 96, 177, 283, 459, 619, 1078, 1845, 5360]
        ),
    }

    observed_data = {
        "sir5_observed_data": np.array([421, 834, 1126, 1362, 1851, 4406]),
    }
