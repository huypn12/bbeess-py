class AbcSmc(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _distance(v1: np.array, v2: np.array) -> float:
        """Distance between two vector"""
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def _kernel(self, s1, s2, threshold):
        return 1 if (SmcRf._distance(s1, s2) > threshold) else 0
