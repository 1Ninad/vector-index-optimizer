from typing import Dict
from src.data.models import Index


class CostEstimator:

    def __init__(self, a_per_col: Dict[int, float], b_per_col: Dict[int, float]):
        self.a = a_per_col
        self.b = b_per_col

    def estimate_num_dist(self, index: Index, ek: int) -> float:
        """
        numDist_estimate = a_avg * ek + b_avg
        Multi-column indexes use average coefficients.
        """

        vids = list(index.vid)

        a_avg = sum(self.a[c] for c in vids) / len(vids)
        b_avg = sum(self.b[c] for c in vids) / len(vids)

        return a_avg * ek + b_avg