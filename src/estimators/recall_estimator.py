import math
from typing import Dict
from src.data.models import Index


class RecallEstimator:

    def __init__(self, a_per_col: Dict[int, float], b_per_col: Dict[int, float]):
        self.a = a_per_col
        self.b = b_per_col

    def estimate_recall(self, index: Index, ek: int) -> float:

        vids = list(index.vid)

        a_avg = sum(self.a[c] for c in vids) / len(vids)
        b_avg = sum(self.b[c] for c in vids) / len(vids)

        return max(0.0, min(1.0, a_avg * math.log(ek) + b_avg))