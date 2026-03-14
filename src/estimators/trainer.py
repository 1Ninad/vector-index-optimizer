import numpy as np
import hnswlib
from sklearn.linear_model import LinearRegression
import config

from typing import Dict, List, Tuple
from src.data.models import WorkloadEntry, Index
from src.estimators.cost_estimator import CostEstimator
from src.estimators.recall_estimator import RecallEstimator


def _sample_database(col_data: Dict[int, np.ndarray]):

    rng = np.random.default_rng(seed=42)

    n = next(iter(col_data.values())).shape[0]
    sample_size = int(n * config.SAMPLE_FRAC)

    idx = rng.choice(n, size=sample_size, replace=False)

    sample = {c: arr[idx] for c, arr in col_data.items()}

    return sample


def _build_small_indexes(sample_data):

    indexes = {}

    for col_id, vectors in sample_data.items():

        dim = vectors.shape[1]

        index = hnswlib.Index(space=config.DISTANCE, dim=dim)
        index.init_index(
            max_elements=vectors.shape[0],
            ef_construction=200,
            M=config.HNSW_MAX_DEGREE
        )

        index.add_items(vectors)

        index.set_ef(200)

        indexes[col_id] = index

    return indexes


def _bruteforce_gt(query_vec, db_vecs, k):

    sims = db_vecs @ query_vec
    topk = np.argsort(-sims)[:k]

    return set(topk)


def train(
    col_data: Dict[int, np.ndarray],
    workload: List[WorkloadEntry]
) -> Tuple[CostEstimator, RecallEstimator]:

    sample_data = _sample_database(col_data)

    small_indexes = _build_small_indexes(sample_data)

    numdist_obs = {c: [] for c in sample_data}
    recall_obs = {c: [] for c in sample_data}

    ek_values = list(range(config.MIN_EK_TRAIN, config.MIN_EK_TRAIN + 500, 100))

    for entry in workload[:50]:   # use small subset for training speed

        q = entry.query

        for col_id in q.vid:

            qvec = q.vectors[col_id]

            gt = _bruteforce_gt(qvec, sample_data[col_id], q.k)

            for ek in ek_values:

                labels, _ = small_indexes[col_id].knn_query(qvec, k=ek)

                retrieved = set(labels[0])

                recall = len(gt & retrieved) / len(gt)

                # approximate numDist = ek (proxy)
                numdist_obs[col_id].append((ek, ek))
                recall_obs[col_id].append((ek, recall))

    # -------- fit cost models --------

    a_cost = {}
    b_cost = {}

    for col_id, obs in numdist_obs.items():

        X = np.array([[ek] for ek, _ in obs])
        y = np.array([nd for _, nd in obs])

        model = LinearRegression().fit(X, y)

        a_cost[col_id] = model.coef_[0]
        b_cost[col_id] = model.intercept_

    # -------- fit recall models --------

    a_rec = {}
    b_rec = {}

    for col_id, obs in recall_obs.items():

        X = np.array([[np.log(ek)] for ek, _ in obs])
        y = np.array([r for _, r in obs])

        model = LinearRegression().fit(X, y)

        a_rec[col_id] = model.coef_[0]
        b_rec[col_id] = model.intercept_

    cost_estimator = CostEstimator(a_cost, b_cost)
    recall_estimator = RecallEstimator(a_rec, b_rec)

    return cost_estimator, recall_estimator