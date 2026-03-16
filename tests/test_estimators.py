"""
Smoke tests for Member 2's estimator layer.
Run from project root: python -m pytest tests/ -v

These tests verify:
- estimator training completes
- estimator APIs behave correctly
- outputs satisfy architectural constraints
"""

import numpy as np
import pytest

from src.estimators.trainer import train
from src.estimators.cost_estimator import CostEstimator
from src.estimators.recall_estimator import RecallEstimator
from src.data.models import Query, WorkloadEntry, Index


# Small synthetic DB so tests run fast even with HNSW
@pytest.fixture(scope="session")
def small_data():

    col_data = {
        0: np.random.randn(300, 50).astype(np.float32),
        1: np.random.randn(300, 100).astype(np.float32),
    }

    workload = []

    for _ in range(5):

        vid = frozenset({0, 1})

        vectors = {
            0: np.random.randn(50).astype(np.float32),
            1: np.random.randn(100).astype(np.float32),
        }

        q = Query(
            vid=vid,
            vectors=vectors,
            k=20,
            dim=150
        )

        workload.append(
            WorkloadEntry(query=q, probability=1/5)
        )

    return col_data, workload


class TestEstimatorTraining:

    def test_train_returns_objects(self, small_data):
        col_data, workload = small_data

        cost_est, recall_est = train(col_data, workload)

        assert isinstance(cost_est, CostEstimator)
        assert isinstance(recall_est, RecallEstimator)


class TestCostEstimator:

    def test_numdist_positive(self, small_data):
        col_data, workload = small_data
        cost_est, _ = train(col_data, workload)

        idx = Index(frozenset({0}), 50)

        val = cost_est.estimate_num_dist(idx, 150)

        assert val > 0

    def test_multi_column_index_supported(self, small_data):
        col_data, workload = small_data
        cost_est, _ = train(col_data, workload)

        idx = Index(frozenset({0, 1}), 150)

        val = cost_est.estimate_num_dist(idx, 200)

        assert isinstance(val, float)


class TestRecallEstimator:

    def test_recall_in_range(self, small_data):
        col_data, workload = small_data
        _, recall_est = train(col_data, workload)

        idx = Index(frozenset({1}), 100)

        r = recall_est.estimate_recall(idx, 150)

        assert 0.0 <= r <= 1.0

    def test_recall_monotonic_with_ek(self, small_data):
        col_data, workload = small_data
        _, recall_est = train(col_data, workload)

        idx = Index(frozenset({1}), 100)

        r1 = recall_est.estimate_recall(idx, 120)
        r2 = recall_est.estimate_recall(idx, 250)

        assert r2 >= r1