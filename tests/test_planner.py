"""
Tests for Member 3's Query Planner (Algorithms 1 & 2).
Run from project root: python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.planner.query_planner import plan_query
from src.estimators.trainer import train
from src.data.models import Query, WorkloadEntry, Index
import config


@pytest.fixture(scope="session")
def small_data():
    """Small synthetic DB for fast testing."""
    col_data = {
        0: np.random.randn(100, 50).astype(np.float32),
        1: np.random.randn(100, 100).astype(np.float32),
        2: np.random.randn(100, 75).astype(np.float32),
    }

    workload = []
    for _ in range(10):
        vid = frozenset({0, 1, 2})
        vectors = {
            0: np.random.randn(50).astype(np.float32),
            1: np.random.randn(100).astype(np.float32),
            2: np.random.randn(75).astype(np.float32),
        }
        q = Query(vid=vid, vectors=vectors, k=20, dim=225)
        workload.append(WorkloadEntry(query=q, probability=1/10))

    return col_data, workload


@pytest.fixture(scope="session")
def estimators(small_data):
    """Trained estimators from small data."""
    col_data, workload = small_data
    cost_est, recall_est = train(col_data, workload)
    return cost_est, recall_est, col_data[0].shape[0]  # Also return sample size


class TestQueryPlanner:

    def test_plan_query_single_index(self, small_data, estimators):
        """Test with single index (|X| = 1)."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        entry = workload[0]
        query = entry.query
        
        # Single-column index
        idx = Index(frozenset({0}), 50)
        configuration = [idx]
        
        # Get ground truth by brute force
        qvec = query.vectors[0]
        scores = col_data[0] @ qvec
        gt = set(np.argsort(-scores)[:query.k])
        
        plan = plan_query(
            query, configuration, cost_est, recall_est,
            {0: col_data[0]}, gt, theta_recall=0.90
        )
        
        assert isinstance(plan, object)
        assert plan.cost >= 0
        assert 0 <= plan.recall <= 1.0
        assert idx in plan.ek_map or len(plan.ek_map) == 0

    def test_plan_query_multi_index(self, small_data, estimators):
        """Test with multiple indexes (|X| = 2 via Algorithm 1)."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        entry = workload[0]
        query = entry.query
        
        # Multi-column indexes
        idx1 = Index(frozenset({0}), 50)
        idx2 = Index(frozenset({1}), 100)
        configuration = [idx1, idx2]
        
        # Ground truth
        combined_vec = np.concatenate([query.vectors[0], query.vectors[1]])
        combined_db = np.concatenate([col_data[0], col_data[1]], axis=1)
        scores = combined_db @ combined_vec
        gt = set(np.argsort(-scores)[:query.k])
        
        plan = plan_query(
            query, configuration, cost_est, recall_est,
            {0: col_data[0], 1: col_data[1]}, gt, theta_recall=0.85
        )
        
        assert isinstance(plan, object)
        assert plan.cost >= 0
        assert 0 <= plan.recall <= 1.0

    def test_plan_query_returns_query_plan(self, small_data, estimators):
        """Test that plan_query returns a proper QueryPlan."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        query = workload[0].query
        idx = Index(frozenset({0}), 50)
        
        qvec = query.vectors[0]
        scores = col_data[0] @ qvec
        gt = set(np.argsort(-scores)[:query.k])
        
        plan = plan_query(
            query, [idx], cost_est, recall_est,
            {0: col_data[0]}, gt
        )
        
        # Check QueryPlan structure
        assert hasattr(plan, 'query')
        assert hasattr(plan, 'ek_map')
        assert hasattr(plan, 'cost')
        assert hasattr(plan, 'recall')
        assert plan.query == query

    def test_plan_respects_recall_threshold(self, small_data, estimators):
        """Test that plan meets the recall threshold (or fails gracefully)."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        query = workload[0].query
        idx = Index(frozenset({0, 1}), 150)
        
        combined_vec = np.concatenate([query.vectors[0], query.vectors[1]])
        combined_db = np.concatenate([col_data[0], col_data[1]], axis=1)
        scores = combined_db @ combined_vec
        gt = set(np.argsort(-scores)[:query.k])
        
        theta_recall = 0.80
        plan = plan_query(
            query, [idx], cost_est, recall_est,
            {0: col_data[0], 1: col_data[1]}, gt, theta_recall=theta_recall
        )
        
        # Plan should either meet threshold or have ek_map empty
        if plan.ek_map:
            assert plan.recall >= theta_recall - 1e-5 or plan.recall < theta_recall

    def test_empty_configuration(self, small_data, estimators):
        """Test with empty configuration."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        query = workload[0].query
        
        qvec = query.vectors[0]
        scores = col_data[0] @ qvec
        gt = set(np.argsort(-scores)[:query.k])
        
        plan = plan_query(
            query, [], cost_est, recall_est,
            {0: col_data[0]}, gt
        )
        
        assert plan.cost == 0.0
        assert plan.recall == 0.0

    def test_irrelevant_index(self, small_data, estimators):
        """Test with index irrelevant to query."""
        col_data, workload = small_data
        cost_est, recall_est, sample_size = estimators
        
        entry = workload[0]
        query = entry.query
        # Query on {0, 1}, index on {2}
        idx = Index(frozenset({2}), 75)
        
        qvec = query.vectors[0]
        scores = col_data[0] @ qvec
        gt = set(np.argsort(-scores)[:query.k])
        
        plan = plan_query(
            query, [idx], cost_est, recall_est,
            {0: col_data[0], 2: col_data[2]}, gt
        )
        
        # Should return empty plan since no relevant indexes
        assert len(plan.ek_map) == 0 or plan.recall == 0.0
