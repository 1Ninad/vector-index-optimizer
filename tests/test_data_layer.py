"""
Smoke tests for Member 1's data layer.
Run from project root: python -m pytest tests/ -v
These tests verify that the foundation is correct before any other member builds on top of it.
"""
import os
import pickle
import numpy as np
import pytest

import config
from src.data.models import Index, Query, WorkloadEntry, Configuration, QueryPlan
from src.data.loader import load_database, load_workload


# Load database once for all tests — takes ~2 min on first run (GloVe parsing)
@pytest.fixture(scope="session")
def col_data():
    return load_database(config.DATA_DIR, config.DATASET_FILES)


@pytest.fixture(scope="session")
def workload(col_data):
    return load_workload(config.WORKLOAD_PATH, col_data)


class TestModels:
    def test_index_is_hashable(self):
        # Index must be usable as a dict key (beam search caches depend on this)
        x = Index(vid=frozenset({0, 1}), dim=150)
        d = {x: 42}
        assert d[x] == 42

    def test_index_vid_is_frozenset(self):
        x = Index(vid=frozenset({0}), dim=50)
        assert isinstance(x.vid, frozenset)

    def test_index_in_set(self):
        x1 = Index(vid=frozenset({0, 1}), dim=150)
        x2 = Index(vid=frozenset({0, 1}), dim=150)
        # two identical Index objects must be treated as equal
        assert x1 == x2
        assert len({x1, x2}) == 1

    def test_query_plan_ek_map(self):
        # QueryPlan.ek_map uses Index as key — must work without error
        x = Index(vid=frozenset({2}), dim=100)
        q = Query(vid=frozenset({2}), vectors={2: np.zeros(100)}, k=100, dim=100)
        plan = QueryPlan(query=q, ek_map={x: 200}, cost=1.0, recall=0.95)
        assert plan.ek_map[x] == 200

    def test_workload_entry_probability_type(self):
        q = Query(vid=frozenset({0}), vectors={0: np.zeros(50)}, k=100, dim=50)
        entry = WorkloadEntry(query=q, probability=0.001)
        assert isinstance(entry.probability, float)


class TestDatabase:
    def test_correct_number_of_columns(self, col_data):
        assert len(col_data) == len(config.DATASET_FILES)

    def test_all_columns_same_row_count(self, col_data):
        row_counts = [v.shape[0] for v in col_data.values()]
        assert len(set(row_counts)) == 1, f"Row counts differ: {row_counts}"

    def test_column_dimensions_match_config(self, col_data):
        for col_id, meta in config.DATASET_FILES.items():
            assert col_data[col_id].shape[1] == meta["dim"], (
                f"col {col_id}: expected dim {meta['dim']}, got {col_data[col_id].shape[1]}"
            )

    def test_all_columns_float32(self, col_data):
        for col_id, arr in col_data.items():
            assert arr.dtype == np.float32, f"col {col_id} dtype is {arr.dtype}, expected float32"

    def test_no_nan_or_inf(self, col_data):
        for col_id, arr in col_data.items():
            assert not np.isnan(arr).any(), f"col {col_id} contains NaN"
            assert not np.isinf(arr).any(), f"col {col_id} contains Inf"


class TestWorkload:
    def test_query_count(self, workload):
        assert len(workload) == config.NUM_QUERIES

    def test_probabilities_sum_to_one(self, workload):
        total = sum(e.probability for e in workload)
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected 1.0"

    def test_each_query_has_at_least_one_column(self, workload):
        for i, entry in enumerate(workload):
            assert len(entry.query.vid) >= 1, f"Query {i} has empty vid"

    def test_vectors_match_vid(self, workload):
        # every column in vid must have a corresponding vector
        for i, entry in enumerate(workload):
            assert set(entry.query.vectors.keys()) == set(entry.query.vid), (
                f"Query {i}: vid={entry.query.vid} but vectors keys={set(entry.query.vectors.keys())}"
            )

    def test_vector_shapes_match_config(self, workload, col_data):
        for i, entry in enumerate(workload):
            for col_id, vec in entry.query.vectors.items():
                expected_dim = col_data[col_id].shape[1]
                assert vec.shape == (expected_dim,), (
                    f"Query {i} col {col_id}: expected shape ({expected_dim},), got {vec.shape}"
                )

    def test_query_dim_matches_sum_of_col_dims(self, workload, col_data):
        for i, entry in enumerate(workload):
            expected = sum(col_data[c].shape[1] for c in entry.query.vid)
            assert entry.query.dim == expected, (
                f"Query {i}: dim={entry.query.dim}, expected {expected}"
            )

    def test_workload_pickle_is_reproducible(self, workload):
        # loading from disk must return the same workload (same vids in same order)
        with open(config.WORKLOAD_PATH, "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == len(workload)
        for i, (a, b) in enumerate(zip(workload, loaded)):
            assert a.query.vid == b.query.vid, f"Query {i} vid mismatch after pickle reload"
