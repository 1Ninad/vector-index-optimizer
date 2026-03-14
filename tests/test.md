# Handoff Checklist

Before telling the next member to start, run the tests for your layer and confirm all pass.
Each member adds their own test file to this folder when their work is done.

Run all tests from the project root:
```
python -m pytest tests/ -v
```

---

## Member 1 (Ninad) — Data Layer

Test file: `tests/test_data_layer.py`

What it checks:
- All 5 dataclasses import correctly and behave as expected (Index is hashable, vid is frozenset)
- All 6 database columns load with correct shapes, dimensions, and float32 dtype
- No NaN or Inf values in any column
- Workload has 1000 queries, probabilities sum to 1.0
- Every query's vectors match its vid, and dim field is correct
- Workload reloads identically from disk (pickle reproducibility)

All 17 tests must pass before handing off to Member 2.

---

## Member 2 — Estimators

Test file: `tests/test_estimators.py` (Member 2 creates this)

What to check before handing off to Member 3:
- `train(col_data, workload)` runs without error and returns a CostEstimator and RecallEstimator
- `CostEstimator.estimate_num_dist(index, ek)` returns a positive float for any valid Index and ek >= 100
- `RecallEstimator.estimate_recall(index, ek)` returns a float between 0.0 and 1.0
- Both estimators work for single-column indexes (one column in vid)
- Both estimators work for multi-column indexes (averaging rule — do not crash or return wrong type)
- Cost estimate increases as ek increases (linear model must have positive slope)
- Recall estimate increases as ek increases (log model must be monotonically increasing)

---

## Member 3 — Query Planner

Test file: `tests/test_planner.py` (Member 3 creates this)

What to check before handing off to Member 4:
- `plan_query(query, config, cost_estimator, recall_estimator, ground_truth)` returns a QueryPlan
- Returned QueryPlan has ek_map covering all indexes in the configuration
- Returned plan's estimated recall meets theta_recall (>= 0.90)
- Algorithm 1 is used when len(configuration.indexes) <= 3
- Algorithm 2 is used when len(configuration.indexes) > 3
- Both algorithms return a valid QueryPlan (no crash, correct types)
- ek values in ek_map are non-negative integers (0 = skip, >= 100 for active indexes)

---

## Member 4 — Searcher + Storage + Index Builder

Test file: `tests/test_searcher.py` (Member 4 creates this)

What to check before handing back to Member 1 to complete main.py:
- `generate_candidates(workload)` returns a non-empty list of Index objects
- `beam_search(...)` returns a Configuration with at least one Index
- Every index in the returned configuration has vid that is a frozenset of valid column IDs
- `StorageEstimator` returns valid=True when index count <= theta_storage
- `build_indexes(col_data, X_star)` creates index files in data/indexes/
- `query_index(index_file, query_vector, ek)` returns (item_ids, scores) of length ek
