# MINT Project Structure

## Overview

Each folder has one clear responsibility. Read any folder in isolation and you understand what it does. The naming follows the component names in `architecture.md` exactly.

**Tech stack:** Python 3.11+ · numpy · hnswlib · scikit-learn · scipy · h5py · stdlib dataclasses. Full rationale and version pinning in `architecture.md` under "Tech Stack". All developers use these exact libraries — no substitutions.

---

## Top-Level

### `main.py`
Entry point. Runs the full pipeline in order: load data → train estimators → generate candidates → run beam search → build real indexes. Also handles query execution at serving time.

### `config.py`
Single file for all constants and default parameter values: `di`, `se`, `im`, `k'`, `beam_width`, `sample_frac`, `theta_recall`, `theta_storage`, HNSW max degree, distance measure, file paths. Never hardcode these values elsewhere — always import from here.

---

## `src/` — All Source Code

### `src/data/`
**What:** Defines the shared data structures (models) and handles reading raw data from disk.

**Why its own folder:** Every other folder imports the models. Keeping them isolated here prevents circular imports and makes the vocabulary stable. Nothing in this folder imports from any other `src/` subfolder.

**What to build here:**
- Data classes for every core concept in the system: `Index`, `Query`, `Configuration`, `WorkloadEntry`, `QueryPlan`. Refer to the Data Models section of `architecture.md` for the exact fields each one needs.
- Functions to load the database (vectors per column) and workload from disk and return the above data objects.

---

### `src/estimators/`
**What:** Trains and stores the two statistical models (cost and recall) that let the searcher evaluate configurations without building real indexes.

**Why its own folder:** These models are trained once and then used by many other components (planner, beam search). Keeping training and inference logic together here makes them easy to swap out or retrain.

**What to build here:**
- Training orchestration: sample 1% of the database, build small single-column HNSW indexes on that sample, run sample queries, observe numDist vs ek and recall vs ek, fit models.
- Cost Estimator: stores linear model coefficients `(a, b)` per column. Given an index and an ek value, returns estimated `numDist`. For multi-column indexes, use averaged coefficients across all single-column models.
- Recall Estimator: stores log model coefficients `(a, b)` per column. Given an index and an ek value, returns estimated recall. Same averaging rule as Cost Estimator.

All formulas are in `architecture.md` under "Component 1: Training Phase".

---

### `src/planner/`
**What:** For one query + one configuration, finds the cheapest query plan (which indexes to use, what ek per index) that still meets the recall threshold.

**Why its own folder:** This is the hot path — called for every (query, configuration) pair during beam search. Keeping it isolated makes it easy to optimize or test independently.

**What to build here:**
- A dispatcher that looks at how many indexes are in the configuration and routes to either Algorithm 1 or Algorithm 2. Returns a `QueryPlan`.
- Algorithm 1 (exhaustive search): used when the configuration has 3 or fewer indexes. Uses the key observation that only k ek values matter per index. Full steps in `architecture.md` under "Algorithm 1".
- Algorithm 2 (DP): used when the configuration has more than 3 indexes. Samples k'=5 ground truth items, enumerates 32 subsets, runs DP. Full recurrence, base case, and `cost_cover` helper formula in `architecture.md` under "Algorithm 2".

Both algorithms use `CostEstimator` and `RecallEstimator` from `src/estimators/`.

---

### `src/searcher/`
**What:** The outer loop — explores the space of possible index configurations via beam search and returns the best one.

**Why its own folder:** This is the highest-level orchestration logic. It ties together candidate generation, the query planner, and the storage estimator into one search loop.

**What to build here:**
- Candidate index generation: for each query in the workload, generate all column subsets within `di` of the query size. Union all per-query candidates into one pool. Also generate seed configurations (subsets of candidates, size <= `se`).
- Beam search: evaluates seed configurations, keeps best `b`, expands by adding one candidate index at a time, prunes unused indexes, stops when improvement < `im`. Manages the query plan cache and ek cache (see "Caching" in `architecture.md`). Returns the best configuration X*.

The exact pruning rule, caching key definitions, and loop structure are all in `architecture.md` under "Component 3: Configuration Searcher".

---

### `src/index_builder/`
**What:** Builds real full-scale HNSW indexes on the full database after the best configuration is found.

**Why its own folder:** This step only runs once, after the search is complete. Keeping it separate ensures the search phase never accidentally triggers a full index build.

**What to build here:**
- For each index in X*: concatenate the vectors of all covered columns per item, build an HNSW index over those concatenated vectors using `hnswlib`, save to `data/indexes/`.
- A query function: load a saved index, accept a query vector, return item IDs and scores for the nearest neighbors.

Parameters (cosine similarity, max degree = 16) come from `config.py`.

---

### `src/storage/`
**What:** Checks whether a configuration is within the storage budget.

**Why its own folder:** The Storage Estimator is a distinct architectural component called independently by the beam search (separately from the Query Planner). Keeping it isolated makes the component boundary explicit.

**What to build here:**
- A function that takes a configuration, counts the number of indexes, and returns the count plus a boolean indicating whether `|X| <= theta_storage`.

Formula and constraint in `architecture.md` under "Component 4: Storage Estimator".

---

## `data/` — Data Files

Place raw dataset files here before running the system (vectors per column, workload file).

- `data/indexes/` — HNSW index files are written here after the index build step. Created automatically at runtime.

---

## Dependency Map

```
main.py
  |
  +--> src/data/              (read first; shared models, no dependencies on other src/ folders)
  +--> src/estimators/        (trains on sample; used by planner)
  |
  +--> src/searcher/          (outer loop)
  |       +--> src/planner/   (called per query per config)
  |       |       +--> src/estimators/
  |       +--> src/storage/
  |
  +--> src/index_builder/     (runs once after search, on full database)
```

**Write order for developers:** `src/data/` → `src/estimators/` → `src/planner/` → `src/storage/` → `src/searcher/` → `src/index_builder/` → `main.py`
