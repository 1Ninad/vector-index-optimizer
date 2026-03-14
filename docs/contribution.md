# Contribution — Work Distribution

4 developers. Work split by implementation complexity, not file count. Each developer owns their components end-to-end: design the internals, write the code, make sure it connects correctly to the interfaces described in `architecture.md`.

Read `architecture.md` fully before starting. The models, formulas, and caching rules in there are binding — not suggestions.

---

## Developer 1 — Data Layer + Config + Pipeline Wiring

**Owns:** `src/data/`, `config.py`, `main.py`

**What to build:**

`src/data/` — Data models and disk I/O.
- Define all 5 dataclasses: `Index`, `Query`, `Configuration`, `WorkloadEntry`, `QueryPlan`. Fields and types are specified exactly in the "Data Models" section of `architecture.md`. Get these right — every other developer depends on them.
- Write the loader: reads the HDF5 database file (one dataset per column, shape `num_items × col_dim`), reads the workload file, returns typed objects. Use `h5py`.

`config.py` — All constants in one place.
- `di = 2`, `se = 2`, `im = 0.05`, `k_prime = 5`, `sample_frac = 0.01`, `hnsw_max_degree = 16`, `distance = "cosine"`, `recall_metric_k = 100`
- `theta_recall`: 0.90 for large datasets, 0.97 for small — expose both, let `main.py` pick.
- `beam_width`: set a reasonable starting default (e.g., 5); it can be tuned later.
- File paths: `DATA_DIR`, `INDEX_DIR` (`data/indexes/`).

`main.py` — Pipeline wiring only. No logic here.
- Calls loader → trainer → candidate_gen → beam_search → index_builder, in order.
- Also contains the query execution loop (serving time): load cached QueryPlan, scan each index with its ek, union results, re-rank by full combined score, return top-k. Full steps in "Component 6" of `architecture.md`.

**Why this scope:** the models are the foundation — if they're wrong, everything else breaks. The pipeline wiring and query execution add real complexity. Together this is a full workload.

**Interfaces you expose to others:**
- All 5 dataclasses (used by every developer)
- `load_database(path) -> dict[int, ndarray]` — column_id → item vectors
- `load_workload(path) -> list[WorkloadEntry]`

---

## Developer 2 — Estimator Training + Cost & Recall Estimators

**Owns:** `src/estimators/`

**What to build:**

Trainer:
- Sample 1% of database rows (random, reproducible with a seed).
- For each column: build one small HNSW index on those sampled rows using `hnswlib`. Distance = cosine. Max degree from `config.py`.
- Run sample training queries against each small index at multiple ek values (ek >= 100 only — the linear relationship only holds above this threshold).
- Record actual `numDist` per (column, ek) observation. `numDist` = number of distance computations; get this from hnswlib's search stats.
- Also run a brute-force scan on the sample (numpy dot products over all sample rows) to get exact ground truth per query.
- Record actual recall per (column, ek) observation: `|gt ∩ retrieved| / |gt|`.

Cost Estimator:
- Fit `numDist = a * ek + b` per column using sklearn `LinearRegression`.
- Store `a` and `b` per column.
- For multi-column indexes: average `a` and average `b` across all single-column models. Do not fit per combination.
- Expose: `estimate_num_dist(index: Index, ek: int) -> float`

Recall Estimator:
- Fit `recall = a * log(ek) + b` per column using sklearn `LinearRegression` (log-transform ek before fitting).
- Store `a` and `b` per column.
- Same averaging rule for multi-column indexes.
- Expose: `estimate_recall(index: Index, ek: int) -> float`

**Why this scope:** building sample indexes, running queries, brute-forcing ground truth, and fitting two statistical models with the multi-column averaging logic is substantial work that requires careful implementation.

**Interfaces you expose to others:**
- `CostEstimator` with `estimate_num_dist(index, ek) -> float`
- `RecallEstimator` with `estimate_recall(index, ek) -> float`
- `train(database, workload) -> (CostEstimator, RecallEstimator)` — called once by `main.py`

---

## Developer 3 — Query Planner (Algorithm 1 + Algorithm 2)

**Owns:** `src/planner/`

**What to build:**

Dispatcher:
- Takes: query `q`, configuration `X`, `CostEstimator`, `RecallEstimator`, ground truth `gt(q)`.
- If `|X| <= 3`: call Algorithm 1. If `|X| > 3`: call Algorithm 2.
- Returns: `QueryPlan`.

Algorithm 1 — Exhaustive Search:
- For each index `x_i`: find the rank of every ground truth item in `x_i` by scanning it (partial score over `x_i`'s columns). Build `ek_i_relevant = sorted([0] + [rank_j for j in gt(q)])`.
- Enumerate all combinations of relevant ek values across indexes 1 through `|X|-1`.
- For the last index: find minimum ek that fills the recall gap left by others. Start at k, decrease.
- Compute `cost_plan` for each valid combination (formula in "Cost Model" section of `architecture.md`).
- Return the `QueryPlan` with lowest cost that meets `theta_recall`.
- Complexity: O(k × sum(x_i.dim) + |X| × k × log(k) + k^(|X|−1))

Algorithm 2 — DP:
- Sample k'=5 items from `gt(q)`. Run multiple samples, average the resulting QueryPlans.
- Enumerate all 2^5 = 32 subsets of those k' items (use bitmasks).
- Build DP table: `DP[i][cover]` = min cost to cover `cover` using first i indexes.
- Base case: `DP[1][cover] = cost_cover(cover, x_1)`
- Recurrence: `DP[i][cover] = min over cvr ⊆ cover: DP[i-1][cover - cvr] + cost_cover(cvr, x_i)`
- Helper: `cost_cover(cvr, x_i) = max over j in cvr: cost_idx(q, x_i, ek_i_j)` — cost of retrieving all items in cvr from x_i = cost at the deepest-ranked item among them.
- Track `EK[i][cover]` to reconstruct actual ek values.
- Find cover where `|cover| >= k' * theta_recall` with minimum `DP[|X|][cover]`. Return corresponding `QueryPlan`.
- Complexity: O(sum(x_i.dim) + |X|)

All formulas are in "Component 2: Query Planner" in `architecture.md`.

**Why this scope:** two distinct algorithms, one of which is a bitmask DP with a non-obvious recurrence and a custom helper function. This is the most algorithmically dense part of the system.

**Interfaces you expose to others:**
- `plan_query(query, config, cost_estimator, recall_estimator, ground_truth) -> QueryPlan`
- This is called by the beam search for every (query, configuration) pair.

---

## Developer 4 — Configuration Searcher + Storage Estimator + Index Builder

**Owns:** `src/searcher/`, `src/storage/`, `src/index_builder/`

**What to build:**

Candidate generation:
- For each query `q_i` in the workload: generate all column subsets of `q_i.vid` with size >= `|q_i.vid| - di`. Each subset becomes a candidate `Index` object (`vid` = subset, `dim` = sum of those column dims).
- Union all per-query candidates → candidate pool (deduplicated by vid).
- For seeds: for each query, generate all subsets of `Cand(q_i)` with at most `se` indexes. Union all → seed pool.

Beam search:
- Evaluate all seeds: for each seed config, call `plan_query` for every workload query, compute total weighted cost `sum(p_i * cost_i)`. Keep `b` lowest-cost valid seeds as initial beam.
- Expand loop: for each config in beam, for each candidate index, try adding it. Prune: remove any index where no query satisfies `x.vid ⊆ q.vid AND |x.vid| >= |q.vid| - di`. Evaluate expanded config. Repeat until cost improvement < `im`.
- Query plan cache: key = `(query_id, frozenset(relevant index vids))`. Hit → reuse cached QueryPlan, skip planner call.
- ek cache: key = `(query_id, frozenset(index.vid))`. Pass cached ek to planner to skip recomputation.
- Return lowest-cost configuration seen across all iterations.

Storage Estimator:
- Single function: takes a `Configuration`, returns `(count: int, valid: bool)` where `valid = count <= theta_storage`. That's it.

Index Builder:
- For each `Index` in X*: concatenate vectors of all covered columns per item (column order = sorted by column ID for determinism), build HNSW index over concatenated vectors using `hnswlib`. Distance = cosine. Max degree from `config.py`.
- Save each index to `data/indexes/<frozenset_of_vid>.bin` (filename encodes which columns it covers).
- Query function: load a saved index file, accept a concatenated query vector, return `(item_ids, scores)` for the top-`ek` nearest neighbors.

All caching rules, pruning conditions, and loop structure are in "Component 3" of `architecture.md`.

**Why this scope:** candidate generation + beam search with two caches + storage check + full index build is a large but well-defined scope. The storage estimator is trivial but logically belongs here since it's called directly by the beam search.

**Interfaces you expose to others:**
- `generate_candidates(workload, col_dims) -> (candidate_pool, seed_pool)`
- `beam_search(workload, candidate_pool, seed_pool, planner_fn, storage_fn, estimators) -> Configuration`
- `build_indexes(config, database) -> None` (writes to disk)
- `query_index(index_path, query_vector, ek) -> (item_ids, scores)`

---

## Interface Summary

These are the boundaries between developers. Agree on these before writing any code.

| From | To | What is passed |
|---|---|---|
| Dev 1 (`data`) | Everyone | `Index`, `Query`, `Configuration`, `WorkloadEntry`, `QueryPlan` dataclasses |
| Dev 1 (`loader`) | Dev 2, Dev 4 | `load_database()`, `load_workload()` |
| Dev 2 (`estimators`) | Dev 3, Dev 4 | `CostEstimator`, `RecallEstimator` objects |
| Dev 3 (`planner`) | Dev 4 | `plan_query(query, config, cost_est, recall_est, gt) -> QueryPlan` |
| Dev 4 (`searcher`) | Dev 1 (`main`) | Best `Configuration` X* |
| Dev 4 (`index_builder`) | Dev 1 (`main`) | Built index files on disk; `query_index()` function |

**Dev 1 writes the models first. All others wait on that before writing any code.**
