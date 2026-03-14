# MINT System Architecture

## What Is This System?

MINT (Multi-Vector Index Tuning) solves one problem: your database has rows of items, and each item carries several different kinds of vectors — like an image vector and a text vector. When a user searches, they want the top-k most relevant items scored across multiple vector columns at once. Scanning everything is too slow. MINT automatically figures out which approximate nearest neighbor (ANN) indexes to build so searches are fast, while keeping storage and result accuracy within limits.

---

## Tech Stack

Every member on the team uses these exact tools and libraries. No substitutions.

### Language
- **Python 3.11+**
  - Reason: all required scientific libraries (numpy, scipy, hnswlib, scikit-learn) are native Python. No benefit to mixing languages for this scale of system.

### Core Libraries

| Library | Version | Used for |
|---|---|---|
| `numpy` | latest stable | All vector math, array operations, dot products, cosine similarity |
| `hnswlib` | latest stable | Building and querying HNSW ANN indexes (both sample-based and full-scale) |
| `scikit-learn` | latest stable | Fitting linear and log regression models for Cost and Recall Estimators |
| `scipy` | latest stable | Combinatorics helpers (power sets, subset enumeration for DP) |
| `dataclasses` | stdlib | Defining `Index`, `Query`, `Configuration`, `WorkloadEntry`, `QueryPlan` models |
| `h5py` | latest stable | Reading/writing dataset files (vectors stored in HDF5 format — `.h5`) |

### No web framework, no database, no async
This system is an offline optimizer — it runs as a script, not a server. No Flask, FastAPI, SQLAlchemy, or async code anywhere.

### Data File Format
- **HDF5 (`.h5`)** via `h5py` — standard format for large vector datasets (same format used by ann-benchmarks, which is the benchmark suite this system is evaluated against).
- Database file: one dataset per column, keyed by column ID string. Shape: `(num_items, dim_of_column)`.
- Workload file: list of query entries, each with `vid`, vectors per column, and probability.

### Required Dataset
-

### Coding conventions
- Use `dataclasses.dataclass` (not plain dicts or namedtuples) for all model types.
- Type-hint every function signature. Use `numpy.ndarray` for vector types.
- All constants and defaults live in `config.py`. Import from there; never hardcode values.
- No external config files (no YAML, no JSON config) — `config.py` is the single source of truth.

---

## Terminology (use these exact terms everywhere in code)

| Term | Meaning |
|---|---|
| **Item** | One row in the database |
| **Column** | One vector type / embedding type (e.g., image, text). Each column has an integer ID. |
| **`vid`** | A set of column IDs. Every index and query has a `vid`. |
| **`dim`** | Total vector dimension. For an index covering columns with dims 100 and 128, `dim = 228`. |
| **Index `x`** | An ANN index built on one or more columns. Has `vid` and `dim`. |
| **Configuration `X`** | A set of indexes. |
| **`ek`** | "Extended-k" — how many items to retrieve from one index during a scan. `ek = 0` means skip that index. |
| **Query Plan** | For one query: which indexes to use and the `ek` value for each index. |
| **Workload `W`** | A list of (query, probability) pairs. Probabilities are normalized and sum to 1. |
| **`gt(q)`** | Ground truth: the exact top-k item IDs for query q (found by brute force). |
| **`numDist`** | Number of distance computations (calls to the score function) during one index scan. |
| **`theta_recall`** | Minimum recall every query must achieve. Default: 90% for large datasets, 97% for small. |
| **`theta_storage`** | Max number of indexes allowed. Default: number of columns in the database. |

---

## Objective

Minimize probability-weighted average latency across the workload:

```
minimize:   sum over (q_i, p_i) in W:  p_i * cost_plan(q_i, X, EK)

subject to:
  recall(q_i, X, EK) >= theta_recall   for every query q_i
  |X|                 <= theta_storage
```

Cost is used as a proxy for latency throughout. The system never directly measures wall-clock time during the search phase.

---

## The Big Picture

```
Database + Workload + theta_recall + theta_storage
           |
           v
  [ Training Phase ]  (run once per database)
    - Take 1% sample of database rows
    - Build one small single-column ANN index per column on that sample
    - Run sample queries; observe numDist vs ek  → fit linear model per column
    - Run brute-force scan on sample; observe recall vs ek → fit log model per column
           |
           v
  [ Configuration Searcher ]   — outer loop, beam search
    - Generates candidate indexes (column subsets from workload queries)
    - Generates seed configurations (small combos of candidates)
    - Beam search: iteratively tries adding indexes, keeps best b configs each round
           |
           +--> [ Query Planner ]  (called for every (query, config) pair)
           |      Finds cheapest EK that meets recall threshold
           |      Uses Cost Estimator + Recall Estimator internally
           |      Algorithm 1 when |X| <= 3, Algorithm 2 when |X| > 3
           |
           +--> [ Storage Estimator ]
                  Returns: count of indexes in config, and whether |X| <= theta_storage
           |
           v
  Best Configuration X*
           |
           v
  [ Index Builder ]
    - Builds real full-scale HNSW indexes for X* on the full database
    - Saves indexes to disk
           |
           v
  [ Query Execution ]  (at serving time)
    - Load cached QueryPlan for the incoming query
    - Scan each index with its ek value
    - Union all retrieved item IDs
    - Re-rank by full combined score across all query columns
    - Return top-k items
```

---

## Data Models

These are the shared data structures used by every component. Define them first; everything else depends on them.

### `Index`
```
vid : set of int       — column IDs this index covers
dim : int              — sum of dimensions of all covered columns
```

### `Query`
```
vid         : set of int          — column IDs this query targets
vectors     : dict[int, ndarray]  — query vector for each column ID in vid
k           : int                 — number of top results to return
dim         : int                 — sum of dimensions of all targeted columns (= q.dim)
```

### `WorkloadEntry`
```
query       : Query
probability : float    — normalized frequency; all entries in workload sum to 1.0
```

### `Configuration`
```
indexes : list of Index
```

### `QueryPlan`
```
query    : Query
ek_map   : dict[Index, int]   — ek value for each index (0 = skip this index)
cost     : float              — estimated total cost of this plan
recall   : float              — estimated recall of this plan
```

---

## Cost Model

Cost is a proxy for latency. It has two parts.

### Part 1 — Index scan cost (per index)

```
cost_idx(q, x_i, ek_i) = x_i.dim * numDist(q, x_i, ek_i)
```

- `x_i.dim` — total dimension of the index (sum of all column dims it covers)
- `numDist` — number of distance computations during the scan (estimated by Cost Estimator)

### Part 2 — Re-ranking cost (once, after all index scans)

```
cost_rerank(q, EK) = q.dim * sum(ek_i for all indexes i in the plan)
```

- `q.dim` — total dimension of all columns the query touches
- Re-ranking computes the full combined score for every retrieved candidate

### Total plan cost

```
cost_plan(q, X, EK) = sum(cost_idx(q, x_i, ek_i) for all i) + cost_rerank(q, EK)
```

---

## Recall Model

```
recall(q, X, EK) = |gt(q) ∩ union(res(q, x_i, ek_i) for all i)| / |gt(q)|
```

- `gt(q)` — set of true top-k item IDs (ground truth)
- `res(q, x_i, ek_i)` — set of item IDs returned by scanning index x_i with ek = ek_i
- Recall = fraction of true top-k items that appear in the combined results

---

## Component 1: Training Phase (Estimator Training)

Run once per database before the search begins. Uses only 1% of the database rows.

### Steps

1. Randomly sample 1% of rows from the full database.
2. For each column, build one small ANN index (HNSW) on those sampled rows.
3. Run a sample of training queries against these small indexes.
4. For each (column, ek) pair observed, record the actual `numDist`.
5. Fit a **linear model** per column: `numDist = a * ek + b`
6. Also run a brute-force (exact) scan on the sample to get ground truth.
7. For each (column, ek) pair observed, record the actual recall.
8. Fit a **logarithmic model** per column: `recall = a * log(ek) + b`

### Cost Estimator

- Stores: `a` and `b` per column (linear model coefficients)
- Input: an index `x`, an ek value
- Output: estimated `numDist`
- For **multi-column indexes**: use the **average `a` and average `b`** across all single-column models. Do not train separate models for every column combination.
- Formula: `numDist_estimate = a_avg * ek + b_avg`

### Recall Estimator

- Stores: `a` and `b` per column (log model coefficients)
- Input: an index `x`, an ek value
- Output: estimated recall fraction (0.0 to 1.0)
- Same averaging rule for multi-column indexes as Cost Estimator.
- Formula: `recall_estimate = a_avg * log(ek) + b_avg`

### Important constraint

The linear relationship for numDist holds reliably only when `ek >= 100`. Keep this in mind when generating ek values for training.

---

## Component 2: Query Planner

**Input:** one query `q`, one configuration `X` (set of indexes), trained Cost Estimator, trained Recall Estimator, ground truth `gt(q)`

**Output:** a `QueryPlan` — the cheapest EK assignment that meets `theta_recall`

### Key Observation — Only k values of ek matter per index

For a query with top-k = k, ground truth has exactly k items. For each index `x_i`, the only ek values that change the retrieval result are the ranks of those k items inside `x_i` — i.e., at what position each ground truth item appears when scanning `x_i` by partial score.

Everything between two consecutive ground-truth ranks produces identical results, so all intermediate ek values can be skipped.

This limits the meaningful ek values per index to at most k+1:
```
ek_i_relevant = sorted([0] + [rank of ground truth item j in index x_i, for j in 1..k])
```
`0` means skip the index entirely.

---

### Algorithm 1 — Exhaustive Search (use when |X| <= 3)

**Input:** query q, top-k k, indexes X, recall threshold, gt(q), cost function

**Steps:**

1. For each index `x_i` in X:
   - Find the rank of each ground truth item in `x_i` (scan x_i, find position of each gt item).
   - Compute `ek_i_relevant = sorted([0] + [rank_j for j in gt(q)])`

2. Enumerate all combinations of relevant ek values across indexes 1 through |X|-1.

3. For the **last index** (index |X|): instead of enumerating, find the minimum ek that satisfies the recall threshold given what the other indexes already cover. Since recall from the last index only needs to fill the remaining gap, start at k and decrease until threshold is no longer met.

4. For each valid combination, compute `cost_plan`. Keep the EK combination with the lowest cost that meets `theta_recall`.

5. Return the best `QueryPlan`.

**Time complexity:** O(k × sum(x_i.dim) + |X| × k × log(k) + k^(|X|−1))

---

### Algorithm 2 — Dynamic Programming (use when |X| > 3)

The exponential factor k^(|X|−1) in Algorithm 1 becomes too slow for many indexes.

**Steps:**

1. Sample k' = 5 items from `gt(q)`. Run multiple samples and average results.

2. Enumerate all 2^k' = 32 subsets of those k' sampled ground truth items. Each subset is a bitmask.

3. Define the DP:
   ```
   DP(i, cover) = minimum cost to cover item subset `cover` using the first i indexes
   ```

   **Base case** (first index):
   ```
   DP(1, cover) = cost_cover(cover, x_1)
   ```

   **Recurrence** (each subsequent index):
   ```
   DP(i, cover) = min over all subsets cvr ⊆ cover:
                    DP(i-1, cover - cvr) + cost_cover(cvr, x_i)
   ```

   **Helper — cost to cover a subset of items from one index:**
   ```
   cost_cover(cvr, x_i) = max over j in cvr: cost_idx(q, x_i, ek_i_j)
   ```
   Explanation: to retrieve all items in `cvr` from index `x_i`, you scan until the deepest-ranked one among them. That rank determines ek, and thus cost.

4. Also track `EK(i, cover)` to reconstruct the actual ek values after DP completes.

5. Find the cover where `|cover| >= k' * theta_recall` that minimizes `DP(|X|, cover)`. Return the corresponding EK as a `QueryPlan`.

**Time complexity with k'=5:** O(sum(x_i.dim) + |X|) — linear.

---

**Decision rule:** Use Algorithm 1 when |X| <= 3, Algorithm 2 when |X| > 3.

---

## Component 3: Configuration Searcher (Beam Search)

**Input:** full database, workload W, trained estimators, theta_recall, theta_storage

**Output:** best configuration X*

### Step 1 — Candidate Index Generation

For each query `q_i` in the workload:
- Generate all column subsets of `q_i.vid` with size >= `|q_i.vid| - di` (default `di = 2`).
- Each subset becomes a candidate index (with `vid` = that subset, `dim` = sum of those column dims).

Example: query on 5 columns, di=2 → all subsets of size 3, 4, or 5 from those columns.

Union all per-query candidates → full candidate pool.

### Step 2 — Seed Configurations

For each query `q_i`:
- Generate all subsets of `Cand(q_i)` with at most `se = 2` indexes.

Union all per-query seed sets → initial seed pool.

### Step 3 — Beam Search Loop

```
1. Evaluate all seed configurations:
   - For each seed config X:
       - For each query q_i in workload: call Query Planner → get cost_i, recall_i
       - Total cost = sum(p_i * cost_i for all queries)
       - Valid if: all queries meet theta_recall AND |X| <= theta_storage
   - Keep the b lowest-cost valid seeds → initial beam (Best)

2. Repeat:
   a. Expand: for each config X in Best, for each candidate index x in pool:
        - Try X' = X ∪ {x}
        - Prune: remove from X' any index that no query in W actually uses
          (an index is "used" by query q if x.vid ⊆ q.vid AND |x.vid| >= |q.vid| - di)
        - Evaluate X' (call Query Planner for each query, compute total cost)
        - If X' is valid (recall + storage constraints met): add to Config set
   b. Best = b lowest-cost configs in Config

3. Stop when cost improvement from one iteration to the next is < im = 5%

4. Return the lowest-cost configuration seen across all iterations
```

### Caching (critical for performance)

**Query plan cache:** key = (query_id, frozenset of relevant index vids). If this (query, index subset) pair has been evaluated before, reuse the cached QueryPlan. Do not call Query Planner again.

**ek cache:** key = (query_id, index_vid). If the ek value for a specific (query, index) pair was computed before, pass it directly to Query Planner to skip recomputation.

**What is the "relevant index subset" for a query?**
Given configuration X', the relevant indexes for query q are those where:
- `x.vid ⊆ q.vid` (the index only uses columns the query touches)
- `|x.vid| >= |q.vid| - di` (the index is not too much smaller than the query)

Only these indexes are passed to the Query Planner for query q.

### Beam search optimization

If the candidate pool is small (few candidates total), evaluate all seed configurations instead of just b of them at the start — this gives a better initial beam.

---

## Component 4: Storage Estimator

**Input:** a configuration X

**Output:** count of indexes in X, and boolean `count <= theta_storage`

The constraint is simply:
```
|X| <= theta_storage
```
where `theta_storage` = number of columns in the database.

For graph-based indexes (HNSW), all indexes consume roughly the same storage per index for a fixed dataset and max degree (default max degree = 16). So counting indexes is sufficient.

---

## Component 5: Index Builder

**Input:** full database, best configuration X*

**Output:** one built HNSW index per entry in X*, saved to disk

- Use `hnswlib` as the ANN library.
- For each index in X*: concatenate the vectors of all covered columns for each item, build the HNSW index over those concatenated vectors.
- Distance measure: cosine similarity.
- Default max degree: 16.
- Save each built index to `data/indexes/`.
- Also provide a query function: given an index file and a query vector, run a nearest-neighbor scan and return item IDs with scores.

**Important:** the Configuration Searcher never builds real full-scale indexes. It uses the sample-based cost and recall estimators to simulate indexes. Real indexes are only built after X* is finalized.

---

## Component 6: Query Execution (serving time)

For each incoming query q:

1. Load the cached `QueryPlan` for q (produced by Query Planner during search).
2. For each index in the plan with `ek > 0`:
   - Scan the index, retrieve `ek` items by partial score (partial = score over only the columns the index covers).
3. Union all retrieved item ID sets.
4. Re-rank all candidates: compute the full combined score across all `q.vid` columns for every candidate.
5. Return the top-k by full combined score.

---

## Default Parameters

| Parameter | Default | What it controls |
|---|---|---|
| `di` | 2 | Max column-count difference between a candidate index and the query it serves |
| `se` | 2 | Max indexes per seed configuration |
| `im` | 5% | Beam search stops when cost improvement drops below this |
| `k'` | 5 | Number of ground truth items sampled in Algorithm 2 DP; keeps 2^5 = 32 states |
| `beam_width` (b) | set by experiment | How many configurations to keep each beam search iteration |
| `sample_frac` | 1% | Fraction of database rows used for estimator training |
| `theta_recall` | 90% (large datasets) / 97% (small datasets) | Minimum recall per query |
| `theta_storage` | number of columns | Max number of indexes allowed |
| HNSW max degree | 16 | Graph degree; affects index quality and storage |
| Distance measure | cosine similarity | Used for all ANN index queries |
| Recall metric | recall@100 | Evaluated at k=100 |

---

## End-to-End Data Flow

```
main.py
  |
  +--> data/loader
  |      Reads database vectors and workload from data/
  |      Returns: list of item vectors per column, list of WorkloadEntry
  |
  +--> estimators/trainer
  |      Samples 1% of database
  |      Builds one small HNSW index per column on sample
  |      Runs sample queries → records numDist vs ek
  |      Fits linear model per column → CostEstimator (a, b per column)
  |      Runs brute-force scan → records recall vs ek
  |      Fits log model per column → RecallEstimator (a, b per column)
  |
  +--> searcher/candidate_gen
  |      For each query in workload: generates candidate indexes (column subsets within di)
  |      Generates seed configurations (subsets of candidates, size <= se)
  |
  +--> searcher/beam_search
  |      Beam search loop:
  |        For each config X:
  |          For each query q: calls query_planner → QueryPlan (cost, recall)
  |          Checks storage via storage_estimator
  |        Keeps best b configs, expands, iterates until < im improvement
  |      Returns X*
  |
  +--> index_builder
  |      Builds real HNSW indexes for every index in X* on full database
  |      Saves to data/indexes/
  |
  +--> query execution (at serving time)
         Load cached QueryPlan
         Scan each index with its ek
         Union → re-rank → return top-k
```

---

## Dependency Rule

The data models module is the shared vocabulary — it must not import from any other module in the project. Every other module imports from it. Stabilize it first before writing anything else.

---

## What-If Index Planning (important implementation note)

During the entire beam search phase, no real full-scale indexes are ever built. The estimators trained on the 1% sample are used to simulate what a real index would cost and recall. Only after X* is selected are real indexes built. This is what makes the search fast enough to be practical.
