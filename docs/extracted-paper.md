# MINT: Implementation Reference

---

## What the System Does

You have a database where each row is an item, each column is a feature, and each cell is a high-dimensional vector. Users send queries that target one or more columns and want the top-k items by combined score. The system figures out which indexes to build so queries run fast, while staying within storage and recall limits.

---

## Core Terminology

- **Item**: one row in the database
- **Column / Feature**: one embedding type (e.g., image vector, text vector)
- **Query q**: targets a subset of columns (`q.vid`), finds top-k items that maximize the sum of per-column scores
- **Score function**: dot product, cosine similarity, Euclidean distance, or Lp-norm between query vector and item vector, summed over all targeted columns
- **Index x**: an ANN index built on one or more columns. `x.vid` = which columns it covers, `x.dim` = total dimension of those columns
- **Configuration X**: a set of indexes
- **Query Plan (X, EK)**: which indexes to use and how many items (`ek_i`) to retrieve from each
- **ek (extended-k)**: how many items to retrieve from one index during a scan. `ek_i = 0` means skip that index
- **Recall**: fraction of true top-k items that appear in the union of retrieved items across all index scans
- **Workload W**: list of (query, probability) pairs. Probability = normalized frequency of that query

---

## Objective

Minimize probability-weighted average latency across the workload, subject to:
- Every query meets a minimum recall threshold
- Total number of indexes does not exceed a storage limit

Formally:

```
minimize  sum over (q_i, p_i) in W:  p_i * latency(q_i, X)

subject to:
  recall(q_i, X) >= theta_recall   for all queries
  storage(X) <= theta_storage
```

---

## System Architecture

Three components, each with a clear input/output:

```
Input: Database, Workload, Recall threshold, Storage threshold
         |
         v
[Configuration Searcher]
  - Explores candidate configurations (sets of indexes)
  - For each candidate, calls Query Planner to get estimated cost & recall
  - Calls Storage Estimator to check storage
  - Outputs the best configuration that satisfies constraints
         |
         +----> [Query Planner]
                  - Input: one query + one configuration
                  - Output: best query plan (which indexes to use, ek for each)
                  - Uses Cost Estimator + Recall Estimator internally
         |
         +----> [Storage Estimator]
                  - Input: configuration
                  - Output: storage used
         |
         v
Output: Recommended Configuration (set of indexes to build)
```

---

## Cost Model

Cost is a proxy for latency. Two parts:

**Index scan cost for index x_i retrieving ek_i items:**

```
cost_idx(q, x_i, ek_i) = x_i.dim * numDist(q, x_i, ek_i)
```

- `x_i.dim` = sum of dimensions of all columns in the index (e.g., a 2-column index with 100-dim and 128-dim vectors has x_i.dim = 228)
- `numDist` = number of distance computations (score function calls) during the index scan

**Re-ranking cost** (after collecting candidates from multiple indexes):

```
cost_rerank(q, EK) = q.dim * sum(ek_i for all i)
```

- `q.dim` = sum of dimensions of all columns the query touches
- Re-ranking computes full combined score for all retrieved candidates

**Total plan cost:**

```
cost_plan(q, X, EK) = sum(cost_idx(q, x_i, ek_i)) + cost_rerank(q, EK)
```

---

## Recall Model

Let `gt(q)` = ground truth top-k item IDs for query q.
Let `res(q, x_i, ek_i)` = item IDs retrieved from index x_i.

```
recall(q, X, EK) = |gt(q) ∩ union(res(q, x_i, ek_i) for all i)| / |gt(q)|
```

---

## Storage Estimator

For graph-based indexes (HNSW, DiskANN):

```
storage(x_i) proportional to:  num_items * max_degree * edge_size
```

In practice, storage is measured as the number of indexes. The paper uses this because, for a fixed dataset and max degree, all indexes consume roughly the same storage per index. The constraint is simply:

```
|X| <= theta_storage
```

where `theta_storage` = number of columns in the database (same budget as one-index-per-column baseline).

---

## Cost Estimator (How to Estimate numDist)

Graph-based indexes (HNSW, DiskANN) have no closed-form for `numDist`. You estimate it as follows:

1. Sample 1% of the database rows
2. Build single-column indexes on this sample for each column
3. Run a sample of training queries against these small indexes
4. Observe actual `numDist` vs `ek` for each (column, ek) pair
5. Fit a **linear model**: `numDist = a * ek + b` per column

This linear relationship holds for both HNSW and DiskANN when `ek >= 100` (verified on GloVe100, SIFT1M, Yandex Text-to-Image, Deep1M).

For multi-column indexes: do not train a separate model per column combination (too many combinations). Instead, use the **average slope and intercept** across all single-column models.

---

## Recall Estimator

Use the same 1% sample and training queries.

On the sample, you can do a full scan (brute force) to get ground truth since the dataset is small.

Observe actual recall vs ek. Fit a **logarithmic model**: `recall = a * log(ek) + b` per column.

Training time: 3 to 32 seconds depending on dataset.

---

## Query Planner: Finding the Best (X, EK) for a Query

Given a query q and a candidate configuration X (set of indexes), find EK (one ek_i per index) that minimizes cost while meeting recall threshold.

This problem is NP-hard (proven by reduction from Set Cover).

### Key Observation: Only k Values of ek_i Matter

For a query with top-k = k, the ground truth has k items. For each index x_i, only the rankings of those k ground truth items in x_i are relevant breakpoints for ek_i. Everything between two consecutive relevant ek values produces the same retrieval result, so you can skip them.

This limits the search space per index to k+1 values: {0, ek_i_1, ek_i_2, ..., ek_i_k} where 0 means skip the index.

### Algorithm 1: Search (for |X| <= 3)

Input: query q, top k, candidate indexes X, recall threshold, ground truth gt(q), cost function

Steps:

1. For each index x_i in X:
   - Find the rank of each ground truth item j in x_i (i.e., what position does it appear at when scanning x_i by partial score). Call this `ek_i_j`.
   - Store `ek_i_relevant = sorted([0] + [ek_i_j for j in 1..k])`

2. Enumerate all combinations of relevant ek values for indexes 1 through |X|-1.

3. For the last index |X|, instead of enumerating, compute the minimum ek that satisfies the recall threshold given what the other indexes already cover. As ek for other indexes increases, the required ek for the last index is non-increasing, so start at k and decrease.

4. For each valid combination, compute total cost. Keep the best (EK, cost) that meets recall threshold.

Time complexity: O(k * sum(x_i.dim) + |X|*k*log(k) + k^(|X|-1))

### Algorithm 2: Dynamic Programming (for |X| > 3)

The search algorithm's exponential factor k^(|X|-1) is too slow for many indexes. Replace with DP.

1. Sample the ground truth: take k' items from gt(q). k' is a small constant (the paper uses 5). Run multiple samples and average.

2. Enumerate the power set of the k' sampled ground truth items. Size = 2^k' = 32 when k'=5.

3. Define: `DP(i, cover)` = minimum cost to cover item subset `cover` using the first i indexes.

   Base case:
   ```
   DP(1, cover) = cost_cover(cover, x_1)
   ```

   Recurrence:
   ```
   DP(i, cover) = min over cvr ⊆ cover:
                    DP(i-1, cover - cvr) + cost_cover(cvr, x_i)
   ```

   where `cost_cover(cvr, x_i) = max over j in cvr: cost_idx(q, x_i, ek_i_j)`

   (The cost to retrieve all items in subset cvr from x_i equals the cost of retrieving up to the deepest-ranked one among them.)

4. Also track EK(i, cover) to reconstruct the actual ek values.

5. Find the cover with |cover| >= k' * theta_recall that minimizes DP(|X|, cover). Return the corresponding EK.

Time complexity with k'=5: O(sum(x_i.dim) + |X|), linear in dimensions and number of indexes.

**Decision rule in practice:** Use Algorithm 1 when |X| <= 3, Algorithm 2 when |X| > 3.

---

## Configuration Searcher: Beam Search

This is the outer loop that picks which indexes to actually build. Also NP-hard (proven by reduction from Densest k-Subgraph).

### Candidate Index Generation

For each query q_i, candidate indexes are all column subsets of q_i's columns with size >= (|q_i.vid| - di), where di is a subset-difference parameter (default di = 2).

Example: query on 5 columns, di=2 → candidates are all subsets of those 5 columns with at least 3 columns.

Union all per-query candidates to get the full candidate pool.

### Seed Configurations

For each query q_i, generate all subsets of Cand(q_i) with at most `se` indexes (default se = 2).
Union all per-query seed sets.

### Beam Search Loop

```
1. Evaluate all seed configurations using Query Planner
2. Keep the b lowest-cost seeds that satisfy both constraints (default b from experiments)
3. Repeat:
   a. For each configuration X in Best, for each candidate index x in Candidate pool:
      - Try X' = X ∪ {x}
      - Remove from X' any index that no query actually uses (prune unused)
      - If X' satisfies storage and recall constraints, add to Config set
   b. Best = b lowest-cost configs in Config
4. Until cost improvement < im (default im = 5%)
5. Return the lowest-cost configuration seen across all iterations
```

### Optimizations in the Beam Search

- If |Candidate| is small, evaluate more than b seeds to start with a better pool
- Cache query plans: if the plan for (q, X'_q) has been computed, reuse it. X'_q is the subset of X' relevant to query q (indexes whose columns are a subset of q's columns, with size >= |q.vid| - di)
- Cache ek values: if ek has been computed for a (query, index) pair, pass it directly to Query Planner instead of recomputing

---

## What-If Index Planning

The configuration searcher never builds real indexes during search. It uses hypothetical (sample-based) indexes:
- Build indexes on the 1% sample
- Use the cost and recall estimators trained on this sample to simulate what a real index would cost
- Only after the final configuration is selected do you build the real full-scale indexes

---

## Workload Format

Each query has:
- `q.vid`: set of column IDs the query targets
- `q.v_i`: the actual query vector for column i
- probability p_i (normalized frequency; all probabilities sum to 1)

---

## Default Parameter Values (from experiments)

| Parameter | Value | Reason |
|---|---|---|
| di (subset difference) | 2 | Allows indexes to be shared across queries without too many candidates |
| se (seed limit) | 2 | Ensures a big enough initial pool |
| im (improvement threshold) | 5% | Stops when gains are likely noise |
| k' (DP sample size) | 5 | Keeps 2^k' = 32 states, fast enough, acceptable accuracy |
| Training data size | 1% of database | Fast training (3-32 sec), still representative |
| Max HNSW/DiskANN degree | 16 | Default; storage = num_items * 16 * edge_size |
| Recall constraint | 90% for large datasets, 97% for small | Makes the task non-trivial |
| Recall metric | recall@100 | |
| Distance measure | Cosine similarity | |

---

## Experimental Datasets and Workloads

**Semi-synthetic construction**: sample equal numbers of rows from multiple single-feature embedding datasets, treat each as one column.

Datasets used per column: GloVe25, GloVe50, GloVe100, GloVe200 (word embeddings), SIFT1M (image descriptors), Deep1M (deep image features), Music (audio), Yandex Text-to-Image (cross-modal).

**Workload generation**: for each query, each column has probability p of being included. p=0.3 for simpler queries, p=0.5 for more complex. Query probabilities drawn from uniform distribution then normalized.

**Real dataset (News)**: 0.1M rows, 4 columns (title, description, content, image). Image and title: CLIP embeddings (512-dim). Description and content: Nomic-BERT embeddings (768-dim).

---

## Baselines

- **PerColumn**: one index per column. Each query uses only the indexes for its queried columns, then merges and re-ranks. This is the main comparison target. Same storage budget as MINT.
- **PerQuery**: one index per query (indexes all queried columns together). Optimal latency lower bound but uses 33-50% more storage than allowed. Shown as reference only.

---

## End-to-End Flow Summary

```
1. Receive: database, workload W, theta_recall, theta_storage

2. Training phase (one-time per database):
   a. Sample 1% of rows
   b. Build single-column indexes on sample for each column
   c. Run sample queries, observe numDist vs ek → fit linear models
   d. Run full scans on sample → get ground truth → observe recall vs ek → fit log models

3. Configuration search:
   a. Generate candidate indexes from workload
   b. Generate seed configurations
   c. Beam search: evaluate each configuration by calling Query Planner for each query
      - Query Planner uses cost + recall estimators (from step 2)
      - Returns estimated cost and recall for best plan
   d. Iterate until convergence
   e. Return best configuration X*

4. Build real indexes for X* on full database

5. At query time:
   - Given query q, use cached plan from Query Planner
   - Scan each index in plan with corresponding ek
   - Union results, re-rank by full score, return top-k
```
