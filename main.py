import os

import numpy as np

import config
from src.data.loader import load_database, load_workload
from src.estimators.trainer import train, _sample_database
from src.index_builder import build_indexes, index_filename, query_index
from src.planner.query_planner import plan_query
from src.searcher import beam_search, generate_candidates
from src.searcher.beam_search import compute_ground_truth


def serve_query(query, query_plan, col_data):
    """
    Execute one query at serving time using its pre-computed plan.

    Steps (Component 6 from architecture):
      1. For each index in the plan with ek > 0: scan the built index, retrieve ek items.
      2. Union all retrieved item ID sets.
      3. Re-rank candidates by full combined cosine similarity across all query columns.
      4. Return top-k item IDs.
    """
    retrieved: set[int] = set()

    for index, ek in query_plan.ek_map.items():
        if ek == 0:
            continue
        col_ids = sorted(index.vid)
        query_vec = np.concatenate(
            [query.vectors[col_id] for col_id in col_ids]
        ).astype(np.float32)
        index_file = os.path.join(config.INDEX_DIR, index_filename(index))
        item_ids, _ = query_index(index_file, query_vec, ek)
        retrieved.update(int(i) for i in item_ids)

    if not retrieved:
        return []

    candidates = list(retrieved)
    scores = np.zeros(len(candidates), dtype=np.float64)

    for col_id in query.vid:
        q_vec = query.vectors[col_id].astype(np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-10
        col_matrix = col_data[col_id][candidates].astype(np.float32)
        db_norms = np.linalg.norm(col_matrix, axis=1) + 1e-10
        scores += (col_matrix @ q_vec) / (db_norms * q_norm)

    top_k = min(query.k, len(candidates))
    top_indices = np.argsort(-scores)[:top_k]
    return [candidates[i] for i in top_indices]


def main():
    # Step 1: Load all 6 database columns from disk.
    # Returns {col_id: numpy array of shape (num_items, dim)}, all truncated to 400K rows.
    col_data = load_database(config.DATA_DIR, config.DATASET_FILES)
    print(f"Loaded {len(col_data)} columns: "
          f"{{ {', '.join(f'{k}: {v.shape}' for k, v in col_data.items())} }}")

    # Step 2: Load or generate the synthetic workload (1000 queries, saved to data/workload.pkl).
    workload = load_workload(config.WORKLOAD_PATH, col_data)
    print(f"Workload: {len(workload)} queries")

    # Step 3: Train cost and recall estimators on a 1% sample of the database.
    print("Training estimators...")
    cost_est, recall_est = train(col_data, workload)
    print("Estimators trained.")

    # Step 4: Draw the same 1% sample for beam search and query planning.
    print("Sampling database for beam search...")
    sample_data = _sample_database(col_data)
    print(f"Sampled {next(iter(sample_data.values())).shape[0]} rows per column.")

    # Step 5: Generate candidate indexes and seed configurations from the workload.
    print("Generating candidates and seeds...")
    candidates, seeds = generate_candidates(workload)

    # Always include the PerColumn baseline as a seed: one single-column index per column.
    # This guarantees the beam search starts with at least one configuration that can
    # serve every query (single-column indexes are relevant for queries with ≤ DI+1 cols).
    from src.data.models import Configuration, Index as Idx
    per_column_seed = Configuration(indexes=[
        Idx(vid=frozenset({col_id}), dim=col_data[col_id].shape[1])
        for col_id in col_data
    ])
    seeds = [per_column_seed] + seeds
    print(f"Candidates: {len(candidates)}, Seeds: {len(seeds)} (includes PerColumn baseline)")

    # Step 6: Beam search — find the best index configuration X*.
    print("Running beam search...")
    x_star = beam_search(
        seeds=seeds,
        candidates=candidates,
        workload=workload,
        cost_estimator=cost_est,
        recall_estimator=recall_est,
        sample_data=sample_data,
        theta_storage=len(col_data),
        theta_recall=config.THETA_RECALL_LARGE,
    )
    print(f"Best configuration: {len(x_star.indexes)} index(es)")
    for idx in x_star.indexes:
        print(f"  vid={sorted(idx.vid)}, dim={idx.dim}")

    # Step 7: Build real full-scale HNSW indexes for X* on the full database.
    print("Building final indexes...")
    built_files = build_indexes(col_data, x_star)
    print("Built index files:")
    for path in built_files:
        print(f"  {path}")

    # Step 8: Compute the final query plan for every workload query against X*.
    # These plans are cached here and reused at serving time.
    print("Planning queries against final configuration...")
    query_plans = {}
    for query_id, entry in enumerate(workload):
        gt = compute_ground_truth(entry.query, sample_data, config.K)
        query_plans[query_id] = plan_query(
            entry.query,
            list(x_star.indexes),
            cost_est,
            recall_est,
            sample_data,
            gt,
            config.THETA_RECALL_LARGE,
        )
    print(f"Plans ready for {len(query_plans)} queries.")

    # Step 9: Serving-time demo — execute the first 3 workload queries using real indexes.
    print("\n--- Serving Time Demo (first 3 queries) ---")
    for query_id in range(min(3, len(workload))):
        entry = workload[query_id]
        plan = query_plans[query_id]
        results = serve_query(entry.query, plan, col_data)
        ek_summary = {tuple(sorted(idx.vid)): ek for idx, ek in plan.ek_map.items()}
        print(
            f"Query {query_id} | cols={sorted(entry.query.vid)} | "
            f"ek={ek_summary} | recall={plan.recall:.2f} | "
            f"top-5 item IDs: {results[:5]}"
        )


if __name__ == "__main__":
    main()