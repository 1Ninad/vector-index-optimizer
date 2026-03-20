import config
from src.data.loader import load_database, load_workload
from src.estimators.trainer import train, _sample_database
from src.searcher import generate_candidates, beam_search
from src.index_builder import build_indexes

def main():
    # Step 1: Load all 6 database columns from disk.
    # Returns {col_id: numpy array of shape (num_items, dim)}, all truncated to 400K rows.
    col_data = load_database(config.DATA_DIR, config.DATASET_FILES)
    print(f"Loaded {len(col_data)} columns: {{ {', '.join(f'{k}: {v.shape}' for k, v in col_data.items())} }}")

    # Step 2: Load or generate the synthetic workload.
    # 1000 queries, each targeting a random subset of columns. Saved to data/workload.pkl.
    workload = load_workload(config.WORKLOAD_PATH, col_data)
    workload = workload[:1]
    print(f"Workload: {len(workload)} queries")

    # Step 3: Train estimators.
    print("Training estimators...")
    cost_est, recall_est = train(col_data, workload)
    print("Estimators trained.")

    # Step 4: Build the same sample database needed by beam search / planner.
    print("Sampling database for beam search...")
    sample_data = _sample_database(col_data)
    print(f"Sampled {next(iter(sample_data.values())).shape[0]} rows per column.")

    # Step 5: Generate candidates and seeds.
    print("Generating candidates and seeds...")
    candidates, seeds = generate_candidates(workload)
    print(f"Candidates: {len(candidates)}")
    print(f"Seeds: {len(seeds)}")

    # Step 6: Run beam search.
    print("Running beam search...")
    x_star = beam_search(
        seeds=seeds,
        candidates=candidates,
        workload=workload,
        cost_estimator=cost_est,
        recall_estimator=recall_est,
        sample_data=sample_data,
        theta_storage=6,
        theta_recall=config.THETA_RECALL_LARGE,
    )
    print(f"Best configuration has {len(x_star.indexes)} indexes:")
    for index in x_star.indexes:
        print(f"  vid={sorted(index.vid)}, dim={index.dim}")

    # Step 7: Build final indexes.
    print("Building final indexes...")
    built_files = build_indexes(col_data, x_star)
    print("Built index files:")
    for path in built_files:
        print(f"  {path}")

    # Steps 3-6 are wired in here after Member 4 hands off their code.
    # cost_est, recall_est = trainer.train(col_data, workload)
    # candidates, seeds = generate_candidates(workload)
    # X_star = beam_search(seeds, candidates, workload, cost_est, recall_est)
    # build_indexes(col_data, X_star)


if __name__ == "__main__":
    main()