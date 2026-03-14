import config
from src.data.loader import load_database, load_workload


def main():
    # Step 1: Load all 6 database columns from disk.
    # Returns {col_id: numpy array of shape (num_items, dim)}, all truncated to 400K rows.
    col_data = load_database(config.DATA_DIR, config.DATASET_FILES)
    print(f"Loaded {len(col_data)} columns: { {k: v.shape for k, v in col_data.items()} }")

    # Step 2: Load or generate the synthetic workload.
    # 1000 queries, each targeting a random subset of columns. Saved to data/workload.pkl.
    workload = load_workload(config.WORKLOAD_PATH, col_data)
    print(f"Workload: {len(workload)} queries")

    # Steps 3-6 are wired in here after Member 4 hands off their code.
    # cost_est, recall_est = trainer.train(col_data, workload)
    # candidates, seeds = generate_candidates(workload)
    # X_star = beam_search(seeds, candidates, workload, cost_est, recall_est)
    # build_indexes(col_data, X_star)


if __name__ == "__main__":
    main()
