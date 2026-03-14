import os
import pickle
import numpy as np
import h5py
from typing import Dict, List

import config
from src.data.models import Query, WorkloadEntry


def _load_txt(path: str, dim: int) -> np.ndarray:
    """Load GloVe .txt file. Each line: 'word f1 f2 ... fd'. Skip the word token."""
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            # first part is the word, rest are floats
            vectors.append([float(x) for x in parts[1:]])
    return np.array(vectors, dtype=np.float32)


def _load_hdf5(path: str) -> np.ndarray:
    """Load ann-benchmarks HDF5 file. Database vectors are under the 'train' key."""
    with h5py.File(path, "r") as f:
        return np.array(f["train"], dtype=np.float32)


def _load_fbin(path: str, dim: int) -> np.ndarray:
    """
    Load .fbin or .bin file — headerless raw float32 array.
    num_items is derived from file size and the known column dimension.
    """
    file_size = os.path.getsize(path)
    num_items = file_size // (dim * 4)  # each item = dim float32 values = dim*4 bytes
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(num_items, dim)


def load_database(data_dir: str, dataset_files: dict) -> Dict[int, np.ndarray]:
    """
    Load all columns from disk. Returns {col_id: array of shape (num_items, dim)}.

    All columns are truncated to the same row count (the smallest across all columns)
    so they all represent the same set of items. GloVe (~400K rows) sets this floor.
    """
    columns: Dict[int, np.ndarray] = {}

    for col_id, meta in dataset_files.items():
        path = os.path.join(data_dir, meta["path"])
        fmt = meta["format"]
        dim = meta["dim"]
        print(f"Loading column {col_id} from {meta['path']} ...")

        if fmt == "txt":
            arr = _load_txt(path, dim)
        elif fmt == "hdf5":
            arr = _load_hdf5(path)
        elif fmt in ("fbin", "bin"):
            arr = _load_fbin(path, dim)
        else:
            raise ValueError(f"Unknown format '{fmt}' for column {col_id}")

        columns[col_id] = arr

    # Truncate all columns to the same number of rows
    min_rows = min(arr.shape[0] for arr in columns.values())
    for col_id in columns:
        columns[col_id] = columns[col_id][:min_rows]
        print(f"  col {col_id}: shape {columns[col_id].shape}")

    # create the indexes folder now so Member 4's index builder never hits a missing-dir error
    os.makedirs(config.INDEX_DIR, exist_ok=True)

    print(f"Database loaded: {len(columns)} columns, {min_rows} rows each.")
    return columns


def generate_workload(
    col_data: Dict[int, np.ndarray],
    num_queries: int,
    p: float,
    k: int,
) -> List[WorkloadEntry]:
    """
    Synthetically generate a workload per the paper's method:
    - For each query, include each column with probability p.
    - Pick a random row from each included column as the query vector.
    - Assign uniform probabilities, then normalize so they sum to 1.0.

    Seed is fixed so the workload is reproducible across runs.
    """
    rng = np.random.default_rng(seed=42)
    col_ids = list(col_data.keys())
    entries: List[WorkloadEntry] = []

    for _ in range(num_queries):
        # decide which columns this query targets
        included = [c for c in col_ids if rng.random() < p]
        if len(included) == 0:
            # every query must target at least one column
            included = [int(rng.choice(col_ids))]

        vid = frozenset(included)
        dim = sum(col_data[c].shape[1] for c in included)

        # pick a random row from each targeted column as the query vector
        vectors = {
            c: col_data[c][rng.integers(0, col_data[c].shape[0])].copy()
            for c in included
        }

        query = Query(vid=vid, vectors=vectors, k=k, dim=dim)
        entries.append(WorkloadEntry(query=query, probability=1.0))  # normalized below

    # normalize so all probabilities sum to 1.0
    n = len(entries)
    for entry in entries:
        entry.probability = 1.0 / n

    return entries


def load_workload(
    workload_path: str,
    col_data: Dict[int, np.ndarray],
) -> List[WorkloadEntry]:
    """
    Load workload from disk if it already exists (for reproducibility across runs).
    If not found, generate it and save it.
    """
    if os.path.exists(workload_path):
        print(f"Loading workload from {workload_path} ...")
        with open(workload_path, "rb") as f:
            return pickle.load(f)

    print("Generating workload ...")
    workload = generate_workload(col_data, config.NUM_QUERIES, config.WORKLOAD_P, config.K)

    os.makedirs(os.path.dirname(workload_path), exist_ok=True)
    with open(workload_path, "wb") as f:
        pickle.dump(workload, f)
    print(f"Workload saved to {workload_path}")

    return workload
