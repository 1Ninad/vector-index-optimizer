"""Build and query persisted ANN indexes."""

from __future__ import annotations

import json
import os
from typing import Any

import hnswlib
import numpy as np

import config
from src.data.models import Configuration, Index


def _index_sort_key(index: Index) -> tuple[tuple[int, ...], int]:
    """Return a deterministic sort key for Index objects."""
    return tuple(sorted(index.vid)), index.dim


def _metadata_path(index_file: str) -> str:
    """Return sidecar metadata path for an index file."""
    return f"{index_file}.meta.json"


def index_filename(index: Index) -> str:
    """Build a deterministic filename from sorted column IDs."""
    columns = sorted(index.vid)
    if not columns:
        raise ValueError("Index vid must be non-empty to create a filename.")
    return f"{'_'.join(str(col_id) for col_id in columns)}.bin"


def build_concat_matrix(col_data: dict[int, np.ndarray], vid: frozenset[int]) -> np.ndarray:
    """Concatenate column vectors in sorted column-id order."""
    columns = sorted(vid)
    if not columns:
        raise ValueError("vid must be non-empty.")

    matrices = [np.asarray(col_data[col_id], dtype=np.float32) for col_id in columns]
    num_rows = matrices[0].shape[0]
    if any(matrix.shape[0] != num_rows for matrix in matrices):
        raise ValueError("All columns must have the same number of rows.")

    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)


def build_indexes(col_data: dict[int, np.ndarray], configuration: Configuration) -> list[str]:
    """Build and persist HNSW indexes for the given configuration."""
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    built_files: list[str] = []

    for index in sorted(configuration.indexes, key=_index_sort_key):
        matrix = build_concat_matrix(col_data, index.vid)
        num_elements, dim = matrix.shape
        if dim != index.dim:
            raise ValueError(
                f"Index dim mismatch for {sorted(index.vid)}: expected {index.dim}, got {dim}"
            )

        hnsw_index = hnswlib.Index(space=config.DISTANCE, dim=dim)
        hnsw_index.init_index(
            max_elements=num_elements,
            ef_construction=200,
            M=config.HNSW_MAX_DEGREE,
        )
        hnsw_index.add_items(matrix)
        hnsw_index.set_ef(max(100, config.K))

        index_file = os.path.join(config.INDEX_DIR, index_filename(index))
        hnsw_index.save_index(index_file)

        metadata: dict[str, Any] = {
            "vid": sorted(index.vid),
            "dim": int(dim),
            "num_elements": int(num_elements),
        }
        with open(_metadata_path(index_file), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)

        built_files.append(index_file)

    return built_files


def query_index(index_file: str, query_vector: np.ndarray, ek: int) -> tuple[np.ndarray, np.ndarray]:
    """Load an index from disk and run a single knn query."""
    with open(_metadata_path(index_file), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    dim = int(metadata["dim"])
    num_elements = int(metadata["num_elements"])
    k = max(1, min(int(ek), num_elements))

    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dim:
        raise ValueError(f"Query vector dim {vector.shape[0]} does not match index dim {dim}.")

    hnsw_index = hnswlib.Index(space=config.DISTANCE, dim=dim)
    hnsw_index.load_index(index_file, max_elements=num_elements)
    hnsw_index.set_ef(max(int(ek), 100))

    labels, distances = hnsw_index.knn_query(vector, k=k)
    return labels[0], distances[0]
