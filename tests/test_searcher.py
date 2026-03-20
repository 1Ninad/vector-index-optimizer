import numpy as np

import config
from src.data.models import Configuration, Index, Query, WorkloadEntry
from src.estimators.cost_estimator import CostEstimator
from src.estimators.recall_estimator import RecallEstimator
from src.index_builder.builder import build_indexes, query_index
from src.searcher.beam_search import (
    beam_search,
    prune_configuration,
    relevant_indexes_for_query,
)
from src.searcher.candidate_generator import generate_candidates
from src.storage.estimator import estimate_storage


def _tiny_sample_data() -> dict[int, np.ndarray]:
    return {
        0: np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        1: np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }


def test_estimate_storage_small_configuration() -> None:
    config_obj = Configuration(
        indexes=[
            Index(vid=frozenset({0}), dim=2),
            Index(vid=frozenset({0, 1}), dim=4),
        ]
    )

    count_ok, valid_ok = estimate_storage(config_obj, theta_storage=2)
    count_bad, valid_bad = estimate_storage(config_obj, theta_storage=1)

    assert count_ok == 2
    assert valid_ok is True
    assert count_bad == 2
    assert valid_bad is False


def test_generate_candidates_tiny_workload(monkeypatch) -> None:
    monkeypatch.setattr(config, "DI", 1)
    monkeypatch.setattr(config, "SE", 2)

    query = Query(
        vid=frozenset({0, 1}),
        vectors={
            0: np.array([1.0, 2.0], dtype=np.float32),
            1: np.array([3.0, 4.0], dtype=np.float32),
        },
        k=2,
        dim=4,
    )
    workload = [WorkloadEntry(query=query, probability=1.0)]

    candidates, seeds = generate_candidates(workload)

    candidate_vids = {index.vid for index in candidates}
    assert candidate_vids == {frozenset({0}), frozenset({1}), frozenset({0, 1})}
    assert all(len(seed.indexes) <= config.SE for seed in seeds)
    assert len(seeds) > 0


def test_relevant_indexes_for_query_filters_correctly(monkeypatch) -> None:
    monkeypatch.setattr(config, "DI", 1)

    query = Query(
        vid=frozenset({0, 1, 2}),
        vectors={
            0: np.array([1.0], dtype=np.float32),
            1: np.array([1.0], dtype=np.float32),
            2: np.array([1.0], dtype=np.float32),
        },
        k=1,
        dim=3,
    )
    indexes = [
        Index(vid=frozenset({0}), dim=1),
        Index(vid=frozenset({0, 1}), dim=2),
        Index(vid=frozenset({0, 1, 2}), dim=3),
        Index(vid=frozenset({3}), dim=1),
    ]

    relevant = relevant_indexes_for_query(query, indexes)

    assert [idx.vid for idx in relevant] == [frozenset({0, 1}), frozenset({0, 1, 2})]


def test_prune_configuration_removes_globally_irrelevant(monkeypatch) -> None:
    monkeypatch.setattr(config, "DI", 0)

    workload = [
        WorkloadEntry(
            query=Query(
                vid=frozenset({0, 1}),
                vectors={
                    0: np.array([1.0], dtype=np.float32),
                    1: np.array([1.0], dtype=np.float32),
                },
                k=1,
                dim=2,
            ),
            probability=0.5,
        ),
        WorkloadEntry(
            query=Query(
                vid=frozenset({1, 2}),
                vectors={
                    1: np.array([1.0], dtype=np.float32),
                    2: np.array([1.0], dtype=np.float32),
                },
                k=1,
                dim=2,
            ),
            probability=0.5,
        ),
    ]
    configuration = Configuration(
        indexes=[
            Index(vid=frozenset({0, 1}), dim=2),
            Index(vid=frozenset({1}), dim=1),
            Index(vid=frozenset({3, 4}), dim=2),
        ]
    )

    pruned = prune_configuration(configuration, workload)

    assert pruned.indexes == [Index(vid=frozenset({0, 1}), dim=2)]


def test_beam_search_returns_configuration(monkeypatch) -> None:
    monkeypatch.setattr(config, "DI", 1)
    monkeypatch.setattr(config, "BEAM_WIDTH", 2)
    monkeypatch.setattr(config, "IM", 0.0)
    monkeypatch.setattr(config, "RECALL_METRIC_K", 2, raising=False)

    sample_data = _tiny_sample_data()
    query = Query(
        vid=frozenset({0, 1}),
        vectors={0: sample_data[0][0], 1: sample_data[1][0]},
        k=2,
        dim=4,
    )
    workload = [WorkloadEntry(query=query, probability=1.0)]

    idx0 = Index(vid=frozenset({0}), dim=2)
    idx1 = Index(vid=frozenset({1}), dim=2)
    idx01 = Index(vid=frozenset({0, 1}), dim=4)

    seeds = [Configuration(indexes=[idx0]), Configuration(indexes=[idx01])]
    candidates = [idx0, idx1, idx01]

    cost_estimator = CostEstimator(a_per_col={0: 1.0, 1: 1.0}, b_per_col={0: 0.0, 1: 0.0})
    recall_estimator = RecallEstimator(a_per_col={0: 1.0, 1: 1.0}, b_per_col={0: 0.0, 1: 0.0})

    result = beam_search(
        seeds=seeds,
        candidates=candidates,
        workload=workload,
        cost_estimator=cost_estimator,
        recall_estimator=recall_estimator,
        sample_data=sample_data,
        theta_storage=3,
        theta_recall=0.0,
    )

    assert isinstance(result, Configuration)
    assert len(result.indexes) >= 1


def test_build_indexes_creates_files_in_temp_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(config, "RECALL_METRIC_K", 3, raising=False)

    col_data = _tiny_sample_data()
    configuration = Configuration(indexes=[Index(vid=frozenset({0, 1}), dim=4)])

    index_files = build_indexes(col_data, configuration)

    assert len(index_files) == 1
    assert (tmp_path / "0_1.bin").exists()
    assert (tmp_path / "0_1.bin.meta.json").exists()


def test_query_index_returns_arrays_of_requested_length(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(config, "RECALL_METRIC_K", 3, raising=False)

    col_data = _tiny_sample_data()
    index = Index(vid=frozenset({0, 1}), dim=4)
    index_file = build_indexes(col_data, Configuration(indexes=[index]))[0]

    query_vector = np.concatenate([col_data[0][0], col_data[1][0]]).astype(np.float32)
    item_ids, scores = query_index(index_file, query_vector, ek=3)

    assert item_ids.shape == (3,)
    assert scores.shape == (3,)
