"""Beam search over candidate index configurations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

import config
from src.data.models import Configuration, Index, Query, QueryPlan, WorkloadEntry
from src.estimators.cost_estimator import CostEstimator
from src.estimators.recall_estimator import RecallEstimator
from src.planner.query_planner import plan_query
from src.storage.estimator import estimate_storage


def _index_sort_key(index: Index) -> tuple[tuple[int, ...], int]:
    """Return a deterministic sort key for Index objects."""
    return tuple(sorted(index.vid)), index.dim


def _configuration_key(configuration: Configuration) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return a deterministic key for configuration deduplication."""
    return tuple(_index_sort_key(index) for index in sorted(configuration.indexes, key=_index_sort_key))


def _normalize_configuration(configuration: Configuration) -> Configuration:
    """Sort and deduplicate indexes inside a configuration."""
    unique_indexes = sorted(set(configuration.indexes), key=_index_sort_key)
    return Configuration(indexes=unique_indexes)


def relevant_indexes_for_query(query: Query, indexes: Iterable[Index]) -> list[Index]:
    """Return indexes relevant to a query under subset and DI constraints."""
    min_size = max(0, len(query.vid) - config.DI)
    relevant = [
        index
        for index in indexes
        if index.vid.issubset(query.vid) and len(index.vid) >= min_size
    ]
    return sorted(relevant, key=_index_sort_key)


def prune_configuration(configuration: Configuration, workload: list[WorkloadEntry]) -> Configuration:
    """Remove indexes that are irrelevant for every workload query."""
    kept: list[Index] = []
    for index in sorted(set(configuration.indexes), key=_index_sort_key):
        if any(
            index.vid.issubset(entry.query.vid)
            and len(index.vid) >= max(0, len(entry.query.vid) - config.DI)
            for entry in workload
        ):
            kept.append(index)
    return Configuration(indexes=kept)


def compute_ground_truth(query: Query, sample_data: dict[int, np.ndarray], top_k: int) -> set[int]:
    """Compute exact top-k neighbors by cosine similarity on concatenated sample vectors."""
    col_ids = sorted(query.vid)
    db_matrix = np.concatenate(
        [sample_data[col_id] for col_id in col_ids],
        axis=1,
    ).astype(np.float32)
    query_vector = np.concatenate(
        [query.vectors[col_id] for col_id in col_ids],
    ).astype(np.float32)

    q_norm = np.linalg.norm(query_vector) + 1e-10
    db_norms = np.linalg.norm(db_matrix, axis=1) + 1e-10
    similarities = (db_matrix @ query_vector) / (db_norms * q_norm)

    k = min(int(top_k), db_matrix.shape[0])
    top_ids = np.argsort(-similarities, kind="mergesort")[:k]
    return {int(item_id) for item_id in top_ids}


def _default_theta_recall(sample_data: dict[int, np.ndarray]) -> float:
    """Pick default theta_recall from sample size."""
    sample_size = next(iter(sample_data.values())).shape[0]
    if sample_size >= 100000:
        return config.THETA_RECALL_LARGE
    return config.THETA_RECALL_SMALL


def evaluate_configuration(
    configuration: Configuration,
    workload: list[WorkloadEntry],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    sample_data: dict[int, np.ndarray],
    theta_storage: int,
    theta_recall: float,
    plan_cache: dict[tuple[int, frozenset[frozenset[int]]], QueryPlan],
    gt_cache: dict[int, set[int]],
) -> tuple[float, bool]:
    """Evaluate weighted cost and validity of one configuration."""
    normalized = _normalize_configuration(configuration)
    _, storage_valid = estimate_storage(normalized, theta_storage)

    print("----")
    print(f"Evaluating config with {len(normalized.indexes)} indexes")
    print(f"Config vids: {[sorted(idx.vid) for idx in normalized.indexes]}")
    print(f"Storage valid: {storage_valid}")

    if not storage_valid:
        return float("inf"), False

    total_weighted_cost = 0.0

    for query_id, entry in enumerate(workload):
        query = entry.query
        relevant = relevant_indexes_for_query(query, normalized.indexes)

        if not relevant:
            return float("inf"), False

        cache_key = (query_id, frozenset(index.vid for index in relevant))
        plan = plan_cache.get(cache_key)
        if plan is None:
            gt = gt_cache.get(query_id)
            if gt is None:
                gt = compute_ground_truth(query, sample_data, config.K)
                gt_cache[query_id] = gt

            plan = plan_query(
                query,
                relevant,
                cost_estimator,
                recall_estimator,
                sample_data,
                gt,
                theta_recall,
            )
            plan_cache[cache_key] = plan

        print(f"Plan recall: {plan.recall}, required: {theta_recall}")
        print(f"Plan cost: {plan.cost}")

        if plan.recall + 1e-9 < theta_recall:
            return float("inf"), False

        total_weighted_cost += float(entry.probability) * float(plan.cost)

    print(f"✅ Valid config, total weighted cost: {total_weighted_cost}")
    return total_weighted_cost, True


def beam_search(
    seeds: list[Configuration],
    candidates: list[Index],
    workload: list[WorkloadEntry],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    sample_data: dict[int, np.ndarray],
    theta_storage: Optional[int] = None,
    theta_recall: Optional[float] = None,
) -> Configuration:
    """Run deterministic beam search and return the best valid configuration."""
    if theta_storage is None:
        theta_storage = len(sample_data)
    if theta_recall is None:
        theta_recall = _default_theta_recall(sample_data)

    normalized_candidates = sorted(set(candidates), key=_index_sort_key)
    normalized_seeds = sorted(
        {_configuration_key(_normalize_configuration(seed)): _normalize_configuration(seed) for seed in seeds}.values(),
        key=_configuration_key,
    )

    plan_cache: dict[tuple[int, frozenset[frozenset]], QueryPlan] = {}
    gt_cache: dict[int, set[int]] = {}

    scored_seeds: list[tuple[float, bool, tuple[tuple[tuple[int, ...], int], ...], Configuration]] = []
    best_valid: Configuration | None = None
    best_valid_cost = float("inf")
    best_seen: Configuration | None = None
    best_seen_rank: tuple[int, float, tuple[tuple[tuple[int, ...], int], ...]] | None = None

    for seed in normalized_seeds:
        pruned = _normalize_configuration(prune_configuration(seed, workload))
        if len(pruned.indexes) > theta_storage:
            continue
        cost, valid = evaluate_configuration(
            pruned,
            workload,
            cost_estimator,
            recall_estimator,
            sample_data,
            theta_storage,
            theta_recall,
            plan_cache,
            gt_cache,
        )
        key = _configuration_key(pruned)
        scored_seeds.append((cost, valid, key, pruned))
        rank = (0 if np.isfinite(cost) else 1, cost, key)
        if best_seen is None or (best_seen_rank is not None and rank < best_seen_rank) or best_seen_rank is None:
            best_seen = pruned
            best_seen_rank = rank
        if valid:
            if cost < best_valid_cost:
                best_valid_cost = cost
                best_valid = pruned

    scored_seeds.sort(
        key=lambda item: (
            0 if np.isfinite(item[0]) else 1,
            item[0],
            item[2],
        )
    )
    beam = scored_seeds[: config.BEAM_WIDTH]
    if not beam:
        if best_valid is None:
            raise ValueError("Beam search could not find any valid configuration.")
        return best_valid

    while True:
        current_best_cost = beam[0][0]
        expansions: dict[
            tuple[tuple[tuple[int, ...], int], ...],
            tuple[float, bool, tuple[tuple[tuple[int, ...], int], ...], Configuration],
        ] = {}

        for _, _, _, conf in beam:
            existing = set(conf.indexes)
            for candidate in normalized_candidates:
                if candidate in existing:
                    continue

                expanded = _normalize_configuration(
                    Configuration(indexes=[*conf.indexes, candidate])
                )
                pruned = _normalize_configuration(prune_configuration(expanded, workload))
                if len(pruned.indexes) > theta_storage:
                    continue
                key = _configuration_key(pruned)
                if key in expansions:
                    continue

                cost, valid = evaluate_configuration(
                    pruned,
                    workload,
                    cost_estimator,
                    recall_estimator,
                    sample_data,
                    theta_storage,
                    theta_recall,
                    plan_cache,
                    gt_cache,
                )
                scored = (cost, valid, key, pruned)
                expansions[key] = scored
                rank = (0 if np.isfinite(cost) else 1, cost, key)
                if best_seen is None or (best_seen_rank is not None and rank < best_seen_rank) or best_seen_rank is None:
                    best_seen = pruned
                    best_seen_rank = rank
                if valid and cost < best_valid_cost:
                    best_valid_cost = cost
                    best_valid = pruned

        next_beam = sorted(
            expansions.values(),
            key=lambda item: (
                0 if np.isfinite(item[0]) else 1,
                item[0],
                item[2],
            ),
        )[: config.BEAM_WIDTH]
        if not next_beam:
            break

        next_best_cost = next_beam[0][0]
        if current_best_cost <= 0:
            improvement = current_best_cost - next_best_cost
        else:
            improvement = (current_best_cost - next_best_cost) / current_best_cost

        beam = next_beam
        if improvement < config.IM:
            break

    print(f"Best valid config found: {best_valid}")
    print(f"Beam size at end: {len(beam)}")
    
    if best_valid is None:
        if best_seen is not None:
            print(
                "Warning: no configuration met recall threshold; "
                "returning best-scored fallback configuration."
            )
            return best_seen
        raise ValueError("Beam search could not find any valid configuration.")
    return best_valid
