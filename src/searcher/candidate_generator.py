"""Candidate and seed generation for configuration search."""

from __future__ import annotations

from itertools import combinations

import config
from src.data.models import Configuration, Index, Query, WorkloadEntry


def _index_sort_key(index: Index) -> tuple[tuple[int, ...], int]:
    """Return a deterministic sort key for Index objects."""
    return tuple(sorted(index.vid)), index.dim


def _configuration_key(configuration: Configuration) -> tuple[Index, ...]:
    """Return a hashable key for deduplicating configurations."""
    return tuple(sorted(configuration.indexes, key=_index_sort_key))


def _configuration_sort_key(
    configuration: Configuration,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return a deterministic primitive sort key for Configuration objects."""
    return tuple(_index_sort_key(index) for index in _configuration_key(configuration))


def _build_query_candidates(query: Query) -> list[Index]:
    """Generate deterministic candidate indexes for a single query."""
    query_columns = sorted(query.vid)
    min_size = max(1, len(query_columns) - config.DI)

    candidates: list[Index] = []
    for subset_size in range(min_size, len(query_columns) + 1):
        for subset in combinations(query_columns, subset_size):
            dim = sum(int(query.vectors[col_id].shape[0]) for col_id in subset)
            candidates.append(Index(vid=frozenset(subset), dim=dim))

    return candidates


def _build_query_seed_configurations(
    query_candidates: list[Index],
) -> list[Configuration]:
    """Generate deterministic seed configurations for one query."""
    sorted_candidates = sorted(query_candidates, key=_index_sort_key)
    max_size = min(config.SE, len(sorted_candidates))

    seed_configurations: list[Configuration] = []
    for subset_size in range(1, max_size + 1):
        for subset in combinations(sorted_candidates, subset_size):
            seed_configurations.append(Configuration(indexes=list(subset)))

    return seed_configurations


def generate_candidates(workload: list[WorkloadEntry]) -> tuple[list[Index], list[Configuration]]:
    """Generate deduplicated global candidates and seed configurations."""
    candidate_set: set[Index] = set()
    configuration_map: dict[tuple[Index, ...], Configuration] = {}

    for entry in workload:
        query_candidates = _build_query_candidates(entry.query)
        candidate_set.update(query_candidates)

        for configuration in _build_query_seed_configurations(query_candidates):
            configuration_map[_configuration_key(configuration)] = configuration

    candidates = sorted(candidate_set, key=_index_sort_key)
    configurations = sorted(configuration_map.values(), key=_configuration_sort_key)
    return candidates, configurations
