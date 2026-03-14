from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


# Index: an ANN index built on one or more columns.
# frozen=True makes it hashable, so it can be used as a dict key (needed by QueryPlan.ek_map
# and the beam search caches in src/searcher).
@dataclass(frozen=True)
class Index:
    vid: frozenset  # frozenset[int] — which column IDs this index covers
    dim: int        # total vector dimension = sum of dims of all covered columns


# Query: one search request from a user.
@dataclass
class Query:
    vid: frozenset                    # frozenset[int] — which column IDs this query targets
    vectors: Dict[int, np.ndarray]   # actual query vector for each column in vid
    k: int                           # how many top results to return
    dim: int                         # sum of dims of all targeted columns


# WorkloadEntry: one (query, probability) pair in the workload.
# probability is normalized — all entries in the workload sum to 1.0.
@dataclass
class WorkloadEntry:
    query: Query
    probability: float


# Configuration: the set of indexes to build and use.
@dataclass
class Configuration:
    indexes: List[Index] = field(default_factory=list)


# QueryPlan: for one query, the cheapest assignment of ek values to indexes.
# ek_map maps each Index to how many items to retrieve from it (ek=0 means skip that index).
@dataclass
class QueryPlan:
    query: Query
    ek_map: Dict[Index, int]
    cost: float    # estimated total latency cost
    recall: float  # estimated recall fraction (0.0 to 1.0)
