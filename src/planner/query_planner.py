"""
Query Planner: Algorithms 1 & 2 for optimal ek assignment.

Given a query, configuration (set of indexes), estimators, and ground truth,
find the cheapest ek assignment that meets the recall threshold.

Algorithm 1: Exhaustive search when |X| <= 3
Algorithm 2: Dynamic programming when |X| > 3
"""

import itertools
import math
import time
from typing import Dict, Set, Tuple, List
import numpy as np
import hnswlib

import config
from src.data.models import Index, Query, QueryPlan
from src.estimators.cost_estimator import CostEstimator
from src.estimators.recall_estimator import RecallEstimator

_ACTIVE_RETRIEVAL_CACHE: Dict[Tuple[frozenset, int], Set[int]] | None = None


def _build_sample_index(query: Query, index: Index, sample_data: Dict[int, np.ndarray]) -> hnswlib.Index:
    """
    Build a temporary HNSW index from the sample data for this specific index definition.
    Used to simulate what items would be retrieved without building real full-scale indexes.
    """
    col_ids = sorted(index.vid)
    col_vecs = [sample_data[col_id] for col_id in col_ids]
    combined_vecs = np.concatenate(col_vecs, axis=1).astype(np.float32)
    
    idx = hnswlib.Index(space=config.DISTANCE, dim=combined_vecs.shape[1])
    idx.init_index(
        max_elements=combined_vecs.shape[0],
        ef_construction=200,
        M=config.HNSW_MAX_DEGREE
    )
    idx.add_items(combined_vecs)
    idx.set_ef(200)
    
    return idx


def _retrieve_items_from_index(query: Query, index: Index, small_index: hnswlib.Index, ek: int) -> Set[int]:
    """Retrieve item IDs from the index with given ek value."""
    if ek == 0:
        return set()
    
    col_ids = sorted(index.vid)
    query_vec = np.concatenate([query.vectors[col_id] for col_id in col_ids]).astype(np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    try:
        labels, _ = small_index.knn_query(query_vec, k=min(ek, len(small_index.get_ids_list())))
        return set(labels[0])
    except:
        return set()


def _find_gt_ranks(query: Query, index: Index, small_index: hnswlib.Index, gt: Set[int], sample_size: int) -> Dict[int, int]:
    """Find the rank of each ground truth item in the index."""
    col_ids = sorted(index.vid)
    query_vec = np.concatenate([query.vectors[col_id] for col_id in col_ids]).astype(np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    try:
        labels, _ = small_index.knn_query(query_vec, k=min(sample_size, len(small_index.get_ids_list())))
        sorted_items = labels[0]
    except:
        sorted_items = np.array([])
    
    rank_map = {}
    for rank, item_id in enumerate(sorted_items):
        if item_id in gt:
            rank_map[item_id] = rank
    
    return rank_map


def _compute_relevant_ek_values(
    index: Index,
    query: Query,
    small_index: hnswlib.Index,
    gt: Set[int],
    sample_size: int
) -> List[int]:
    """Compute meaningful ek values for this (query, index) pair."""
    if not gt:
        return [0]
    
    gt_ranks = _find_gt_ranks(query, index, small_index, gt, sample_size)
    relevant = [0] + [rank + 1 for rank in gt_ranks.values()]
    
    return sorted(set(relevant))


def _compute_plan_cost_and_recall(
    query: Query,
    ek_map: Dict[Index, int],
    small_indexes: Dict[Index, hnswlib.Index],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    gt: Set[int]
) -> Tuple[float, float]:
    """Compute cost and recall of a specific ek assignment."""
    if not ek_map:
        return 0.0, 0.0
    
    if not gt:
        return 0.0, 1.0
    
    cost = 0.0
    retrieved_union = set()
    
    for index, ek in ek_map.items():
        if ek > 0:
            num_dist = cost_estimator.estimate_num_dist(index, ek)
            index_cost = index.dim * num_dist
            cost += index_cost
            
            if index in small_indexes:
                cache_key = (index.vid, ek)
                if _ACTIVE_RETRIEVAL_CACHE is not None and cache_key in _ACTIVE_RETRIEVAL_CACHE:
                    retrieved_items = _ACTIVE_RETRIEVAL_CACHE[cache_key]
                else:
                    retrieved_items = _retrieve_items_from_index(query, index, small_indexes[index], ek)
                    if _ACTIVE_RETRIEVAL_CACHE is not None:
                        _ACTIVE_RETRIEVAL_CACHE[cache_key] = retrieved_items
                retrieved_union.update(retrieved_items)
    
    total_ek = sum(ek for ek in ek_map.values() if ek > 0)
    rerank_cost = query.dim * total_ek
    cost += rerank_cost
    
    recall = len(retrieved_union & gt) / len(gt) if gt else 1.0
    
    return cost, recall


def _algorithm_1(
    query: Query,
    indexes: List[Index],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    small_indexes: Dict[Index, hnswlib.Index],
    gt: Set[int],
    theta_recall: float,
    sample_size: int
) -> QueryPlan:
    """Algorithm 1: Exhaustive search for |X| <= 3."""
    start_time = time.perf_counter()
    if not indexes:
        return QueryPlan(query=query, ek_map={}, cost=0.0, recall=0.0)
    
    relevant_ek_per_index = {}
    for index in indexes:
        relevant_ek_per_index[index] = _compute_relevant_ek_values(
            index, query, small_indexes[index], gt, sample_size
        )
    ek_counts = [len(relevant_ek_per_index[index]) for index in indexes]
    estimated_combinations = math.prod(ek_counts)
    
    best_plan = None
    best_cost = float('inf')
    
    if len(indexes) == 1:
        index = indexes[0]
        for ek in relevant_ek_per_index[index]:
            ek_map = {index: ek}
            cost, recall = _compute_plan_cost_and_recall(
                query, ek_map, small_indexes, cost_estimator, recall_estimator, gt
            )
            if recall >= theta_recall - 1e-6:
                if cost < best_cost:
                    best_cost = cost
                    best_plan = QueryPlan(query=query, ek_map=ek_map, cost=cost, recall=recall)
    else:
        indexes_except_last = indexes[:-1]
        last_index = indexes[-1]
        
        ek_combinations = list(itertools.product(*[
            relevant_ek_per_index[idx] for idx in indexes_except_last
        ]))
        
        for ek_tuple in ek_combinations:
            partial_ek_map = {indexes_except_last[i]: ek_tuple[i] for i in range(len(indexes_except_last))}
            
            best_last_ek = None
            best_with_last = (float('inf'), 0.0)
            
            for last_ek in sorted(relevant_ek_per_index[last_index]):
                full_ek_map = {**partial_ek_map, last_index: last_ek}
                cost, recall = _compute_plan_cost_and_recall(
                    query, full_ek_map, small_indexes, cost_estimator, recall_estimator, gt
                )
                
                if recall >= theta_recall - 1e-6 and cost < best_with_last[0]:
                    best_with_last = (cost, recall)
                    best_last_ek = last_ek
            
            if best_last_ek is not None and best_with_last[0] < best_cost:
                best_cost = best_with_last[0]
                full_ek_map = {**partial_ek_map, last_index: best_last_ek}
                best_plan = QueryPlan(
                    query=query,
                    ek_map=full_ek_map,
                    cost=best_with_last[0],
                    recall=best_with_last[1]
                )
    
    if best_plan is None:
        ek_map = {index: query.k for index in indexes}
        cost, recall = _compute_plan_cost_and_recall(
            query, ek_map, small_indexes, cost_estimator, recall_estimator, gt
        )
        best_plan = QueryPlan(query=query, ek_map=ek_map, cost=cost, recall=recall)

    elapsed = time.perf_counter() - start_time
    print(
        f"[Algorithm1] ek_counts={ek_counts}, "
        f"est_combinations={estimated_combinations}, "
        f"elapsed={elapsed:.3f}s"
    )
    
    return best_plan


def _algorithm_2(
    query: Query,
    indexes: List[Index],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    small_indexes: Dict[Index, hnswlib.Index],
    gt: Set[int],
    theta_recall: float,
    sample_size: int
) -> QueryPlan:
    """Algorithm 2: Dynamic programming for |X| > 3."""
    if not indexes:
        return QueryPlan(query=query, ek_map={}, cost=0.0, recall=0.0)
    
    k_prime = min(config.K_PRIME, max(1, len(gt)))
    rng = np.random.default_rng(seed=42)
    if len(gt) > 0:
        gt_sample = set(rng.choice(list(gt), size=k_prime, replace=False))
    else:
        gt_sample = set()
    
    all_subsets = []
    for i in range(1 << k_prime):
        subset = frozenset(
            gt_item for j, gt_item in enumerate(sorted(gt_sample))
            if (i >> j) & 1
        )
        all_subsets.append(subset)
    
    DP = [{} for _ in range(len(indexes) + 1)]
    DP[0][frozenset()] = (0.0, {})
    
    for i, index in enumerate(indexes):
        for cover_so_far, (cost_so_far, ek_map_so_far) in DP[i].items():
            for subset_to_cover in all_subsets:
                if not subset_to_cover or not gt_sample:
                    ek_needed = 0
                else:
                    gt_ranks = _find_gt_ranks(query, index, small_indexes[index], subset_to_cover, sample_size)
                    ek_needed = max(gt_ranks.values()) + 1 if gt_ranks else 0
                
                if ek_needed == 0:
                    index_cost = 0.0
                else:
                    num_dist = cost_estimator.estimate_num_dist(index, ek_needed)
                    index_cost = index.dim * num_dist
                
                new_cover = cover_so_far | subset_to_cover
                new_cost = cost_so_far + index_cost
                new_ek_map = {**ek_map_so_far, index: ek_needed}
                
                if new_cover not in DP[i + 1] or new_cost < DP[i + 1][new_cover][0]:
                    DP[i + 1][new_cover] = (new_cost, new_ek_map)
    
    best_plan = None
    best_cost = float('inf')
    
    for cover, (cost, ek_map) in DP[len(indexes)].items():
        if gt_sample:
            cover_recall_estimate = len(cover & gt_sample) / len(gt_sample)
        else:
            cover_recall_estimate = 1.0
        
        if cover_recall_estimate >= theta_recall - 1e-6:
            if cost < best_cost:
                best_cost = cost
                total_ek = sum(ek for ek in ek_map.values() if ek > 0)
                rerank_cost = query.dim * total_ek
                final_cost = cost + rerank_cost
                
                _, actual_recall = _compute_plan_cost_and_recall(
                    query, ek_map, small_indexes, cost_estimator, recall_estimator, gt
                )
                best_plan = QueryPlan(
                    query=query,
                    ek_map=ek_map,
                    cost=final_cost,
                    recall=actual_recall
                )
    
    if best_plan is None:
        ek_map = {index: query.k for index in indexes}
        cost, recall = _compute_plan_cost_and_recall(
            query, ek_map, small_indexes, cost_estimator, recall_estimator, gt
        )
        best_plan = QueryPlan(query=query, ek_map=ek_map, cost=cost, recall=recall)
    
    return best_plan


def plan_query(
    query: Query,
    configuration: List[Index],
    cost_estimator: CostEstimator,
    recall_estimator: RecallEstimator,
    sample_data: Dict[int, np.ndarray],
    gt: Set[int],
    theta_recall: float = None
) -> QueryPlan:
    """
    Main Query Planner function.
    
    Given a query and configuration (set of indexes), find the cheapest ek assignment
    that meets the recall threshold.
    
    Args:
        query: Query object with vid, vectors, k, dim
        configuration: List of Index objects to consider
        cost_estimator: CostEstimator object with estimate_num_dist method
        recall_estimator: RecallEstimator object with estimate_recall method
        sample_data: Dict mapping col_id to np.ndarray of sample database vectors
        gt: Set of ground truth item IDs for this query
        theta_recall: Minimum recall threshold (default from config)
    
    Returns:
        QueryPlan with optimal ek_map, cost, and recall
    """
    if theta_recall is None:
        db_size = len(sample_data[next(iter(sample_data.keys()))])
        if db_size >= 100000:
            theta_recall = config.THETA_RECALL_LARGE
        else:
            theta_recall = config.THETA_RECALL_SMALL
    
    relevant_indexes = [idx for idx in configuration if idx.vid.issubset(query.vid)]
    
    if not relevant_indexes:
        return QueryPlan(query=query, ek_map={}, cost=0.0, recall=0.0)
    
    small_indexes = {}
    for index in relevant_indexes:
        small_indexes[index] = _build_sample_index(query, index, sample_data)
    
    sample_size = len(sample_data[next(iter(sample_data.keys()))])
    
    global _ACTIVE_RETRIEVAL_CACHE
    previous_cache = _ACTIVE_RETRIEVAL_CACHE
    _ACTIVE_RETRIEVAL_CACHE = {}
    try:
        if len(relevant_indexes) <= 3:
            return _algorithm_1(
                query, relevant_indexes, cost_estimator, recall_estimator,
                small_indexes, gt, theta_recall, sample_size
            )
        else:
            return _algorithm_2(
                query, relevant_indexes, cost_estimator, recall_estimator,
                small_indexes, gt, theta_recall, sample_size
            )
    finally:
        _ACTIVE_RETRIEVAL_CACHE = previous_cache
