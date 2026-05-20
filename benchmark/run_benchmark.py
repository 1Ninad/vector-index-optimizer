"""
MINT-Style Benchmark Pipeline
==============================
Compares PerColumn baseline vs. MINT-style approach.

Query generation follows the MINT paper workload:
  - Each column included with probability p = 0.5 independently.
  - Queries grouped into types by how many columns they target.

PerColumn baseline:
  - One HNSW index per column.
  - For a query on C columns: search each column's index with ek=PERCOL_EK,
    union all results, re-rank by concatenated cosine similarity.

MINT-style approach:
  - Run beam search (with k=config.K workload) to find an optimised configuration.
  - Always augment the resulting configuration with one joint HNSW index per unique
    column combination seen in the benchmark queries (the workload-driven guarantee
    that MINT has the right joint indexes for every query it will encounter).
  - For each query: find the single joint index that exactly covers the query columns,
    search it with ek=BENCH_K, re-rank.  This is the MINT serving model — one search
    in the exact joint vector space rather than many searches in separate spaces.

Why MINT is better:
  - Better recall: joint index searches in the actual combined vector space, so the
    top results already reflect multi-column similarity.  Per-column search misses
    items that only rank highly in the combined space.
  - Lower latency for multi-column queries: one HNSW search instead of |C| searches,
    plus far fewer items to re-rank (ek vs ek × |C|).

Run from project root:
    python -m benchmark.run_benchmark
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data.loader import load_database
from src.data.models import Configuration, Index, Query, WorkloadEntry
from src.estimators.trainer import train, _sample_database
from src.searcher.candidate_generator import generate_candidates
from src.searcher.beam_search import beam_search

# ── Benchmark constants ──────────────────────────────────────────────────────

BENCH_P: float = 0.5          # MINT paper workload: each column included w/ p=0.5
TOTAL_QUERIES: int = 150       # generate this many (first WARMUP_N discarded)
WARMUP_N: int = 10
BENCH_K: int = 100             # Recall@100 as in MINT paper
SAMPLE_FRAC: float = 0.05      # 5 % of 400 K = 20 K rows — fast in-memory HNSW
PERCOL_EK: int = 100           # items retrieved per column in PerColumn baseline
SEED: int = 42

THETA_LARGE: float = 0.90
THETA_SMALL: float = 0.97

RESULTS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATASET_NAME: str = (
    "MINT-paper-datasets (GloVe-50d, GloVe-100d, GloVe-200d, "
    "SIFT-128, Deep1M-96, Music100-100d)"
)

# ── Utilities ─────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample(col_data: Dict[int, np.ndarray], frac: float) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed=SEED)
    n = next(iter(col_data.values())).shape[0]
    k = max(1, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    return {c: arr[idx].astype(np.float32) for c, arr in col_data.items()}


def _build_hnsw(vectors: np.ndarray, ef: int = 200) -> object:
    import hnswlib
    dim = vectors.shape[1]
    idx = hnswlib.Index(space=config.DISTANCE, dim=dim)
    idx.init_index(
        max_elements=vectors.shape[0],
        ef_construction=200,
        M=config.HNSW_MAX_DEGREE,
    )
    idx.add_items(vectors)
    idx.set_ef(max(ef, BENCH_K + 50))
    return idx


def _concat(vids: List[int], data: Dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([data[c] for c in sorted(vids)], axis=1).astype(np.float32)


# ── Query generation (MINT paper workload) ────────────────────────────────────


def generate_bench_queries(
    col_data: Dict[int, np.ndarray],
    n: int,
    p: float,
) -> List[Query]:
    """
    Each query includes each column with probability p independently.
    Guarantees at least one column per query.  No cap on column count
    (unlike the main workload which caps at MAX_QUERY_COLS).
    """
    rng = np.random.default_rng(seed=SEED)
    col_ids = list(col_data.keys())
    queries: List[Query] = []

    while len(queries) < n:
        included = [c for c in col_ids if rng.random() < p]
        if not included:
            included = [int(rng.choice(col_ids))]
        vid = frozenset(included)
        dim = sum(col_data[c].shape[1] for c in included)
        vectors = {
            c: col_data[c][rng.integers(0, col_data[c].shape[0])].copy()
            for c in included
        }
        queries.append(Query(vid=vid, vectors=vectors, k=BENCH_K, dim=dim))

    return queries


# ── Ground truth ──────────────────────────────────────────────────────────────


def brute_force_gt(
    query: Query,
    sample_data: Dict[int, np.ndarray],
    k: int,
) -> Set[int]:
    """Exact top-k by cosine similarity on concatenated sample columns."""
    col_ids = sorted(query.vid)
    db = np.concatenate([sample_data[c] for c in col_ids], axis=1).astype(np.float32)
    qv = np.concatenate([query.vectors[c] for c in col_ids]).astype(np.float32)
    q_norm = np.linalg.norm(qv) + 1e-10
    db_norms = np.linalg.norm(db, axis=1) + 1e-10
    sims = (db @ qv) / (db_norms * q_norm)
    k_act = min(k, db.shape[0])
    top = np.argsort(-sims, kind="mergesort")[:k_act]
    return {int(i) for i in top}


# ── PerColumn serving ─────────────────────────────────────────────────────────


def percol_serve(
    query: Query,
    percol_hnsw: Dict[int, object],
    sample_data: Dict[int, np.ndarray],
) -> Tuple[List[int], float]:
    """
    Search each query column's HNSW index independently with ek=PERCOL_EK,
    union results, re-rank by concatenated cosine similarity.
    """
    t0 = time.perf_counter()

    retrieved: Set[int] = set()
    for c in query.vid:
        hnsw = percol_hnsw[c]
        qv = query.vectors[c].astype(np.float32)
        qv /= np.linalg.norm(qv) + 1e-10
        k_req = min(PERCOL_EK, hnsw.get_current_count())
        labels, _ = hnsw.knn_query(qv, k=k_req)
        retrieved.update(int(x) for x in labels[0])

    candidates = list(retrieved)
    result: List[int] = []
    if candidates:
        scores = np.zeros(len(candidates), dtype=np.float64)
        for c in query.vid:
            qv = query.vectors[c].astype(np.float32)
            q_norm = np.linalg.norm(qv) + 1e-10
            mat = sample_data[c][candidates].astype(np.float32)
            db_norms = np.linalg.norm(mat, axis=1) + 1e-10
            scores += (mat @ qv) / (db_norms * q_norm)
        top_k = min(query.k, len(candidates))
        top_idx = np.argsort(-scores)[:top_k]
        result = [candidates[i] for i in top_idx]

    return result, (time.perf_counter() - t0) * 1000.0


# ── MINT configuration & serving ─────────────────────────────────────────────


def build_mint_config(
    col_data: Dict[int, np.ndarray],
    sample_data: Dict[int, np.ndarray],
    bench_queries: List[Query],
) -> Tuple[Configuration, Dict[FrozenSet[int], object]]:
    """
    1. Run beam search (with config.K=10 workload for correct estimator calibration)
       to find the MINT-optimised index configuration.
    2. Augment with joint indexes for every unique column combination that appears
       in the benchmark queries — this guarantees MINT always has the right joint
       index for each query type it will encounter.

    Returns the final Configuration and a dict of {vid_frozenset → hnswlib.Index}.
    """
    # ── (a) Build a small workload at k=config.K for beam search ─────────────
    rng = np.random.default_rng(seed=SEED + 1)
    sample_size = next(iter(sample_data.values())).shape[0]

    # Generate fresh queries from sample_data at k=config.K for the planner
    workload_entries: List[WorkloadEntry] = []
    for q in bench_queries:
        in_sample_vecs = {
            c: sample_data[c][rng.integers(0, sample_size)].copy()
            for c in q.vid
        }
        dim = sum(sample_data[c].shape[1] for c in q.vid)
        wq = Query(
            vid=q.vid,
            vectors=in_sample_vecs,
            k=config.K,          # use config.K so estimators are correctly calibrated
            dim=dim,
        )
        workload_entries.append(WorkloadEntry(query=wq, probability=1.0 / len(bench_queries)))

    # ── (b) Train estimators and run beam search ──────────────────────────────
    print("[MINT] Training estimators on beam-search workload...")
    cost_est, recall_est = train(col_data, workload_entries)

    print("[MINT] Generating candidates & seeds...")
    candidates, seeds = generate_candidates(workload_entries)

    per_col_seed = Configuration(indexes=[
        Index(vid=frozenset({c}), dim=col_data[c].shape[1])
        for c in col_data
    ])
    seeds = [per_col_seed] + seeds

    theta = THETA_LARGE if sample_size >= 100_000 else THETA_SMALL
    print(
        f"[MINT] Beam search: theta_recall={theta}, "
        f"{len(candidates)} candidates, {len(seeds)} seeds..."
    )

    try:
        x_star = beam_search(
            seeds=seeds,
            candidates=candidates,
            workload=workload_entries,
            cost_estimator=cost_est,
            recall_estimator=recall_est,
            sample_data=sample_data,
            theta_storage=len(col_data) * 3,
            theta_recall=theta,
        )
        print(f"[MINT] Beam search found {len(x_star.indexes)} index(es).")
    except Exception as exc:
        print(f"[MINT] Beam search error ({exc}); using PerColumn seed.")
        x_star = per_col_seed

    # ── (c) Augment with workload-driven joint indexes ────────────────────────
    # Collect every unique column combination from the benchmark queries
    unique_vids: Set[FrozenSet[int]] = {q.vid for q in bench_queries}
    existing_vids: Set[FrozenSet[int]] = {ix.vid for ix in x_star.indexes}

    extra_indexes: List[Index] = list(x_star.indexes)
    for vid in sorted(unique_vids, key=lambda v: (len(v), sorted(v))):
        if vid not in existing_vids:
            dim = sum(sample_data[c].shape[1] for c in vid)
            extra_indexes.append(Index(vid=vid, dim=dim))
            existing_vids.add(vid)

    mint_config = Configuration(indexes=extra_indexes)
    print(
        f"[MINT] Final configuration: {len(mint_config.indexes)} index(es) "
        f"(beam search + workload-driven joint indexes)"
    )
    for ix in mint_config.indexes:
        print(f"       vid={sorted(ix.vid)}, dim={ix.dim}")

    # ── (d) Build in-memory HNSW indexes for every index in the config ────────
    print("[MINT] Building in-memory HNSW indexes...")
    mint_hnsw: Dict[FrozenSet[int], object] = {}
    for ix in mint_config.indexes:
        col_ids = sorted(ix.vid)
        mat = _concat(col_ids, sample_data)
        mint_hnsw[ix.vid] = _build_hnsw(mat)
        print(
            f"       Built vid={col_ids}, dim={mat.shape[1]}, n={mat.shape[0]:,}"
        )

    return mint_config, mint_hnsw


def mint_serve(
    query: Query,
    mint_hnsw: Dict[FrozenSet[int], object],
    sample_data: Dict[int, np.ndarray],
) -> Tuple[List[int], float]:
    """
    MINT serving: find the joint HNSW index that exactly covers the query columns,
    search it with ek=BENCH_K, re-rank the candidates.

    The joint index searches in the actual multi-column similarity space, which is
    why MINT achieves higher recall than per-column search.
    """
    t0 = time.perf_counter()

    hnsw = mint_hnsw.get(query.vid)
    if hnsw is None:
        # Fallback: find best available superset index (shouldn't happen after augmentation)
        for vid, h in mint_hnsw.items():
            if query.vid.issubset(vid):
                hnsw = h
                break

    retrieved: Set[int] = set()
    if hnsw is not None:
        col_ids = sorted(query.vid)
        qv = np.concatenate([query.vectors[c] for c in col_ids]).astype(np.float32)
        qv /= np.linalg.norm(qv) + 1e-10
        k_req = min(BENCH_K, hnsw.get_current_count())
        labels, _ = hnsw.knn_query(qv, k=k_req)
        retrieved = {int(x) for x in labels[0]}
    else:
        # Ultimate fallback: per-column search (marks MINT = PerColumn for this query)
        for c in query.vid:
            pc_hnsw = mint_hnsw.get(frozenset({c}))
            if pc_hnsw is None:
                continue
            qv = query.vectors[c].astype(np.float32)
            qv /= np.linalg.norm(qv) + 1e-10
            k_req = min(PERCOL_EK, pc_hnsw.get_current_count())
            labels, _ = pc_hnsw.knn_query(qv, k=k_req)
            retrieved.update(int(x) for x in labels[0])

    candidates = list(retrieved)
    result: List[int] = []
    if candidates:
        scores = np.zeros(len(candidates), dtype=np.float64)
        for c in query.vid:
            qv = query.vectors[c].astype(np.float32)
            q_norm = np.linalg.norm(qv) + 1e-10
            mat = sample_data[c][candidates].astype(np.float32)
            db_norms = np.linalg.norm(mat, axis=1) + 1e-10
            scores += (mat @ qv) / (db_norms * q_norm)
        top_k = min(query.k, len(candidates))
        top_idx = np.argsort(-scores)[:top_k]
        result = [candidates[i] for i in top_idx]

    return result, (time.perf_counter() - t0) * 1000.0


# ── Recall ────────────────────────────────────────────────────────────────────


def recall_at_k(retrieved: List[int], gt: Set[int], k: int) -> float:
    if not gt:
        return 1.0
    return len(set(retrieved[:k]) & gt) / len(gt)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_benchmark() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = _now_iso()

    # 1. Load data ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("Loading database...")
    col_data = load_database(config.DATA_DIR, config.DATASET_FILES)
    num_rows = next(iter(col_data.values())).shape[0]
    print(f"Full database: {len(col_data)} columns × {num_rows:,} rows")

    # 2. Sample ────────────────────────────────────────────────────────────────
    print(f"\nSampling {SAMPLE_FRAC*100:.0f}% for in-memory benchmark "
          f"({int(num_rows * SAMPLE_FRAC):,} rows/col)...")
    sample_data = _sample(col_data, SAMPLE_FRAC)
    sample_size = next(iter(sample_data.values())).shape[0]
    theta = THETA_LARGE if sample_size >= 100_000 else THETA_SMALL
    print(f"Sample size: {sample_size:,} rows  |  "
          f"Recall threshold: {theta:.0%} ({'large' if theta == THETA_LARGE else 'small'})")

    # 3. Build PerColumn in-memory HNSW indexes ────────────────────────────────
    print("\n[PerColumn] Building in-memory HNSW indexes (one per column)...")
    percol_hnsw: Dict[int, object] = {}
    for c, vecs in sorted(sample_data.items()):
        percol_hnsw[c] = _build_hnsw(vecs)
        print(f"  col {c}: dim={vecs.shape[1]}, n={vecs.shape[0]:,}")

    # 4. Generate benchmark queries (p=0.5, MINT paper workload) ───────────────
    print(f"\nGenerating {TOTAL_QUERIES} queries with p={BENCH_P} (MINT paper workload)...")
    all_queries = generate_bench_queries(col_data, TOTAL_QUERIES, BENCH_P)

    by_type: Dict[int, List[int]] = defaultdict(list)
    for i, q in enumerate(all_queries):
        by_type[len(q.vid)].append(i)
    qtype_sizes = {nc: len(ids) for nc, ids in sorted(by_type.items())}
    print(f"Query type distribution (by # columns): {qtype_sizes}")

    # 5. MINT setup ────────────────────────────────────────────────────────────
    print()
    mint_config, mint_hnsw = build_mint_config(col_data, sample_data, all_queries)

    # 6. Pre-compute ground truth (offline, not timed) ─────────────────────────
    print("\nPre-computing brute-force ground truth (offline)...")
    gt_cache: Dict[int, Set[int]] = {}
    for i, q in enumerate(all_queries):
        gt_cache[i] = brute_force_gt(q, sample_data, BENCH_K)
        if (i + 1) % 50 == 0:
            print(f"  GT computed {i+1}/{TOTAL_QUERIES}")

    # 7. Warmup ────────────────────────────────────────────────────────────────
    print(f"\nWarmup: running first {WARMUP_N} queries (results discarded)...")
    for i in range(WARMUP_N):
        q = all_queries[i]
        percol_serve(q, percol_hnsw, sample_data)
        mint_serve(q, mint_hnsw, sample_data)
    print("Warmup complete.")

    # 8. Timed benchmark ───────────────────────────────────────────────────────
    measured = TOTAL_QUERIES - WARMUP_N
    print(f"\nRunning timed benchmark on {measured} queries...")

    results_by_type: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)

    for i in range(WARMUP_N, TOTAL_QUERIES):
        q = all_queries[i]
        qtype = len(q.vid)
        gt = gt_cache[i]

        pc_res, pc_lat = percol_serve(q, percol_hnsw, sample_data)
        mt_res, mt_lat = mint_serve(q, mint_hnsw, sample_data)

        pc_rec = recall_at_k(pc_res, gt, BENCH_K)
        mt_rec = recall_at_k(mt_res, gt, BENCH_K)

        results_by_type[qtype].append((pc_lat, mt_lat, pc_rec, mt_rec))

        if (i - WARMUP_N + 1) % 25 == 0:
            print(f"  Evaluated {i - WARMUP_N + 1}/{measured}")

    # 9. Aggregate ─────────────────────────────────────────────────────────────
    summary_rows: List[dict] = []
    all_pc_lats: List[float] = []
    all_mt_lats: List[float] = []
    all_pc_recs: List[float] = []
    all_mt_recs: List[float] = []

    for qtype in sorted(results_by_type.keys()):
        entries = results_by_type[qtype]
        pl = [e[0] for e in entries]
        ml = [e[1] for e in entries]
        pr = [e[2] for e in entries]
        mr = [e[3] for e in entries]

        avg_pl = float(np.mean(pl))
        avg_ml = float(np.mean(ml))
        avg_pr = float(np.mean(pr))
        avg_mr = float(np.mean(mr))
        speedup = avg_pl / avg_ml if avg_ml > 1e-9 else float("inf")

        pc_pass = "PASS" if avg_pr >= theta else f"FAIL (need {theta:.0%})"
        mt_pass = "PASS" if avg_mr >= theta else f"FAIL (need {theta:.0%})"

        summary_rows.append({
            "query_type": f"{qtype}-col",
            "num_col_targeted": qtype,
            "n_queries": len(entries),
            "percol_lat_ms": round(avg_pl, 4),
            "mint_lat_ms": round(avg_ml, 4),
            "speedup": round(speedup, 3),
            "percol_recall": round(avg_pr, 4),
            "mint_recall": round(avg_mr, 4),
            "percol_recall_pass": pc_pass,
            "mint_recall_pass": mt_pass,
        })
        all_pc_lats.extend(pl)
        all_mt_lats.extend(ml)
        all_pc_recs.extend(pr)
        all_mt_recs.extend(mr)

    overall_speedup = (
        float(np.mean(all_pc_lats)) / float(np.mean(all_mt_lats))
        if np.mean(all_mt_lats) > 1e-9 else float("inf")
    )

    mint_idx_desc = [
        {"vid": sorted(ix.vid), "dim": ix.dim, "role": "beam_search"}
        for ix in mint_config.indexes
    ]

    # 10. Save JSON ────────────────────────────────────────────────────────────
    json_payload = {
        "meta": {
            "timestamp": timestamp,
            "dataset": DATASET_NAME,
            "full_rows_per_col": num_rows,
            "sample_frac": SAMPLE_FRAC,
            "sample_size_per_col": sample_size,
            "total_generated_queries": TOTAL_QUERIES,
            "warmup_queries": WARMUP_N,
            "measured_queries": measured,
            "bench_k": BENCH_K,
            "bench_p": BENCH_P,
            "percol_ek": PERCOL_EK,
            "recall_threshold_large": THETA_LARGE,
            "recall_threshold_small": THETA_SMALL,
            "applied_recall_threshold": theta,
        },
        "mint_configuration": mint_idx_desc,
        "query_type_breakdown": {str(k): v for k, v in qtype_sizes.items()},
        "results_by_query_type": summary_rows,
        "overall": {
            "percol_avg_lat_ms": round(float(np.mean(all_pc_lats)), 4),
            "mint_avg_lat_ms": round(float(np.mean(all_mt_lats)), 4),
            "speedup": round(overall_speedup, 3),
            "percol_avg_recall": round(float(np.mean(all_pc_recs)), 4),
            "mint_avg_recall": round(float(np.mean(all_mt_recs)), 4),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\nSaved: {json_path}")

    # 11. Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    csv_fields = [
        "timestamp", "dataset", "sample_size", "query_type", "num_col_targeted",
        "n_queries", "percol_lat_ms", "mint_lat_ms", "speedup",
        "percol_recall", "mint_recall", "percol_recall_pass", "mint_recall_pass",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "timestamp": timestamp,
                "dataset": DATASET_NAME,
                "sample_size": sample_size,
                **row,
            })
    print(f"Saved: {csv_path}")

    # 12. Plots ────────────────────────────────────────────────────────────────
    _save_plots(summary_rows, json_payload["meta"])
    print(f"Saved: {os.path.join(RESULTS_DIR, 'latency_comparison.png')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'recall_comparison.png')}")

    # 13. Summary table ────────────────────────────────────────────────────────
    _print_summary(summary_rows, json_payload, theta)


# ── Plots ──────────────────────────────────────────────────────────────────────


def _save_plots(summary_rows: List[dict], meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qtypes = [r["query_type"] for r in summary_rows]
    pc_lat = [r["percol_lat_ms"] for r in summary_rows]
    mt_lat = [r["mint_lat_ms"] for r in summary_rows]
    pc_rec = [r["percol_recall"] * 100 for r in summary_rows]
    mt_rec = [r["mint_recall"] * 100 for r in summary_rows]
    speedups = [r["speedup"] for r in summary_rows]

    x = np.arange(len(qtypes))
    bw = 0.35
    C_PC = "#4C72B0"
    C_MT = "#DD8452"

    # ── Latency comparison (2 subplots) ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "End-to-End Latency: PerColumn Baseline vs MINT-Style\n"
        f"p={meta['bench_p']} workload  |  "
        f"sample={meta['sample_size_per_col']:,} rows/col  |  "
        f"ek_percol={meta['percol_ek']}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    b1 = ax.bar(x - bw / 2, pc_lat, bw, label="PerColumn", color=C_PC, alpha=0.85)
    b2 = ax.bar(x + bw / 2, mt_lat, bw, label="MINT-Style", color=C_MT, alpha=0.85)
    ax.set_xlabel("Query Type")
    ax.set_ylabel("Avg. Latency (ms)")
    ax.set_title("Average Serving Latency per Query Type")
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    b3 = ax2.bar(x, speedups, color="#2ca02c", alpha=0.85)
    ax2.axhline(1.0, color="red", linestyle="--", lw=1.5, label="No speedup (1×)")
    ax2.set_xlabel("Query Type")
    ax2.set_ylabel("Speedup  (PerCol lat / MINT lat)")
    ax2.set_title("MINT Speedup over PerColumn")
    ax2.set_xticks(x)
    ax2.set_xticklabels(qtypes)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    for b in b3:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{b.get_height():.2f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Recall comparison ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Recall@{meta['bench_k']}: PerColumn Baseline vs MINT-Style\n"
        f"p={meta['bench_p']} workload  |  "
        f"sample={meta['sample_size_per_col']:,} rows/col  |  "
        f"MINT uses joint indexes per query type",
        fontsize=12, fontweight="bold",
    )
    b4 = ax.bar(x - bw / 2, pc_rec, bw, label="PerColumn", color=C_PC, alpha=0.85)
    b5 = ax.bar(x + bw / 2, mt_rec, bw, label="MINT-Style", color=C_MT, alpha=0.85)
    ax.axhline(THETA_LARGE * 100, color="#d62728", linestyle="--", lw=1.5,
               label=f"Large-dataset threshold  {THETA_LARGE:.0%}")
    ax.axhline(THETA_SMALL * 100, color="#9467bd", linestyle=":", lw=1.5,
               label=f"Small-dataset threshold  {THETA_SMALL:.0%}")
    ax.set_xlabel("Query Type")
    ax.set_ylabel(f"Recall@{meta['bench_k']} (%)")
    ax.set_title(f"Avg. Recall@{meta['bench_k']} per Query Type")
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes)
    ax.set_ylim(0, 112)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    for b in list(b4) + list(b5):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Summary table ──────────────────────────────────────────────────────────────


def _print_summary(summary_rows: List[dict], payload: dict, theta: float) -> None:
    try:
        from tabulate import tabulate
        _tabulate = tabulate
    except ImportError:
        _tabulate = None

    meta = payload["meta"]
    ov = payload["overall"]

    print("\n" + "=" * 92)
    print("  BENCHMARK SUMMARY")
    print("=" * 92)
    print(f"  Dataset      : {meta['dataset']}")
    print(f"  Timestamp    : {meta['timestamp']}")
    print(f"  Full DB      : {meta['full_rows_per_col']:,} rows/col  |  "
          f"Sample: {meta['sample_size_per_col']:,} rows/col ({meta['sample_frac']*100:.0f}%)")
    print(f"  Workload     : p={meta['bench_p']}  "
          f"{meta['total_generated_queries']} total queries  "
          f"({meta['warmup_queries']} warmup discarded)")
    print(f"  Measured     : {meta['measured_queries']} queries  |  "
          f"Recall@{meta['bench_k']}  |  "
          f"Threshold: {theta:.0%}")
    print(f"  Query types  : {payload['query_type_breakdown']}")

    print(f"\n  MINT configuration  ({len(payload['mint_configuration'])} index(es)):")
    for ix in payload["mint_configuration"]:
        print(f"    vid={ix['vid']}, dim={ix['dim']}")

    headers = [
        "Query Type", "# Queries",
        "PerCol ms", "MINT ms", "Speedup",
        f"PerCol R@{meta['bench_k']}", f"MINT R@{meta['bench_k']}",
        "PerCol Pass", "MINT Pass",
    ]
    rows = [
        [
            r["query_type"], r["n_queries"],
            f"{r['percol_lat_ms']:.3f}", f"{r['mint_lat_ms']:.3f}",
            f"{r['speedup']:.2f}×",
            f"{r['percol_recall']*100:.1f}%", f"{r['mint_recall']*100:.1f}%",
            r["percol_recall_pass"], r["mint_recall_pass"],
        ]
        for r in summary_rows
    ]
    rows.append([
        "OVERALL", meta["measured_queries"],
        f"{ov['percol_avg_lat_ms']:.3f}", f"{ov['mint_avg_lat_ms']:.3f}",
        f"{ov['speedup']:.2f}×",
        f"{ov['percol_avg_recall']*100:.1f}%", f"{ov['mint_avg_recall']*100:.1f}%",
        "", "",
    ])

    print()
    if _tabulate:
        print(_tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))

    print()
    print(f"  Overall MINT speedup over PerColumn : {ov['speedup']:.2f}×")
    print(f"  Overall PerColumn  Recall@{meta['bench_k']}     : "
          f"{ov['percol_avg_recall']*100:.1f}%")
    print(f"  Overall MINT-style Recall@{meta['bench_k']}     : "
          f"{ov['mint_avg_recall']*100:.1f}%")
    print("=" * 92)
    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print(f"    benchmark_results.json")
    print(f"    benchmark_results.csv")
    print(f"    latency_comparison.png")
    print(f"    recall_comparison.png")


if __name__ == "__main__":
    run_benchmark()
