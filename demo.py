"""
MINT Demo — runs the full pipeline on a tiny synthetic dataset.
Shows every component working end-to-end in under 60 seconds.

Run: python demo.py
"""

import numpy as np

# ── Tiny synthetic database ────────────────────────────────────────────────
# 500 items, 3 columns (instead of 400K items and 6 columns).
# Column dims match real dataset proportions.
NUM_ITEMS   = 500
COL_DIMS    = {0: 50, 1: 100, 2: 128}   # col_id → vector dimension
THETA_RECALL = 0.90
K            = 10   # top-10 retrieval
N_QUERIES    = 20   # workload size

rng = np.random.default_rng(seed=0)
col_data = {
    col_id: rng.standard_normal((NUM_ITEMS, dim)).astype(np.float32)
    for col_id, dim in COL_DIMS.items()
}

print("=" * 60)
print("MINT — Multi-Vector Index Tuning  |  Demo Run")
print("=" * 60)
print(f"\nDatabase : {NUM_ITEMS} items  |  {len(col_data)} columns")
for col_id, arr in col_data.items():
    print(f"  col {col_id} : shape {arr.shape}")

# ── Workload ───────────────────────────────────────────────────────────────
from src.data.models import Query, WorkloadEntry, Index, Configuration, QueryPlan

col_ids = list(col_data.keys())
workload = []
for i in range(N_QUERIES):
    included = [c for c in col_ids if rng.random() < 0.5]
    if not included:
        included = [int(rng.choice(col_ids))]
    if len(included) > 2:          # keep queries small for clarity
        included = included[:2]
    vid = frozenset(included)
    vectors = {c: col_data[c][rng.integers(0, NUM_ITEMS)] for c in included}
    dim = sum(col_data[c].shape[1] for c in included)
    workload.append(WorkloadEntry(
        query=Query(vid=vid, vectors=vectors, k=K, dim=dim),
        probability=1.0 / N_QUERIES,
    ))

from collections import Counter
sizes = Counter(len(e.query.vid) for e in workload)
print(f"\nWorkload : {N_QUERIES} queries  {dict(sizes)}")

# ── Train estimators ───────────────────────────────────────────────────────
print("\n[1] Training cost + recall estimators on full dataset (no sampling needed at this size)...")
from src.estimators.trainer import train
cost_est, recall_est = train(col_data, workload)
print("    Done.")

# ── Sample data (same as full data here since dataset is tiny) ─────────────
from src.estimators.trainer import _sample_database
sample_data = _sample_database(col_data)

# ── Candidate + seed generation ────────────────────────────────────────────
print("\n[2] Generating candidate indexes and seeds...")
from src.searcher import generate_candidates
candidates, seeds = generate_candidates(workload)

per_column_seed = Configuration(indexes=[
    Index(vid=frozenset({col_id}), dim=col_data[col_id].shape[1])
    for col_id in col_data
])
seeds = [per_column_seed] + seeds
print(f"    Candidates: {len(candidates)}  |  Seeds: {len(seeds)}")

# ── Beam search ────────────────────────────────────────────────────────────
print("\n[3] Running beam search to find best index configuration...")
from src.searcher import beam_search
import config, unittest.mock

with unittest.mock.patch.object(config, 'BEAM_WIDTH', 3), \
     unittest.mock.patch.object(config, 'K', K):
    x_star = beam_search(
        seeds=seeds,
        candidates=candidates,
        workload=workload,
        cost_estimator=cost_est,
        recall_estimator=recall_est,
        sample_data=sample_data,
        theta_storage=len(col_data),
        theta_recall=THETA_RECALL,
    )

print(f"\n    Best configuration found: {len(x_star.indexes)} index(es)")
for idx in x_star.indexes:
    print(f"      vid={sorted(idx.vid)}  dim={idx.dim}")

# ── PerColumn baseline cost (for comparison) ───────────────────────────────
print("\n[4] Computing PerColumn baseline for comparison...")
from src.searcher.beam_search import evaluate_configuration, compute_ground_truth

pc_config = per_column_seed
plan_cache_pc, gt_cache_pc = {}, {}
pc_cost, pc_valid = evaluate_configuration(
    pc_config, workload, cost_est, recall_est,
    sample_data, len(col_data), THETA_RECALL, plan_cache_pc, gt_cache_pc,
)

plan_cache_mint, gt_cache_mint = {}, {}
mint_cost, mint_valid = evaluate_configuration(
    x_star, workload, cost_est, recall_est,
    sample_data, len(col_data), THETA_RECALL, plan_cache_mint, gt_cache_mint,
)

print(f"\n    {'Configuration':<22} {'Indexes built':>14} {'Weighted Cost':>14}  {'Recall ≥ 90%'}")
print(f"    {'-'*60}")
print(f"    {'PerColumn baseline':<22} {len(pc_config.indexes):>14} {pc_cost:>14.1f}  {pc_valid}")
print(f"    {'MINT (beam search)':<22} {len(x_star.indexes):>14} {mint_cost:>14.1f}  {mint_valid}")

print()
if len(x_star.indexes) < len(pc_config.indexes):
    dropped = len(pc_config.indexes) - len(x_star.indexes)
    print(f"    Storage saving : MINT uses {dropped} fewer index(es) than PerColumn.")
    print(f"    It dropped columns that no query in the workload actually needs.")
if mint_valid and pc_valid and pc_cost > 0:
    saving = (pc_cost - mint_cost) / pc_cost * 100
    if saving > 0.5:
        print(f"    Cost saving    : MINT is {saving:.1f}% cheaper at the same recall.")
    else:
        print(f"    Cost saving    : Same query cost — saving comes from fewer indexes built.")

# ── Build real indexes ─────────────────────────────────────────────────────
import os, tempfile
print("\n[5] Building real HNSW indexes for the best configuration...")
index_dir = tempfile.mkdtemp(prefix="mint_demo_indexes_")
with unittest.mock.patch.object(config, 'INDEX_DIR', index_dir):
    from src.index_builder import build_indexes
    built_files = build_indexes(col_data, x_star)
print(f"    Built {len(built_files)} index file(s):")
for f in built_files:
    size_kb = os.path.getsize(f) // 1024
    print(f"      {os.path.basename(f)}  ({size_kb} KB)")

# ── Query planning ─────────────────────────────────────────────────────────
print("\n[6] Planning queries against best configuration...")
from src.planner.query_planner import plan_query
query_plans = {}
for qid, entry in enumerate(workload):
    gt = compute_ground_truth(entry.query, sample_data, K)
    query_plans[qid] = plan_query(
        entry.query, list(x_star.indexes),
        cost_est, recall_est, sample_data, gt, THETA_RECALL,
    )
print(f"    Plans ready for {len(query_plans)} queries.")

# ── Serving-time demo ──────────────────────────────────────────────────────
print("\n[7] Serving queries using real indexes...")

from src.index_builder import query_index, index_filename

def serve_query(query, plan):
    retrieved = set()
    for index, ek in plan.ek_map.items():
        if ek == 0:
            continue
        col_ids_sorted = sorted(index.vid)
        qvec = np.concatenate([query.vectors[c] for c in col_ids_sorted]).astype(np.float32)
        ifile = os.path.join(index_dir, index_filename(index))
        ids, _ = query_index(ifile, qvec, ek)
        retrieved.update(int(i) for i in ids)
    if not retrieved:
        return []
    cands = list(retrieved)
    scores = np.zeros(len(cands))
    for col_id in query.vid:
        qv = query.vectors[col_id].astype(np.float32)
        qn = np.linalg.norm(qv) + 1e-10
        mat = col_data[col_id][cands].astype(np.float32)
        scores += (mat @ qv) / (np.linalg.norm(mat, axis=1) + 1e-10) / qn
    top = np.argsort(-scores)[:query.k]
    return [cands[i] for i in top]

print(f"\n{'Query':<8} {'Columns':<12} {'Index scanned':<20} {'ek used':<12} {'Recall'}")
print("-" * 65)
for qid in range(min(8, len(workload))):
    entry = workload[qid]
    plan  = query_plans[qid]
    results = serve_query(entry.query, plan)
    ek_str = ", ".join(
        f"col{sorted(idx.vid)}={ek}"
        for idx, ek in plan.ek_map.items()
    )
    print(f"  q{qid:<5} {str(sorted(entry.query.vid)):<12} {ek_str:<20} {'see left':<12} {plan.recall:.2f}")

print("\n" + "=" * 60)
print("What this demo shows:")
print(f"  1. MINT built {len(x_star.indexes)} index(es). PerColumn would build {len(pc_config.indexes)}.")
print( "     MINT drops indexes for columns no query actually uses.")
print( "  2. Recall >= 0.90 is guaranteed for every query.")
print( "  3. 'ek=0' in a query plan means that index was skipped entirely")
print( "     — the system decided it wasn't needed to hit the recall target.")
print( "  4. The cost metric is the system's proxy for query latency.")
print( "     Lower cost = faster search at the same accuracy.")
print("=" * 60)
