# Vector Index Optimizer
---

## Problem Statement
The project addresses the Multi-Vector Search Index Tuning problem.  
Given a database where each item is represented by multiple vectors and a query workload, the goal is to determine a set of indexes that minimizes query latency while satisfying constraints on recall and storage.

**What task will the project help complete?**
The system determines which vector indexes should be created and how they should be used to answer multi-vector queries efficiently.

**What value does the project provide?**
The system recommends an optimized index configuration and query plan that improves query performance.

---

## Features
**Query Planner**: Determines the best query execution plan for a query using a given set of indexes.  
It selects a subset of indexes and parameters that minimize cost while satisfying the recall constraint.

**Configuration Searcher**: Searches the space of possible index configurations and identifies the configuration that minimizes overall workload latency. 

**Cost Estimator**: Estimates query cost (a proxy for latency) based on the number of distance computations required during index scans and re-ranking. 

**Recall Estimator**: Estimates the recall achieved by a query plan relative to the ground truth results.

**Storage Estimator**: Estimates the storage required by an index configuration. 

**Index Recommendation**: Outputs a recommended configuration of indexes that satisfies recall and storage constraints while minimizing workload latency. 

---

**Inputs**
The system takes the following inputs:
- A database of items where each item is represented by multiple vectors  
- A query workload with associated query probabilities  
- A recall threshold constraint  
- A storage constraint for indexes

**Outputs**
The system produces:
- A recommended index configuration (set of indexes to build)
- Efficient query plans for executing the workload queries

---

## Setup (first time on a new machine)

**1. Clone the repo**
```bash
git clone https://github.com/1Ninad/vector-index-optimizer
cd vector-index-optimizer
```

**2. Python version**

Python 3.11 or higher required. Check with:
```bash
python --version
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download datasets**

Download the following files and place them exactly at these paths inside the project:

```
data/glove/glove.6B.50d.txt
data/glove/glove.6B.100d.txt
data/glove/glove.6B.200d.txt
data/sift-128-euclidean.hdf5
data/deep1M_base.fbin
data/database_music100.bin
```

Sources:
- GloVe: https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip — download `glove.6B.zip`, extract into `data/glove/`, delete glove.6B.300d.txt
- SIFT: http://ann-benchmarks.com/sift-128-euclidean.hdf5 — download
- Deep1M: run: 
curl -L "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin" \
  --range 0-383999999 \
  -o data/deep1M_base.fbin
- Database Music100: https://disk.yandex.com/d/F85at7Df685hGA


**5. Verify setup**
```bash
python -m pytest tests/ -v
```

All tests must pass. This also generates `data/workload.pkl` on first run (takes ~2 min).

**7. Run the pipeline**
```bash
python main.py
```