import os

# Candidate generation: max column-count difference between a candidate index and the query.
# e.g. di=2 means a 5-column query can be served by indexes with 3, 4, or 5 columns.
DI: int = 2

# Seed configuration limit: max number of indexes per seed configuration.
SE: int = 2

# Beam search stopping threshold: stop when cost improvement drops below this fraction.
IM: float = 0.05

# Algorithm 2 (DP) sample size. Keeps 2^K_PRIME = 32 DP states — fast and accurate enough.
K_PRIME: int = 5

# Beam width: how many configurations to keep per beam search iteration.
BEAM_WIDTH: int = 5

# Fraction of database rows used for estimator training (1% sample).
SAMPLE_FRAC: float = 0.01

# Minimum ek value used during training. The linear numDist model only holds for ek >= 100.
MIN_EK_TRAIN: int = 100

# HNSW graph degree. Higher = better recall but more storage and slower build.
HNSW_MAX_DEGREE: int = 16

# Distance metric for all ANN indexes.
DISTANCE: str = "cosine"

# Top-k: how many results each query returns. recall@100 is the paper's evaluation metric.
K: int = 100

# Recall thresholds: 90% for large datasets (>= 100K rows), 97% for small ones.
THETA_RECALL_LARGE: float = 0.90
THETA_RECALL_SMALL: float = 0.97

# Workload generation: each column is included in a query with this probability.
WORKLOAD_P: float = 0.5

# Number of synthetic workload queries to generate.
NUM_QUERIES: int = 1000

# Root data directory (absolute path relative to this file's location).
DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Where built HNSW index files are saved after the configuration search completes.
INDEX_DIR: str = os.path.join(DATA_DIR, "indexes")

# Workload is saved here after first generation so it's reused on subsequent runs.
WORKLOAD_PATH: str = os.path.join(DATA_DIR, "workload.pkl")

# Maps each column ID to its file, binary format, and vector dimension.
# To add Yandex as column 6, append: 6: {"path": "yandex_text_to_image_1M.fbin", "format": "fbin", "dim": 200}
DATASET_FILES: dict = {
    0: {"path": "glove/glove.6B.50d.txt",       "format": "txt",  "dim": 50},
    1: {"path": "glove/glove.6B.100d.txt",       "format": "txt",  "dim": 100},
    2: {"path": "glove/glove.6B.200d.txt",       "format": "txt",  "dim": 200},
    3: {"path": "sift-128-euclidean.hdf5",       "format": "hdf5", "dim": 128},
    4: {"path": "deep1M_base.fbin",              "format": "fbin", "dim": 96},
    5: {"path": "database_music100.bin",         "format": "bin",  "dim": 100},
}
