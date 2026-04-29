import os
import numpy as np

OUTPUT_DIR = "output"
VERSIONS_DIR = os.path.join(OUTPUT_DIR, "versions")
DATASETS_DIR = "datasets"

BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
RUNS_DIR = os.path.join(BENCHMARK_DIR, "runs")
SUMMARIZED_DIR = os.path.join(BENCHMARK_DIR, "summarized_graphs")
SWEEP_DIR = os.path.join(OUTPUT_DIR, "parameter_sweep")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

JAR_ORIGINAL = "mosso-original.jar"
JAR_HYBRID = "mosso-hybrid.jar"
ORIGINAL_REPO_URL = "https://github.com/jihoonko/kdd20-mosso"
BASE_REPO_URL = "https://github.com/TDT4900-Masteroppgave/mosso-mags-dm.git"

ALGORITHMS = {
    "local": {
        "template": ["escape", "samples", "interval", "thr"]
    },
    "kdd20-mosso": {
        "repo": "https://github.com/jihoonko/kdd20-mosso.git",
        "branch": "master",
        "params" : {"samples": 120, "escape": 3},
        "template": ["escape", "samples", "interval"]
    },
    "strat_1": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_1",
        "template": ["escape", "samples", "interval"]
    },
    "strat_2": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_2",
        "template": ["escape", "samples", "interval"]
    },
    "similarity_measure_thr": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/similarity_measure_thr",
        "template": ["escape", "samples", "interval", "thr"]
    },
    "strat_1_2": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_1_2",
        "template": ["escape", "samples", "interval", "b"]
    },
}

DATASETS = {
    "small": [
        ("https://snap.stanford.edu/data/as-caida20071105.txt.gz", "as-caida20071105.txt"),
        ("https://snap.stanford.edu/data/email-Enron.txt.gz", "Email-Enron.txt"),
        ("https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz", "Brightkite_edges.txt"),
        ("https://snap.stanford.edu/data/email-EuAll.txt.gz", "Email-EuAll.txt"),
        ("https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz", "Slashdot0902.txt"),
        ("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz", "com-dblp.ungraph.txt")
    ],
    "large": [
        ("https://snap.stanford.edu/data/amazon0601.txt.gz", "amazon0601.txt"),
        ("https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz", "com-youtube.ungraph.txt"),
        ("https://snap.stanford.edu/data/as-skitter.txt.gz", "as-skitter.txt"),
        ("https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz", "com-lj.ungraph.txt")
    ]
}

SWEEP_CONFIG = {
    "samples": {"values": [i for i in range(10, 240, 10)], "default": 120},
    "escape": {"values": [i for i in range(1, 9, 2)], "default": 3},
    "b": {"values": [i for i in range(1, 10, 2)], "default": 5},
    "thr": {"values": [i for i in np.arange(0.0, 0.4, 0.05)], "default": 0.0}
}