from pathlib import Path

OUTPUT_DIR = Path("output")
DATASETS_DIR = Path("datasets")
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
ALGORITHMS_DIR = OUTPUT_DIR / "algorithms"

BASE_REPO_URL = "https://github.com/TDT4900-Masteroppgave/mosso-mags-dm.git"

PARAM_CONFIG = {
    "c": {"description": "sample number", "default": 120, "bounds": (10, 240), "step": 10, "type": int},
    "e": {"description": "escape", "default": 3, "bounds": (1, 5), "step": 1, "type": int},
    "interval": {"description": "interval", "default": 10000, "type": int},
    "b": {"description": "top candidates", "default": 5, "bounds": (1, 10), "step": 1, "type": int},
    "h": {"description": "hashes", "default": 4, "bounds": (5, 50), "step": 5, "type": int},
    "thr_start": {"description": "start threshold", "default": 1.0, "type": float},
    "thr_end": {"description": "end threshold", "default": 0.0, "bounds": (0.0, 1.0), "step": 0.05, "type": float},
    "cap": {"description": "cap of coarse cluster buckets", "default": 100, "bounds": (100, 500), "step": 50,
            "type": int},
    "T": {"description": "iterations over partitions", "default": 1, "bounds": (10, 50), "step": 10, "type": int},
    "p": {"description": "number of threads", "default": 1, "bounds": (1, 40), "type": int},
}

STREAM_TYPES = ["IO", "FD"]  # Insertion-Only, Fully-Dynamic

ALGORITHMS = {
    "local": {
        "target_dir": ".",
        "type": "mosso",
        "template": ["e", "c", "interval"]
    },
    "kdd20-mosso": {
        "repo": "https://github.com/jihoonko/kdd20-mosso.git",
        "branch": "master",
        "params": {"c": 120, "e": 3},
        "type": "mosso",
        "template": ["e", "c", "interval"],
        "binary_file": "kdd20-mosso.jar"
    },
    "strat_2": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/similarity_measure",
        "type": "mosso",
        "template": ["e", "c", "interval", "h"],
        "binary_file": "mosso.jar",
    },
    "strat_1_2": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/similarity_measure_top_b",
        "type": "mosso",
        "template": ["e", "c", "interval", "b", "h"],
        "binary_file": "mosso.jar",
    },
    "strat_2_thr": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/similarity_measure_thr",
        "type": "mosso",
        "template": ["e", "c", "interval", "h", "thr_end"],
        "binary_file": "mosso.jar",
    },
    "cap": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/cap",
        "type": "mosso",
        "template": ["e", "c", "interval", "cap"],
        "binary_file": "mosso-strat_2_cap.jar",
    },
    "ds": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/divide_strategy",
        "type": "mosso",
        "template": ["e", "c", "interval"],
        "binary_file": "mosso.jar",
    },
    "ds_thr": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/divide_strategy_thr",
        "type": "mosso",
        "template": ["e", "c", "interval", "thr_start", "thr_end", "T"],
        "binary_file": "mosso.jar",
    },
    "ds_sm_thr": {
        "repo": BASE_REPO_URL,
        "branch": "mags_strat/divide_strategy_similarity_measure_thr",
        "type": "mosso",
        "template": ["e", "c", "interval", "h", "thr_start", "thr_end", "T"],
        "binary_file": "mosso.jar",
    },
    "mags": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": [],
        "binary_file": "mags",
    },
    "para_mags": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": ["p"],
        "binary_file": "pmags",
    },
    "mags_dm": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": [],
        "binary_file": "mags_dm",
    },
    "para_mags_dm": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": ["p"],
        "binary_file": "pmags_dm",
    },
}

DATASETS = {
    "PR": {
        "url": "http://konect.cc/files/download.tsv.reactome.tar.bz2",
        "filename": "out.reactome",
        "meta": {}
    },
    "CA": {
        "url": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
        "filename": "as-caida20071105.txt",
        "meta": {
            "nodes": 26475,
            "edges": 53381,
            "size": "0.66 MB",
        },
    },
    "FB": {
        "url": "https://nrvis.com/download/data/dynamic/fb-wosn-friends.zip",
        "filename": "fb-wosn-friends.edges",
        "meta": {}
    },
    "EN": {
        "url": "https://nrvis.com/download/data/dynamic/ia-enron-email-dynamic.zip",
        "filename": "email-enron.edges",
        "meta": {
            "nodes": 86977,
            "edges": 297456,
            "size": "0.75 MB",
        },
    },
    "BK": {
        "url": "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz",
        "filename": "Brightkite_edges.txt",
        "meta": {
            "nodes": 58228,
            "edges": 214078,
            "size": "2.59 MB",
            "avg_degree": 7.35,
        },
    },
    "EA": {
        "url": "https://snap.stanford.edu/data/email-EuAll.txt.gz",
        "filename": "Email-EuAll.txt",
        "meta": {
            "nodes": 265009,
            "edges": 364481,
            "size": "4.52 MB",
        },
    },
    "SL": {
        "url": "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz",
        "filename": "Slashdot0902.txt",
        "meta": {
            "nodes": 82168,
            "edges": 504230,
            "size": "6.18 MB",
        },
    },
    "DB": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "filename": "com-dblp.ungraph.txt",
        "meta": {
            "nodes": 317080,
            "edges": 1049866,
            "size": "15.29 MB",
        },
    },
    "AM": {
        "url": "https://snap.stanford.edu/data/amazon0601.txt.gz",
        "filename": "amazon0601.txt",
        "meta": {
            "nodes": 403394,
            "edges": 2443408,
            "size": "35.09 MB",
        },
    },
    "EU": {
        "url": "https://sparse.tamu.edu/MM/LAW/eu-2005.tar.gz",
        "filename": "eu-2005.mtx",
        "meta": {
            "nodes": 862664,
            "edges": 16138468,
            "type": "Hyperlink"
        },
    },
    "YT": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "filename": "com-youtube.ungraph.txt",
        "meta": {
            "nodes": 1134890,
            "edges": 2987624,
            "size": "42.63 MB",
        },
    },
    "SK": {
        "url": "https://snap.stanford.edu/data/as-skitter.txt.gz",
        "filename": "as-skitter.txt",
        "meta": {
            "nodes": 1696415,
            "edges": 11095298,
            "size": "163.36 MB",
        },
    },
    "HW": {
        "url": "https://sparse.tamu.edu/MM/LAW/hollywood-2009.tar.gz",
        "filename": "hollywood-2009.mtx",
        "meta": {
            "nodes": 1139905,
            "edges": 113891327,
            "type": "Collaboration"
        },
    },
    "UK": {
        "url": "https://sparse.tamu.edu/MM/LAW/uk-2002.tar.gz",
        "filename": "uk-2002.mtx",
        "meta": {
            "nodes": 18483186,
            "edges": 261787258,
            "type": "Hyperlink"
        },
    },
    "LJ": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "filename": "com-lj.ungraph.txt",
        "meta": {
            "nodes": 3997962,
            "edges": 34681189,
            "size": "544.47 MB",
        },
    },
}

DATASET_GROUP = {
    "small": [
        "CA", "EN", "BK", "EA", "SL", "DB"
    ],
    "large": [
        "AM", "YT", "SK", "LJ"
    ],
    "insertion": [
        "PR", "EN", "FB", "EU", "HW", "UK",
    ],
    "dynamic": [
        "DB", "YT", "SK",  "LJ"
    ]
}
