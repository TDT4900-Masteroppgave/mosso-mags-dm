from pathlib import Path

OUTPUT_DIR = Path("output")
DATASETS_DIR = Path("datasets")
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
ALGORITHMS_DIR = Path("algorithms")

BASE_REPO_URL = "https://github.com/TDT4900-Masteroppgave/mosso-mags-dm.git"

PARAM_CONFIG = {
    "c": {"description": "sample number", "default": 120, "bounds": (10, 240), "step": 10, "type": int},
    "e": {"description": "escape", "default": 3, "bounds": (1, 5), "step": 1, "type": int},
    "interval": {"description": "interval", "default": 1000, "type": int},
    "b": {"description": "top candidates", "default": 5, "bounds": (1, 10), "step": 1, "type": int},
    "h": {"description": "hashes", "default": 4, "bounds": (5, 50), "step": 5, "type": int},
    "thr_start": {"description": "start threshold", "default": 1.0, "type": float},
    "thr_end": {"description": "end threshold", "default": 0.0, "bounds": (0.0, 1.0), "step": 0.05, "type": float},
    "cap": {"description": "cap of coarse cluster buckets", "default": 100, "bounds": (100, 500), "step": 50,
            "type": int},
    "T": {"description": "iterations over partitions", "default": 1, "bounds": (10, 50), "step": 10, "type": int},
    "p": {"description": "number of threads", "default": 1, "bounds": (1, 40), "type": int},
}

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
        "meta": {
            "type": "Protein Interaction"
        }
    },
    "CA": {
        "url": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
        "filename": "as-caida20071105.txt",
        "meta": {
            "type": "Autonomous System"
        },
    },
    "FB": {
        "url": "https://nrvis.com/download/data/dynamic/fb-wosn-friends.zip",
        "filename": "fb-wosn-friends.edges",
        "meta": {
            "type": "Social"
        }
    },
    "EN": {
        "url": "https://nrvis.com/download/data/dynamic/ia-enron-email-dynamic.zip",
        "filename": "email-enron.edges",
        "meta": {
            "type": "Email"
        },
    },
    "BK": {
        "url": "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz",
        "filename": "Brightkite_edges.txt",
        "meta": {
            "type": "Location-based Social"
        },
    },
    "EA": {
        "url": "https://snap.stanford.edu/data/email-EuAll.txt.gz",
        "filename": "Email-EuAll.txt",
        "meta": {
            "type": "Email"
        },
    },
    "SL": {
        "url": "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz",
        "filename": "Slashdot0902.txt",
        "meta": {
            "type": "Social"
        },
    },
    "DB": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "filename": "com-dblp.ungraph.txt",
        "meta": {
            "type": "Collaboration"
        },
    },
    "AM": {
        "url": "https://snap.stanford.edu/data/amazon0601.txt.gz",
        "filename": "amazon0601.txt",
        "meta": {
            "type": "Co-purchasing"
        },
    },
    "EU": {
        "url": "https://sparse.tamu.edu/MM/LAW/eu-2005.tar.gz",
        "filename": "eu-2005.mtx",
        "meta": {
            "type": "Hyperlink"
        },
    },
    "YT": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "filename": "com-youtube.ungraph.txt",
        "meta": {
            "type": "Social"
        },
    },
    "SK": {
        "url": "https://snap.stanford.edu/data/as-skitter.txt.gz",
        "filename": "as-skitter.txt",
        "meta": {
            "type": "Internet Infrastructure"
        },
    },
    "HW": {
        "url": "https://sparse.tamu.edu/MM/LAW/hollywood-2009.tar.gz",
        "filename": "hollywood-2009.mtx",
        "meta": {
            "type": "Collaboration"
        },
    },
    "UK": {
        "url": "https://sparse.tamu.edu/MM/LAW/uk-2002.tar.gz",
        "filename": "uk-2002.mtx",
        "meta": {
            "type": "Hyperlink"
        },
    },
    "LJ": {
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "filename": "com-lj.ungraph.txt",
        "meta": {
            "type": "Social"
        },
    },
}

DATASET_GROUP = {
    "small": ["CA", "PR", "EN", "BK", "EA", "SL"],   # < 600K edges (Fast debugging)
    "medium": ["FB", "DB", "AM", "YT"],              # 600K - 3M edges (Standard benchmarks)
    "large": ["SK", "EU", "LJ", "HW", "UK"],         # 10M+ edges (Scalability & Memory testing)

    "insertion": [ "PR", "EN", "FB", "EU",
                   #"HW", "UK"
                   ],
    "dynamic": ["DB", "YT", "SK",
                #"LJ"
                ],

    "tuning": ["PR", "EN", "FB", "DB", "YT"],

    "dense": ["PR", "FB", "EU", "HW"],               # High-average degree / high clustering
    "sparse": ["CA", "EA", "DB", "YT"],              # Low average degree / tree-like
    "skewed": ["EN", "YT", "SK", "EU"],              # Massive max-degree hub nodes

    "social": ["FB", "BK", "SL", "YT", "LJ"],
    "web": ["EU", "UK", "SK"]
}
