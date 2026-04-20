import os

OUTPUT_DIR = "output"
DATASETS_DIR = "datasets"
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmarks")
VERSIONS_DIR = os.path.join(BENCHMARK_DIR, "versions")

BASE_REPO_URL = "https://github.com/TDT4900-Masteroppgave/mosso-mags-dm.git"

PARAM_CONFIG = {
    "c": {"description": "sample number", "default": 120, "bounds": (10, 240)},
    "e": {"description": "escape", "default": 3, "bounds": (1, 5)},
    "interval": {"description": "interval", "default": 10000},
    "b": {"description": "top candidates", "default": 5, "bounds": (1, 10)},
    "h": {"description": "hashes", "default": 4, "bounds": (4, 40)},
    "thr": {"description": "threshold", "default": 0, "bounds": (0.0, 1.0)},
    "cap": {"description": "cap of coarse cluster buckets", "default": 100, "bounds": (5, 240)},
}

ALGORITHMS = {
    "local": {
        "target_dir": ".",
        "type": "mosso",
        "template": ["e", "c", "interval", "h"]
    },
    "kdd20-mosso": {
        "repo": "https://github.com/jihoonko/kdd20-mosso.git",
        "branch": "master",
        "params" : {"c": 120, "e": 3},
        "type": "mosso",
        "template": ["e", "c", "interval"],
        "binary_file": "kdd20-mosso.jar"
    },
    "strat_1": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_1",
        "type": "mosso",
        "template": ["e", "c", "interval", "b"],
        "binary_file": "mosso-strat_1.jar",
    },
    "strat_2": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_2",
        "type": "mosso",
        "template": ["e", "c", "interval", "h"],
        "binary_file": "mosso-strat_2.jar",
    },
    "strat_1_2": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_1_2",
        "type": "mosso",
        "template": ["e", "c", "interval", "b", "h"],
        "binary_file": "mosso-strat_1_2.jar",
    },
    "strat_2_thr": {
        "repo": BASE_REPO_URL,
        "branch": "feature/strat_2_threshold",
        "type": "mosso",
        "template": ["e", "c", "interval", "h", "thr"],
        "binary_file": "mosso-strat_2_thr.jar",
    },
    "strat_2_cap": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_2_cap",
        "type": "mosso",
        "template": ["e", "c", "interval", "h", "cap"],
        "binary_file": "mosso-strat_2_cap.jar",
    },
    "strat_2_2hop": {
        "repo": BASE_REPO_URL,
        "branch": "feature/merging_strategy_2_2hop",
        "type": "mosso",
        "template": ["e", "c",  "interval"],
        "binary_file": "mosso-strat_2_2hop.jar",
    },
    "strat_div": {
        "repo": BASE_REPO_URL,
        "branch": "feature/dividing-strategy",
        "type": "mosso",
        "template": ["e", "c",  "interval"],
        "binary_file": "mosso-strat_div.jar",
    },
    "fix_testing_nodes": {
        "repo": BASE_REPO_URL,
        "branch": "fix/testing_nodes",
        "type": "mosso",
        "template": ["e", "c", "interval"],
        "binary_file": "mosso-fix_testing_nodes.jar",
    },
    "mags": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": [],
        "binary_file": "mags",
    },
    "mags-dm": {
        "repo": "https://github.com/nedchu/mags-release",
        "branch": "main",
        "type": "mags",
        "template": [],
        "binary_file": "mags_dm",
    },
}

DATASETS = {
    "small": [
        {
            "url": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
            "filename": "as-caida20071105.txt",
            "short_name": "CA",
            "meta": {
                "nodes": 26475,
                "edges": 53381,
                "size": "0.66 MB",
                "avg_degree": 4.03,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/email-Enron.txt.gz",
            "filename": "Email-Enron.txt",
            "short_name": "EN",
            "meta": {
                "nodes": 36692,
                "edges": 183831,
                "size": "2.11 MB",
                "avg_degree": 10.02,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz",
            "filename": "Brightkite_edges.txt",
            "short_name": "BK",
            "meta": {
                "nodes": 58228,
                "edges": 214078,
                "size": "2.59 MB",
                "avg_degree": 7.35,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/email-EuAll.txt.gz",
            "filename": "Email-EuAll.txt",
            "short_name": "EA",
            "meta": {
                "nodes": 265009,
                "edges": 364481,
                "size": "4.52 MB",
                "avg_degree": 2.75,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz",
            "filename": "Slashdot0902.txt",
            "short_name": "SL",
            "meta": {
                "nodes": 82168,
                "edges": 504230,
                "size": "6.18 MB",
                "avg_degree": 12.27,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
            "filename": "com-dblp.ungraph.txt",
            "short_name": "DB",
            "meta": {
                "nodes": 317080,
                "edges": 1049866,
                "size": "15.29 MB",
                "avg_degree": 6.62,
            },
        }
    ],
    "large": [
        {
            "url": "https://snap.stanford.edu/data/amazon0601.txt.gz",
            "filename": "amazon0601.txt",
            "short_name": "AM",
            "meta": {
                "nodes": 403394,
                "edges": 2443408,
                "size": "35.09 MB",
                "avg_degree": 12.11,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
            "filename": "com-youtube.ungraph.txt",
            "short_name": "YT",
            "meta": {
                "nodes": 1134890,
                "edges": 2987624,
                "size": "42.63 MB",
                "avg_degree": 5.27,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/as-skitter.txt.gz",
            "filename": "as-skitter.txt",
            "short_name": "SK",
            "meta": {
                "nodes": 1696415,
                "edges": 11095298,
                "size": "163.36 MB",
                "avg_degree": 13.08,
            },
        },
        {
            "url": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
            "filename": "com-lj.ungraph.txt",
            "short_name": "LJ",
            "meta": {
                "nodes": 3997962,
                "edges": 34681189,
                "size": "544.47 MB",
                "avg_degree": 17.35,
            },
        }
    ]
}