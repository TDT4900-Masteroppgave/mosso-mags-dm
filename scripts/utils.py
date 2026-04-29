"""Benchmark utilities: logging, environment info, filesystem helpers, dataframe formatting."""
import glob
import logging
import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.config import EXPERIMENT_DIR, DATASETS, DATASETS_DIR, OUTPUT_DIR, ALGORITHMS_DIR, DATASET_GROUP


def setup_logging(log_file_path: str):
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_repo_info(path: str = ".") -> dict:
    """Git commit SHA, branch, and dirty-flag for repo at path."""
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=path, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unavailable"

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def _cmd_version(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return "unavailable"


def get_env_info() -> dict:
    """Collect host environment metadata for reproducibility."""
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        ram_gb = "unavailable"

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": sys.version.split()[0],
        "java": _cmd_version(["java", "-version"]),
        "cmake": _cmd_version(["cmake", "--version"]),
        "cxx": (
            _cmd_version(["clang++", "--version"])
            if platform.system() == "Darwin"
            else _cmd_version(["g++", "--version"])
        ),
    }


def get_fastutil_path() -> str:
    fastutil_files = glob.glob("fastutil-*.jar")
    return fastutil_files[0] if fastutil_files else "fastutil-missing.jar"


def setup_directories() -> None:
    for d in [DATASETS_DIR, OUTPUT_DIR, EXPERIMENT_DIR, ALGORITHMS_DIR]:
        os.makedirs(d, exist_ok=True)


def get_datasets_to_run(args) -> list[dict]:
    datasets = []

    if getattr(args, "dataset", None):
        for req in args.dataset:
            if req in DATASETS:
                ds = DATASETS[req].copy()
                ds["short_name"] = req
                datasets.append(ds)
            else:
                print(f"[!] Warning: Dataset '{req}' not found.")

    elif getattr(args, "group", None):
        if args.group == "all":
            keys_to_run = list(DATASETS.keys())
        else:
            keys_to_run = DATASET_GROUP.get(args.group, [])

        for key in keys_to_run:
            ds = DATASETS[key].copy()
            ds["short_name"] = key
            datasets.append(ds)

    return datasets

def format_long_dataframe_with_baseline(df: pd.DataFrame, baseline_algo: str | None = None) -> pd.DataFrame:
    """Aggregates raw results into averages and formats them with baseline speedup multipliers."""
    if df.empty:
        return df

    summary = df.groupby(['dataset', 'algorithm'], as_index=False)[['time', 'ratio']].mean()

    baselines = {}
    if baseline_algo and baseline_algo in summary['algorithm'].values:
        baselines = summary[summary['algorithm'] == baseline_algo].set_index('dataset').to_dict('index')

    def format_row(row):
        t, r = row['time'], row['ratio']
        t_str, r_str = f"{t:.3f}s", f"{r:.5f}"

        base = baselines.get(row['dataset'])
        if base and row['algorithm'] != baseline_algo:
            # Append baseline multipliers
            if t > 0 and base['time'] > 0:
                t_str += f" [green]({base['time'] / t:.2f}x)[/green]"
            if base['ratio'] > 0:
                r_str += f" [green]({r / base['ratio']:.2f}x)[/green]"

        return pd.Series({
            "Dataset": str(row['dataset']).capitalize(),
            "Algorithm": row['algorithm'],
            "Avg Time": t_str,
            "Avg Ratio": r_str
        })

    return summary.apply(format_row, axis=1).sort_values(by=["Dataset", "Algorithm"])

def get_confidence_interval(
        xs: List[float],
        stat=np.median,
        n: int = 10000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for *stat* applied to *xs*.
    Returns (lower, upper) bounds at the (alpha/2, 1-alpha/2) percentiles.
    """
    if not xs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    resampled = rng.choice(arr, size=(n, len(arr)), replace=True)
    stats = stat(resampled, axis=1)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi