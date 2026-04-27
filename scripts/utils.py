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
    """Formats a long-form DataFrame with baseline speedup multipliers and CIs."""
    display_df = df.copy()

    base_time_map, base_ratio_map = {}, {}
    if baseline_algo and baseline_algo in df['algorithm'].values:
        baseline_data = df[df['algorithm'] == baseline_algo].groupby('dataset')[['time', 'ratio']].mean()
        base_time_map = baseline_data['time'].to_dict()
        base_ratio_map = baseline_data['ratio'].to_dict()

    formatted_times = []
    formatted_ratios = []

    for _, row in display_df.iterrows():
        t_val = row.get('time')
        r_val = row.get('ratio')
        dataset = row.get('dataset')
        algo = row.get('algorithm')

        # Extract DB Confidence Intervals
        t_lo, t_hi = row.get('time_ci_lo'), row.get('time_ci_hi')
        r_lo, r_hi = row.get('ratio_ci_lo'), row.get('ratio_ci_hi')

        # Format Time with CI and speedup
        t_str = "N/A"
        if pd.notna(t_val):
            t_str = f"{t_val:.3f}s"
            if pd.notna(t_lo) and pd.notna(t_hi):
                t_str += f" [{t_lo:.3f}, {t_hi:.3f}]"

            if baseline_algo and algo != baseline_algo and dataset in base_time_map:
                t_base = base_time_map[dataset]
                if t_val > 0 and t_base > 0:
                    t_str += f" ({t_base / t_val:.2f}x)"

        # Format Ratio with CI and multiplier
        r_str = "N/A"
        if pd.notna(r_val):
            r_str = f"{r_val:.5f}"
            if pd.notna(r_lo) and pd.notna(r_hi):
                r_str += f" [{r_lo:.5f}, {r_hi:.5f}]"

            if baseline_algo and algo != baseline_algo and dataset in base_ratio_map:
                r_base = base_ratio_map[dataset]
                if r_base > 0:
                    r_str += f" ({r_val / r_base:.2f}x)"

        formatted_times.append(t_str)
        formatted_ratios.append(r_str)

    display_df['time'] = formatted_times
    display_df['ratio'] = formatted_ratios

    # Drop the raw CI columns from the final terminal table to keep it clean
    cols_to_drop = [c for c in display_df.columns if '_ci_' in c]
    return display_df.drop(columns=cols_to_drop, errors='ignore')

def _run_build(cmd: list[str], cwd=None, env=None):
    return subprocess.run(
        cmd, cwd=cwd, env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

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