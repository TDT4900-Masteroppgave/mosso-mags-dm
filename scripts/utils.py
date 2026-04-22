"""Benchmark utilities: logging, environment info, filesystem helpers, dataframe formatting."""
import glob
import logging
import os
import platform
import subprocess
import sys

import pandas as pd

from scripts.config import BENCHMARK_DIR, DATASETS, DATASETS_DIR, OUTPUT_DIR, VERSIONS_DIR


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
    for d in [DATASETS_DIR, OUTPUT_DIR, BENCHMARK_DIR, VERSIONS_DIR]:
        os.makedirs(d, exist_ok=True)


def get_datasets_to_run(args) -> list[dict]:
    datasets: list[dict] = []

    if getattr(args, "dataset", None):
        all_available = [d for group in DATASETS.values() for d in group]
        for req in args.dataset:
            matched = next(
                (d for d in all_available if d["short_name"] == req or d["filename"] == req),
                None,
            )
            if matched:
                if matched not in datasets:
                    datasets.append(matched)
            else:
                print(f"[!] Warning: Dataset '{req}' not found in configuration. Skipping.")

        if not datasets:
            raise ValueError("No valid datasets found based on your --dataset argument.")
    else:
        if args.group == "all":
            for group_datasets in DATASETS.values():
                datasets.extend(group_datasets)
        else:
            datasets = DATASETS.get(args.group, [])
            if not datasets:
                raise ValueError(f"Dataset group '{args.group}' not found in config.")

    return datasets


def format_dataframe_with_baseline(
    df: pd.DataFrame,
    strategies: list[str],
    baseline_algo: str | None = None,
    time_prefix: str = "Time_",
    ratio_prefix: str = "Ratio_",
) -> pd.DataFrame:
    """Format time/ratio columns with ±std and optional speedup vs baseline."""
    display_df = df.copy()

    for strat in strategies:
        time_col = f"{time_prefix}{strat}"
        ratio_col = f"{ratio_prefix}{strat}"
        t_std_col = f"{time_prefix}std_{strat}"
        r_std_col = f"{ratio_prefix}std_{strat}"

        formatted_times, formatted_ratios = [], []

        for _, row in df.iterrows():
            t_val, r_val = row.get(time_col), row.get(ratio_col)
            t_std = row.get(t_std_col, 0.0)
            r_std = row.get(r_std_col, 0.0)

            t_str = (
                f"{t_val:.3f}s ± {t_std:.3f}s" if t_std and t_std > 0 else f"{t_val:.3f}s"
            ) if pd.notna(t_val) else "N/A"

            r_str = (
                f"{r_val:.5f} ± {r_std:.5f}" if r_std and r_std > 0 else f"{r_val:.5f}"
            ) if pd.notna(r_val) else "N/A"

            if baseline_algo and baseline_algo in strategies and strat != baseline_algo:
                t_base = row.get(f"{time_prefix}{baseline_algo}")
                r_base = row.get(f"{ratio_prefix}{baseline_algo}")

                if pd.notna(t_val) and pd.notna(t_base) and t_val > 0:
                    t_str += f" ({t_base / t_val:.2f}x)"

                if pd.notna(r_val) and pd.notna(r_base) and r_base > 0:
                    r_str += f" ({r_val / r_base:.2f}x)"

            formatted_times.append(t_str)
            formatted_ratios.append(r_str)

        display_df[time_col] = formatted_times
        display_df[ratio_col] = formatted_ratios

    std_cols = [c for c in display_df.columns if "_std_" in c]
    return display_df.drop(columns=std_cols)
