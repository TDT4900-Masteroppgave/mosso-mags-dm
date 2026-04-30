"""Benchmark utilities: logging, environment info, filesystem helpers, dataframe formatting."""
import glob
import os
import platform
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.errors import MarkupError

from scripts.config import EXPERIMENT_DIR, DATASETS, DATASETS_DIR, OUTPUT_DIR, ALGORITHMS_DIR, DATASET_GROUP

from rich.console import Console
from rich.text import Text
import logging
import sys
import psutil

def setup_logging(log_file_path: str) -> logging.Logger:
    """Sets up a file-ONLY logger. Rich handles the terminal natively."""
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to completely prevent double-printing
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(fh)
    return logger


class Logger:
    def __init__(self, log_file_path: str):
        self.console = Console(highlight=False)
        self.text_console = Console(color_system=None, width=250)
        self.logger = setup_logging(log_file_path)

    @staticmethod
    def _clean(msg: str) -> str:
        """Strips rich markup tags (like [bold red]) for clean log files."""
        try:
            return Text.from_markup(str(msg)).plain
        except MarkupError:
            return str(msg)

    # --- Standard Print & UI Methods ---
    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        with self.text_console.capture() as cap:
            self.text_console.print(*args, **kwargs)
        for line in cap.get().splitlines():
            if line.strip():
                self.logger.info(line)

    def rule(self, *args, **kwargs):
        self.console.rule(*args, **kwargs)
        with self.text_console.capture() as cap:
            self.text_console.rule(*args, **kwargs)
        for line in cap.get().splitlines():
            if line.strip():
                self.logger.info(line)

    def status(self, *args, **kwargs):
        if args:
            with self.text_console.capture() as cap:
                self.text_console.print(f"Starting: {args[0]}")
            for line in cap.get().splitlines():
                if line.strip():
                    self.logger.info(line.strip())
        return self.console.status(*args, **kwargs)

    # --- Standard Logging Equivalents ---
    def debug(self, msg: str):
        """Debug goes ONLY to the log file, keeping the terminal clean."""
        self.logger.debug(self._clean(msg))

    def info(self, msg: str):
        """Info prints to the terminal normally, and logs to the file cleanly."""
        self.logger.info(self._clean(msg))
        self.console.print(msg)

    def warning(self, msg: str):
        """Warnings get automatic yellow formatting in the terminal, plain in file."""
        self.logger.warning(self._clean(msg))
        self.console.print(f"[bold yellow]{msg}[/bold yellow]")

    def error(self, msg: str):
        """Errors get automatic red formatting in the terminal, plain in file."""
        self.logger.error(self._clean(msg))
        self.console.print(f"[bold red]{msg}[/bold red]")


def get_repo_info(path: str = ".") -> dict:
    """Git commit SHA, branch, and dirty-flag for repo at the path."""
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=path, stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.SubprocessError, OSError):
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
    except (subprocess.SubprocessError, OSError, IndexError):
        return "unavailable"


def get_env_info() -> dict:
    """Collect host environment metadata for reproducibility."""
    try:
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