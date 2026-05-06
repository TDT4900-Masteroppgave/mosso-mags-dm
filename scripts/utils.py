import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from rich.console import Console
from rich.errors import MarkupError
from rich.text import Text

from scripts.config import (ALGORITHMS_DIR, DATASET_GROUP, DATASETS,
                            DATASETS_DIR, EXPERIMENT_DIR, OUTPUT_DIR)


def setup_logging(log_file_path: str) -> logging.Logger:
    """Sets up a file-ONLY logger. Rich handles the terminal natively."""
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.DEBUG)
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
        """Strips rich markup tags for clean log files."""
        try:
            return Text.from_markup(str(msg)).plain
        except MarkupError:
            return str(msg)

    def _capture_and_log(self, method_name: str, *args, **kwargs):
        """Helper to run a console method, capture its plain output, and log it."""
        with self.text_console.capture() as cap:
            getattr(self.text_console, method_name)(*args, **kwargs)

        for line in cap.get().splitlines():
            if line.strip():
                self.logger.info(line)

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        self._capture_and_log("print", *args, **kwargs)

    def rule(self, *args, **kwargs):
        self.console.rule(*args, **kwargs)
        self._capture_and_log("rule", *args, **kwargs)

    def status(self, *args, **kwargs):
        if args:
            self._capture_and_log("print", f"Starting: {args[0]}")
        return self.console.status(*args, **kwargs)

    def debug(self, msg: str):
        self.logger.debug(self._clean(msg))

    def info(self, msg: str):
        self.logger.info(self._clean(msg))
        self.console.print(msg)

    def warning(self, msg: str):
        self.logger.warning(self._clean(msg))
        self.console.print(f"[bold yellow]{msg}[/bold yellow]")

    def error(self, msg: str):
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
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip().splitlines()[0]
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
        "cxx": _cmd_version(["clang++", "--version"]) if platform.system() == "Darwin" else _cmd_version(["g++", "--version"]),
    }


def get_fastutil_path() -> str:
    return next((str(p) for p in Path(".").glob("fastutil-*.jar")), "fastutil-missing.jar")


def setup_directories() -> None:
    for d in [DATASETS_DIR, OUTPUT_DIR, EXPERIMENT_DIR, ALGORITHMS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_datasets_to_run(args) -> list[dict]:
    # Determine the keys to run based on group vs specific dataset
    keys = args.dataset if getattr(args, "dataset", None) else (
        list(DATASETS.keys()) if args.group == "all" else DATASET_GROUP.get(args.group, [])
    )

    datasets = []
    for key in keys:
        if key in DATASETS:
            datasets.append({**DATASETS[key], "short_name": key})
        else:
            print(f"[!] Warning: Dataset '{key}' not found.")

    return datasets


def format_long_dataframe_with_baseline(df: pd.DataFrame, baseline_algo: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    summary = df.groupby(['dataset', 'algorithm'], as_index=False)[['time', 'ratio']].mean()

    baselines = {}
    if baseline_algo and baseline_algo in summary['algorithm'].values:
        baselines = summary[summary['algorithm'] == baseline_algo].set_index('dataset').to_dict('index')

    def format_row(row):
        t, rat = row['time'], row['ratio']
        algo, ds = row['algorithm'], row['dataset']

        t_str, r_str = f"{t:.3f}s", f"{rat:.5f}"
        base = baselines.get(ds)

        if base and algo != baseline_algo:
            if t > 0 and base['time'] > 0:
                t_str += f" [green]({base['time'] / t:.2f}x)[/green]"
            if base['ratio'] > 0:
                r_str += f" [green]({rat / base['ratio']:.2f}x)[/green]"

        return pd.Series({
            "Dataset": str(ds).capitalize(),
            "Algorithm": algo,
            "Avg Time": t_str,
            "Avg Ratio": r_str
        })

    return summary.apply(format_row, axis=1).sort_values(by=["Dataset", "Algorithm"])


def get_confidence_interval(
        xs: List[float], stat=np.median, n: int = 10000, alpha: float = 0.05, seed: Optional[int] = None
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for *stat* applied to *xs*."""
    if not xs:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    resampled = rng.choice(arr, size=(n, len(arr)), replace=True)
    stats = stat(resampled, axis=1)

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi