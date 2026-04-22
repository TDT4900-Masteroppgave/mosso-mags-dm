"""Benchmark base class and ResultWriter for CSV/table output."""
import json
import random
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
import traceback

import time

import numpy as np
import pandas as pd
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.config import PARAM_CONFIG, ALGORITHMS, DATASETS, BENCHMARK_DIR
from scripts.utils import (
    setup_logging, setup_directories, get_datasets_to_run,
    get_repo_info, get_env_info,
)
from scripts.datasets import download_dataset
from scripts.runners import get_runner
import scripts.db as db


# ---------------------------------------------------------------------------
# Result output
# ---------------------------------------------------------------------------

class ResultWriter:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir

    def save_csv(self, df: pd.DataFrame, filename: str = "results.csv") -> Path:
        path = self.session_dir / filename
        df.to_csv(path, index=False)
        return path

    def render_table(self, df: pd.DataFrame, floatfmt: str = "g") -> str:
        return tabulate(df, headers="keys", tablefmt="grid", showindex=False, floatfmt=floatfmt)

    def save_table_txt(self, content: str, filename: str = "table_results.txt") -> Path:
        path = self.session_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def log_table(self, logger, table_str: str) -> None:
        for line in table_str.split("\n"):
            logger.info(line)

    def emergency_dump(self, results: list, timestamp: str) -> Path:
        path = self.session_dir / f"EMERGENCY_DUMP_{timestamp}.json"
        path.write_text(json.dumps(results, indent=4), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Abstract benchmark lifecycle
# ---------------------------------------------------------------------------

class Benchmark(ABC):
    def __init__(self, benchmark_type: str):
        self.benchmark_type = benchmark_type
        self.results: list[dict[str, Any]] = []
        self.datasets_to_run = None
        self.active_algos: dict = {}

        self.args = self._parse_arguments()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(BENCHMARK_DIR) / self.benchmark_type / self.get_session_name()
        self.runs_dir = self.session_dir / "runs"
        self.summaries_dir = self.session_dir / "summarized_graphs"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.session_dir / "execution.log"
        self.logger = setup_logging(str(log_file))
        self.console = Console(highlight=False)
        self.db_conn = None

    def run(self) -> None:
        """The main execution lifecycle."""
        start_time = time.time()

        self._run_setup()
        self._process_datasets()
        self._handle_results()

        elapsed = time.time() - start_time
        self.logger.info(f"[*] Total Benchmark Time: {elapsed:.2f} seconds")
        self.logger.info(f"[*] Artifacts available in: {self.session_dir}")

        if self.db_conn:
            self.db_conn.close()

    def _run_setup(self):
        self.console.rule("[bold]SETUP[/bold]")
        self.console.print(f"[dim]Output:[/dim] {self.session_dir}")
        try:
            self.setup()
            self.db_conn = db.init_db(self.session_dir)
            db.write_metadata(self.db_conn, self.benchmark_type,
                              {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
                              self.session_dir)
            self.print_parameters()
            self._build_algorithms()
            self._write_manifest()
        except Exception as e:
            self.logger.error(f"[!] Setup aborted: {e}")
            self.logger.debug(traceback.format_exc())

    def _process_datasets(self) -> None:
        self.console.print()
        self.console.rule("[bold]Processing[/bold]")
        n = len(self.datasets_to_run)
        for i, ds in enumerate(self.datasets_to_run, 1):
            url = ds["url"]
            filename = ds["filename"]
            short_name = ds.get("short_name", filename)

            try:
                dataset_path = download_dataset(url, filename, self.logger)

                if not dataset_path:
                    raise RuntimeError(f"Failed to download dataset {filename}.")

                self.console.print(f"[bold cyan][{i}/{n}][/bold cyan] {short_name} ({self.args.runs} run{'s' if self.args.runs != 1 else ''})")
                self.process(dataset_path, ds, short_name)

            except Exception as e:
                self.logger.error(f"[!] Processing aborted for {filename}: {e}")
                self.logger.debug(traceback.format_exc())
                continue

    def _handle_results(self) -> None:
        if not self.results:
            self.logger.warning("[!] No results generated. Nothing to save.")
            return

        self.console.print()
        self.console.rule("[bold]Results[/bold]")
        try:
            self.print_table()
            self.finalize()
        except Exception as e:
            self.logger.error(f"[!] Error during table printing or plotting: {e}")
            self.logger.debug(traceback.format_exc())
            self._emergency_dump()

    def _emergency_dump(self) -> None:
        """Ultimate fallback: dump raw results to session_dir so compute data is not lost."""
        fallback_path = self.session_dir / f"EMERGENCY_DUMP_{self.timestamp}.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        self.logger.warning(f"[*] Saved raw fallback data to {fallback_path}")

    def _write_manifest(self) -> None:
        """Write metadata.json to the session dir for reproducibility tracing."""
        algo_shas = {}
        for algo_name, algo_cfg in self.active_algos.items():
            target = algo_cfg.get("target_dir", ".")
            info = get_repo_info(target)
            algo_shas[algo_name] = {
                "commit": info["commit"],
                "branch": algo_cfg.get("branch", info["branch"]),
                "repo": algo_cfg.get("repo", "local"),
                "dirty": info["dirty"],
            }

        manifest = {
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp,
            "seed": self.args.seed,
            "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
            "this_repo": get_repo_info(),
            "algorithms": algo_shas,
            "environment": get_env_info(),
        }
        manifest_path = self.session_dir / "metadata.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        self.logger.info(f"[*] Manifest written to {manifest_path}")

    def _parse_arguments(self) -> argparse.Namespace:
        """Builds the parser, collects custom args, and parses them."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--runs", type=int, default=1)
        parser.add_argument("--group", choices=["all"] + list(DATASETS.keys()), default="all")
        parser.add_argument("--algorithm", nargs='+', help="Specific algorithms to run")
        parser.add_argument("--baseline", type=str, help="Algorithm for relative comparisons")
        parser.add_argument("--dataset", nargs='+', type=str,
                            help="Specific dataset(s) to run by short_name (e.g., YT) or filename. Overrides --group.")
        parser.add_argument("--is-local", action="store_true",
                            help="Include the local directory code in the benchmark.")
        parser.add_argument("--seed", type=int, default=42,
                            help="Global random seed for reproducibility (LHS, Optuna, numpy).")

        for p_name, p_data in PARAM_CONFIG.items():
            parser.add_argument(f"--{p_name}", type=type(p_data["default"]), default=p_data["default"])

        self.add_custom_args(parser)
        args = parser.parse_args()

        # Seed all RNGs immediately after parsing
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Configure Active Algorithms
        if args.algorithm:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k in args.algorithm}
        else:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k != "local"}

        if args.is_local:
            self.active_algos["local"] = ALGORITHMS["local"]

        if args.baseline and args.baseline not in ALGORITHMS:
            print(f"[!] The specified baseline '{args.baseline}' is not in the active algorithms list.")
            exit(1)

        return args

    def _resolve_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a resolved parameter dict from defaults, then apply overrides."""
        resolved: dict[str, Any] = {}
        for p_key in PARAM_CONFIG.keys():
            resolved[p_key] = getattr(self.args, p_key)
        if overrides:
            resolved.update(overrides)
        return resolved

    def _resolve_algo_params(self, algo_config: dict, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Resolve params for a specific algorithm: base defaults → fixed algo params → extra overrides."""
        resolved = self._resolve_params()
        resolved.update(algo_config.get("params", {}))
        if extra:
            resolved.update(extra)
        return resolved

    def print_parameters(self) -> None:
        self.console.print()

        # General parameters
        gen_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        gen_table.add_column("Argument")
        gen_table.add_column("Value")
        for k, v in vars(self.args).items():
            if k not in PARAM_CONFIG:
                display_v = ", ".join(map(str, v)) if isinstance(v, list) else str(v) if v is not None else ""
                gen_table.add_row(k, display_v)
        self.console.print("[bold]Parameters[/bold]")
        self.console.print(gen_table)

        # Hyperparameters per algorithm
        for algo_name, algo_config in self.active_algos.items():
            template = algo_config.get('template', [])
            params = algo_config.get('params', {})

            self.console.print(f"[bold]Hyperparameters:[/bold] {algo_name}")
            if not template:
                self.console.print("  (none required)\n")
                continue

            hp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            hp_table.add_column("Param")
            hp_table.add_column("Value", justify="right")
            hp_table.add_column("State")
            for p_key in template:
                if p_key in params:
                    hp_table.add_row(p_key, str(params[p_key]), "[dim]FIXED[/dim]")
                else:
                    base_val = getattr(self.args, p_key, "N/A")
                    display_val = self.get_algo_param_display(p_key, base_val)
                    hp_table.add_row(p_key, display_val, "DYNAMIC")
            self.console.print(hp_table)

        # Datasets
        self.console.print(f"[bold]Datasets[/bold] ({len(self.datasets_to_run)})")
        ds_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        ds_table.add_column("ID")
        ds_table.add_column("Dataset")
        ds_table.add_column("Size", justify="right")
        ds_table.add_column("Nodes", justify="right")
        ds_table.add_column("Edges", justify="right")
        ds_table.add_column("Avg Deg", justify="right")
        for ds in self.datasets_to_run:
            filename = ds["filename"]
            short_name = ds.get("short_name", "N/A")
            meta = ds.get("meta", {})
            nodes = meta.get("nodes", "N/A")
            edges = meta.get("edges", "N/A")
            disp_nodes = f"{nodes:,}" if isinstance(nodes, int) else str(nodes)
            disp_edges = f"{edges:,}" if isinstance(edges, int) else str(edges)
            ds_table.add_row(
                short_name, filename,
                str(meta.get("size", "N/A")),
                disp_nodes, disp_edges,
                str(meta.get("avg_degree", "N/A")),
            )
        self.console.print(ds_table)

    def execute_runner(self, algo_name: str, algo_config: dict, dataset_path: str, dataset_name: str,
                       resolved_params: dict) -> tuple[float | None, float | None, list, list, list]:
        """A helper method to standardize runner execution across subclasses."""
        runner = get_runner(algo_name, self.logger, str(self.session_dir))

        if not runner.binary_exists():
            self.logger.warning(f"[!] Binary not found for {algo_name}. Skipping.")
            return None, None, [], [], []

        template = algo_config.get('template', [])

        return runner.run_multiple(
            dataset_path=dataset_path,
            base_output_name=f"{algo_name}_{dataset_name}_{self.timestamp}",
            runs=self.args.runs,
            parameters=resolved_params,
            template=template
        )

    def _build_algorithms(self) -> None:
        self.console.print()
        self.console.rule("[bold]Building Algorithms[/bold]")
        for algo_name, config in self.active_algos.items():
            try:
                runner = get_runner(algo_name, self.logger, str(self.session_dir))
                runner.build()
                self.console.print(f"  [green]✓[/green]  {algo_name}")
            except Exception as e:
                self.console.print(f"  [red]✗[/red]  {algo_name}  ({e})")
                raise

    def setup(self) -> None:
        self.datasets_to_run = get_datasets_to_run(self.args)
        setup_directories()

    def get_session_name(self) -> str:
        """Hook to allow subclasses to name output folder"""
        return f"run_{self.timestamp}"

    def get_algo_param_display(self, p_key: str, default_val: Any) -> str:
        """Hook to allows subclasses to override parameter display formatting."""
        return str(default_val)

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        return

    @abstractmethod
    def process(self, dataset_path: str, ds: dict, dataset_name: str) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass

    def print_table(self) -> None:
        return
