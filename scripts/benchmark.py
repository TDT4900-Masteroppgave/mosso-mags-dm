import json
import os.path
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
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.config import PARAM_CONFIG, ALGORITHMS, DATASETS, BENCHMARK_DIR
from scripts.utils import (
    setup_logging, setup_directories, get_datasets_to_run,
    get_repo_info, get_env_info, format_dataframe_with_baseline,
    get_confidence_interval
)
from scripts.datasets import download_dataset
from scripts.runners.base_runner import get_runner
import scripts.db as db


class Benchmark(ABC):
    def __init__(self, benchmark_type: str):
        self.benchmark_type = benchmark_type
        self.results: list[dict[str, Any]] = []
        self.datasets_to_run = None
        self.active_algos: dict = {}

        self.args = self._parse_arguments()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(BENCHMARK_DIR) / self.benchmark_type / f"run_{self.timestamp}"
        self.runs_dir = self.session_dir / "runs"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.session_dir / "execution.log"
        self.logger = setup_logging(str(log_file))
        self.console = Console(highlight=False)
        self.db_conn = None

        self.all_times_dict, self.all_ratios_dict = {}, {}

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        return

    @abstractmethod
    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict[str, Any]] | None:
        pass

    def run(self) -> None:
        """The main execution lifecycle."""
        start_time = time.time()

        try:
            self._run_setup()
            self._process_datasets()
            self._handle_results()
        except Exception as e:
            self.logger.error(f"[!] Benchmark aborted: {e}")
            self.logger.debug(traceback.format_exc())
            exit(1)

        elapsed = time.time() - start_time
        self.console.print(f"[dim]Total Benchmark Time:[/dim] {elapsed:.2f} seconds")
        self.console.print(f"[dim]Output:[/dim] {self.session_dir}")

        if self.db_conn:
            self.db_conn.close()

    def _run_setup(self):
        self.console.rule("[bold]Setup[/bold]")

        self.datasets_to_run = get_datasets_to_run(self.args)
        setup_directories()

        self.db_conn = db.init_db(self.session_dir)
        self.print_parameters()
        self._build_algorithms()

    def _process_datasets(self) -> None:
        self.console.rule("[bold]Processing[/bold]")
        n = len(self.datasets_to_run)
        for i, ds in enumerate(self.datasets_to_run, 1):
            url = ds.get("url", "None")
            filename = ds.get("filename", "None")
            short_name = ds.get("short_name", "N/A")

            dataset_path = download_dataset(url, filename, self.logger)
            if not dataset_path:
                raise RuntimeError(f"Failed to download dataset {filename}.")

            self.console.print(
                f"[bold cyan][{i}/{n}][/bold cyan] {short_name} ({self.args.runs} run{'s' if self.args.runs != 1 else ''})")

            metrics_list = self.process(dataset_path, short_name)
            if not metrics_list:
                self.logger.warning(f"No results returned for {filename}")
                continue

            for row in metrics_list:
                if "parameters" in row:
                    row["params"] = json.dumps(row.pop("parameters"))
                self.results.append(row)

    def _handle_results(self) -> None:
        if not self.results:
            self.logger.warning("[!] No results generated")
            return

        if self.db_conn and self.results:
            raw_df = pd.DataFrame(self.results)
            raw_df.columns = raw_df.columns.str.lower()
            db.write_results_bulk(self.db_conn, raw_df)

        self.console.rule("[bold]Results[/bold]")
        self.finalize()

    def finalize(self):
        if not self.db_conn:
            self.logger.warning("No database connection. Skipping table generation.")
            return

        long_df = db.read_results(self.db_conn)
        if long_df.empty:
            return

        metadata_cols = ['dataset', 'algorithm', 'run', 'parameters']

        dynamic_metrics = [
            col for col in long_df.columns
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(long_df[col])
        ]

        def ci_lo(x):
            valid = x.dropna()
            if len(valid) < 2: return np.nan
            lower_bound, _ = get_confidence_interval(valid.tolist(), stat=np.mean, seed=self.args.seed)
            return lower_bound

        def ci_hi(x):
            valid = x.dropna()
            if len(valid) < 2: return np.nan
            _, upper_bound = get_confidence_interval(valid.tolist(), stat=np.mean, seed=self.args.seed)
            return upper_bound

        wide_df = long_df.pivot_table(
            index=['dataset'],
            columns='algorithm',
            values=dynamic_metrics,
            aggfunc=['mean', 'std', ci_lo, ci_hi]
        )

        flattened_cols = []
        for agg_func, metric, algo in wide_df.columns:
            m_cap = metric.capitalize()
            if agg_func == 'mean':
                flattened_cols.append(f"{m_cap}_{algo}")
            elif agg_func == 'std':
                flattened_cols.append(f"{m_cap}_std_{algo}")
            elif agg_func == 'ci_lo':
                flattened_cols.append(f"{m_cap}_ci_lo_{algo}")
            elif agg_func == 'ci_hi':
                flattened_cols.append(f"{m_cap}_ci_hi_{algo}")

        wide_df.columns = flattened_cols
        wide_df = wide_df.reset_index()

        avg_row = wide_df.mean(numeric_only=True).to_dict()
        avg_row['dataset'] = 'AVERAGE'

        algorithms = long_df['algorithm'].unique().tolist()
        for algo in algorithms:
            for metric in dynamic_metrics:
                m_cap = metric.capitalize()
                col = f"{m_cap}_{algo}"
                if col in wide_df.columns:
                    vals = wide_df[col].dropna().tolist()
                    if len(vals) >= 2:
                        lo, hi = get_confidence_interval(vals, stat=np.mean, seed=self.args.seed)
                        avg_row[f"{m_cap}_ci_lo_{algo}"] = lo
                        avg_row[f"{m_cap}_ci_hi_{algo}"] = hi
                    else:
                        avg_row[f"{m_cap}_ci_lo_{algo}"] = np.nan
                        avg_row[f"{m_cap}_ci_hi_{algo}"] = np.nan

        table_df = pd.concat([wide_df, pd.DataFrame([avg_row])], ignore_index=True)

        strategies = [
            col.replace("Time_", "") for col in table_df.columns
            if col.startswith("Time_") and not col.startswith("Time_ci") and not col.startswith("Time_std")
        ]

        display_df = format_dataframe_with_baseline(table_df, strategies, self.args.baseline)

        res_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")

        for col in display_df.columns:
            res_table.add_column(str(col))

        for _, row in display_df.iterrows():
            res_table.add_row(*[str(val) if pd.notna(val) else "" for val in row])

        self.console.print(res_table)

        text_console = Console(width=250, color_system=None)
        with text_console.capture() as capture:
            text_console.print(res_table)

        table_str = capture.get()
        # Log to execution log
        for line in table_str.split('\n'):
            if line.strip():
                self.logger.debug(line)

    def _resolve_algo_params(self, algo_config: dict) -> dict[str, str]:
        cmd: dict[str, str] = {}
        template = algo_config.get('template', [])
        for param in template:
            param_val: str
            if algo_config.get('params', {}).get(param) is not None:
                param_val = algo_config.get('params', {}).get(param)
            elif [param in self.args for param in PARAM_CONFIG]:
                param_val = getattr(self.args, param)
            else:
                param_val = PARAM_CONFIG.get(param, {}).get("default")
            cmd[param] = str(param_val)
        return cmd

    def print_metrics(self, algo_name: str, t_avg: float, r_avg: float, t_list: list[float],
                      r_list: list[float]) -> None:
        """Prints the metrics in a table."""
        if len(t_list) > 1 and len(r_list) > 1:
            t_std = np.std(t_list)
            r_std = np.std(r_list)

            t_lo, t_hi = get_confidence_interval(t_list, seed=self.args.seed)
            r_lo, r_hi = get_confidence_interval(r_list, seed=self.args.seed)

            t_ci_str = f"CI=[{t_lo:.3f},{t_hi:.3f}]" if t_lo is not None and t_hi is not None else ""
            r_ci_str = f"CI=[{r_lo:.3f},{r_hi:.3f}]" if r_lo is not None and r_hi is not None else ""
            self.logger.info(f"=> {algo_name: <12} Time: {t_avg:.3f}s ± {t_std:.3f}s {t_ci_str} "
                             f"| Ratio: {r_avg:.3f} ± {r_std:.3f} {r_ci_str}")

            self.all_times_dict[algo_name] = t_list
            self.all_ratios_dict[algo_name] = r_list
        else:
            self.logger.info(f"=> {algo_name: <12} Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}")

    def execute_runner(self, algo_name: str, dataset_path: str,
                       parameters: dict[str, str]) -> tuple[float | None, float | None, list | None, list | None]:
        """A helper method to standardize runner execution across subclasses."""
        runner = get_runner(algo_name, self.logger, str(self.session_dir))

        if not runner.binary_exists():
            self.logger.warning(f"[!] Binary not found for {algo_name}. Skipping.")
            return None, None, None, None

        dataset_name = os.path.basename(dataset_path)

        with self.console.status(f"[bold blue]Running {algo_name} | params: {parameters} [/bold blue]"):
            t_avg, r_avg, t_list, r_list = runner.run_multiple(
                dataset_path=dataset_path,
                base_output_name=f"{algo_name}_{dataset_name}_{self.timestamp}",
                runs=self.args.runs,
                parameters=list(parameters.values())
            )

        if t_avg is None or r_avg is None or t_list is None or r_list is None:
            self.logger.warning(f"\t=> {algo_name} failed to run with params: {parameters}")
            return None, None, None, None

        self.print_metrics(algo_name, t_avg, r_avg, t_list, r_list)

        return t_avg, r_avg, t_list, r_list

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
        self.logger.debug(f"Manifest written to {manifest_path}")

    def _parse_arguments(self) -> argparse.Namespace:
        """Builds the parser, collects custom args, and parses them."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--runs", type=int, default=1)

        data_group = parser.add_mutually_exclusive_group()
        data_group.add_argument("--group", choices=["all"] + list(DATASETS.keys()), default="all")
        data_group.add_argument("--dataset", nargs='+', type=str,
                                help="Specific dataset(s) to run by short_name (e.g., YT) or filename. Overrides --group.")

        parser.add_argument("--algorithm", nargs='+', help="Specific algorithms to run")
        parser.add_argument("--baseline", type=str, help="Algorithm for relative comparisons")
        parser.add_argument("--is-local", action="store_true",
                            help="Include the local directory code in the benchmark")
        parser.add_argument("--seed", type=int, default=42,
                            help="Global random seed for reproducibility")

        for p_name, p_data in PARAM_CONFIG.items():
            parser.add_argument(f"--{p_name}", type=type(p_data["default"]), default=p_data["default"])

        self.add_custom_args(parser)
        args = parser.parse_args()

        if args.dataset:
            delattr(args, "group")
        elif args.group:
            delattr(args, "dataset")

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

        if args.baseline and args.baseline not in ALGORITHMS.keys():
            print(f"[!] The specified baseline '{args.baseline}' is not in the active algorithms list.")
            exit(1)

        return args

    def _build_algorithms(self) -> None:
        self.console.rule("[bold]Building Algorithms[/bold]")
        for algo_name, config in self.active_algos.items():
            try:
                runner = get_runner(algo_name, self.logger, str(self.session_dir))
                self.logger.debug(f"[bold blue]Building {algo_name} {config}[/bold blue]")

                with self.console.status(f"[bold blue]Building {algo_name} "
                                         f"{config.get('repo', '')} "
                                         f"{config.get('branch', '')} "
                                         f"{config.get('binary_file', '')}[/bold blue]"):
                    runner.build()
                self.console.print(f"[green]✓[/green] {algo_name} "
                                   f"{config.get('repo', '')} "
                                   f"{config.get('branch', '')} "
                                   f"{config.get('binary_file', '')}")
            except Exception as e:
                self.console.print(f"[red]✗[/red] {algo_name} ({e})")
                raise

    def print_parameters(self) -> None:
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
                if getattr(self.args, "param", None) == p_key:
                    if getattr(self.args, "range", None):
                        r = self.args.range
                        display_val = f"range({r[0]}, {r[1]}, {r[2]})"
                    elif getattr(self.args, "values", None):
                        display_val = f"values{self.args.values}"
                    else:
                        display_val = f"bounds{PARAM_CONFIG[p_key].get('bounds')}"

                    hp_table.add_row(p_key, display_val, "[yellow]SWEEP[/yellow]")

                elif p_key in params or PARAM_CONFIG[p_key].get("bounds") is None:
                    fixed_val = params.get(p_key, getattr(self.args, p_key))
                    hp_table.add_row(p_key, str(fixed_val), "[dim]FIXED[/dim]")

                else:
                    base_val = getattr(self.args, p_key, "N/A")
                    hp_table.add_row(p_key, str(base_val), "DYNAMIC")
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
