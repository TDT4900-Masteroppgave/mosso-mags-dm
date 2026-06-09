import argparse
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table

import scripts.db as db
from scripts.config import ALGORITHMS, DATASET_GROUP, DATASETS, EXPERIMENT_DIR, PARAM_CONFIG
from scripts.datasets import prepare_datasets
from scripts.runners.base_runner import get_runner
from scripts.utils import (
    Logger, get_confidence_interval, get_datasets_to_run,
    get_env_info, get_repo_info, setup_directories
)


class Experiment(ABC):
    def __init__(self, benchmark_type: str):
        self.datasets_to_run = None
        self.benchmark_type = benchmark_type
        self.results: list[dict[str, Any]] = []
        self.args = self._parse_arguments()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(EXPERIMENT_DIR) / self.benchmark_type / f"run_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(str(self.session_dir / "execution.log"))
        self.db_conn = None
        self.datasets = {}

        self._db_initialized = False

        missing_algos = [name for name in self.args.algorithm if name not in ALGORITHMS]
        if missing_algos:
            self.logger.error(f"Missing algorithms: {missing_algos}")
            self.logger.print(
                f"\n[bold red]Error:[/] The following algorithms were not found in config.py: [yellow]{', '.join(missing_algos)}[/]")
            self.logger.print(f"Available options are: [green]{', '.join(ALGORITHMS.keys())}[/]\n")
            sys.exit(1)

        self.active_algos = {name: ALGORITHMS[name] for name in self.args.algorithm}

    def __enter__(self):
        self.db_conn = db.init_db(self.session_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db_conn:
            self.db_conn.close()

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def output(self, df: pd.DataFrame):
        pass

    def run(self) -> None:
        start_time = time.time()
        try:
            self.logger.rule("[bold]Setup[/bold]")
            self.logger.print(f"[bold dim]Command:[/bold dim] [dim]{' '.join(sys.argv)}[/dim]\n")

            self.datasets_to_run = get_datasets_to_run(self.args)
            setup_directories()

            self.logger.print(f"[dim]Output:[/dim] {self.session_dir}")
            self.print_parameters()

            self.logger.rule("[bold]Preprocessing[/bold]")
            self.datasets = prepare_datasets(
                self.datasets_to_run, self.active_algos, self.logger, self.args.dynamic
            )
            self.print_dataset()

            self._build_algorithms()
            self.write_metadata()

            self.logger.rule("[bold]Processing[/bold]")
            self.process()

            if self.results:
                self._handle_results()
            else:
                self.logger.warning("No results generated")

        except Exception as e:
            self.logger.error(f"[!] Benchmark aborted: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            elapsed = time.time() - start_time
            self.logger.print(f"[dim]Total Time:[/dim] {elapsed:.2f} seconds")
            self.logger.print(f"[dim]Output:[/dim] {self.session_dir}")

    def _print_status(self, i: int, short_name: str) -> None:
        run_text = f"({self.args.runs} run{'s' if self.args.runs != 1 else ''})"
        self.logger.print(f"[bold cyan][{i}/{len(self.datasets)}][/bold cyan] {short_name} {run_text}")

    def record_result(self, metric: dict) -> None:
        self.results.append(metric)

        if self.db_conn:
            if not self._db_initialized:
                db.init_results_schema(self.db_conn, list(metric.keys()))
                self._db_initialized = True

            db.write_result(self.db_conn, metric)

    def _handle_results(self) -> None:
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        self.print_result(df)
        self.output(df)

    def print_result(self, df: pd.DataFrame):
        self.logger.rule("[bold]Results[/bold]")
        display_df = df.copy()

        float_cols = display_df.select_dtypes(include=['float', 'float32', 'float64']).columns
        for col in float_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        table = Table(title="All Results", box=box.SIMPLE, show_header=True, header_style="bold cyan")

        for col in display_df.columns:
            table.add_column(str(col).capitalize())

        for _, row in display_df.iterrows():
            row_data = [str(val) if pd.notna(val) else "" for val in row]
            table.add_row(*row_data)

        self.logger.print(table, "\n")

    def _resolve_algo_params(self, algo_config: dict) -> dict[str, str]:
        params = {}
        for param in algo_config.get('template', []):
            config_val = algo_config.get('params', {}).get(param)
            cli_val = getattr(self.args, param, None)
            default_val = PARAM_CONFIG.get(param, {}).get("default")

            # Priority: 1. Algo Config -> 2. CLI Args -> 3. Param Defaults
            params[param] = str(
                config_val if config_val is not None else (cli_val if cli_val is not None else default_val))
        return params

    def _get_dataset_path(self, short_name: str, algo_type: str) -> str | None:
        dataset_path = self.datasets[short_name][algo_type]
        if not dataset_path:
            self.logger.warning(f"[!] Dataset not found for {short_name}. Skipping.")
            return None
        return dataset_path

    def execute_runner(self, dataset_path: str, short_name: str, algo_name: str, params: dict):
        runner = get_runner(algo_name, self.logger, str(self.session_dir))
        if not runner.binary_exists():
            self.logger.warning(f"[!] Binary not found for {algo_name}. Skipping.")
            return None

        with self.logger.status(f"[bold blue]Running {algo_name} | params: {params} [/bold blue]"):
            output_name = f"{algo_name}_{short_name}_{self.timestamp}"
            timeout = getattr(self.args, "timeout", None)
            res = runner.run_multiple(dataset_path, output_name, self.args.runs, list(params.values()), timeout=timeout)

        t_avg, r_avg, t_list, r_list, intermediates = res
        if t_avg is None or r_avg is None:
            self.logger.warning(f"=> {algo_name} failed to run with params: {params}")
            return None

        if len(t_list) > 1:
            t_lo, t_hi = get_confidence_interval(t_list, seed=self.args.seed)
            r_lo, r_hi = get_confidence_interval(r_list, seed=self.args.seed)
            self.logger.info(
                f"=> {algo_name: <12} Time: {t_avg:.3f}s ± {np.std(t_list):.3f}s CI=[{t_lo:.3f},{t_hi:.3f}] | Ratio: {r_avg:.3f} ± {np.std(r_list):.3f} CI=[{r_lo:.3f},{r_hi:.3f}]")
        else:
            self.logger.info(f"=> {algo_name: <12} Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}")

        return res

    def write_metadata(self) -> None:
        if not self.db_conn:
            return

        algorithms_meta = {}
        for name, cfg in self.active_algos.items():
            info = get_repo_info(cfg.get("target_dir", "."))
            algorithms_meta[name] = {
                "commit": info["commit"],
                "branch": cfg.get("branch", info["branch"]),
                "repo": cfg.get("repo", "local"),
                "dirty": info["dirty"]
            }

        datasets_meta = []
        for short_name in self.datasets.keys():
            ds_info = {
                "short_name": short_name,
                "filename": DATASETS.get(short_name, {}).get("filename", "N/A")
            }
            meta = DATASETS.get(short_name, {}).get("meta", {})
            ds_info.update(meta)
            datasets_meta.append(ds_info)

        db.write_metadata(self.db_conn, {
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp,
            "seed": self.args.seed,
            "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
            "this_repo": get_repo_info(),
            "environment": get_env_info(),
            "algorithms": algorithms_meta,
            "datasets": datasets_meta,
        })

    def _build_algorithms(self) -> None:
        for algo_name, config in self.active_algos.items():
            repo = config.get('repo', '')
            branch = config.get('branch', '')

            with self.logger.status(f"[bold blue] Building {algo_name} | repo :{repo} | branch: {branch} [/bold blue]"):
                runner = get_runner(algo_name, self.logger, str(self.session_dir))
                runner.build()

            self.logger.print(f"[green]✓[/green] {algo_name} | repo: {repo} | branch: {branch} ")

    def _parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--runs", type=int, default=1)

        data_group = parser.add_mutually_exclusive_group()
        data_group.add_argument("--group", nargs="+", choices=["all"] + list(DATASET_GROUP.keys()), default=["all"])
        data_group.add_argument("--dataset", nargs='+', choices=list(DATASETS.keys()), type=str)
        parser.add_argument("--dynamic", nargs='*', default=[], type=str, help="List of datasets to treat as dynamic (FD)")

        parser.add_argument("--algorithm", nargs='+')
        parser.add_argument("--is-local", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for algorithm execution.")

        for p_name, p_data in PARAM_CONFIG.items():
            parser.add_argument(f"--{p_name}", type=type(p_data["default"]), default=p_data["default"])

        self.add_custom_args(parser)
        args = parser.parse_args()

        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.algorithm:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k in args.algorithm}
        else:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k != "local"}

        if args.is_local:
            self.active_algos["local"] = ALGORITHMS["local"]

        return args

    def print_dataset(self):
        num_datasets = len(self.datasets)
        self.logger.print(f"\n[bold]Datasets[/bold] ({num_datasets})")

        ds_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        ds_table.add_column("ID")
        ds_table.add_column("Type")  # New column added here
        ds_table.add_column("Nodes", justify="right")
        ds_table.add_column("Edges", justify="right")

        for short_name in self.datasets.keys():
            ds = DATASETS.get(short_name, {})
            meta = ds.get("meta", {})
            nodes = meta.get("nodes", "N/A")
            edges = meta.get("edges", "N/A")
            ds_type = meta.get("type", "N/A")  # Extracting type

            disp_nodes = f"{nodes:,}" if isinstance(nodes, int) else str(nodes)
            disp_edges = f"{edges:,}" if isinstance(edges, int) else str(edges)

            ds_table.add_row(
                short_name,
                ds_type,
                disp_nodes,
                disp_edges,
            )

        self.logger.print(ds_table)

    def print_parameters(self) -> None:
        self.logger.print("[bold]Parameters[/bold]")
        gen_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        gen_table.add_column("Argument")
        gen_table.add_column("Value")

        gen_table.add_row("script_type", self.benchmark_type)

        for key, value in vars(self.args).items():
            # Skip parameters that belong to algorithm configs
            if key not in PARAM_CONFIG:
                if isinstance(value, list):
                    display_val = ", ".join(map(str, value))
                elif value is not None:
                    display_val = str(value)
                else:
                    display_val = ""

                gen_table.add_row(key, display_val)

        self.logger.print(gen_table)

        for algo_name, algo_config in self.active_algos.items():
            self.logger.print(f"\n[bold]Hyperparameters:[/bold] {algo_name}")
            template = algo_config.get('template', [])

            if not template:
                self.logger.print("  (none required)")
                continue

            params = algo_config.get('params', {})
            hp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            hp_table.add_column("Parameter")
            hp_table.add_column("Value", justify="right")
            hp_table.add_column("Type")
            hp_table.add_column("Bounds")
            hp_table.add_column("Step")

            for param_key in template:
                val = params.get(param_key, getattr(self.args, param_key, "N/A"))
                p_config = PARAM_CONFIG.get(param_key, {})
                p_type = p_config.get("type").__name__ if "type" in p_config else ""
                p_bounds = str(p_config.get("bounds", ""))
                p_step = str(p_config.get("step", ""))
                hp_table.add_row(param_key, str(val), p_type, p_bounds, p_step)

            self.logger.print(hp_table)
