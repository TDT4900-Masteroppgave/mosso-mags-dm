import json
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import traceback

from tabulate import tabulate

from scripts.config import PARAM_CONFIG, ALGORITHMS, DATASETS, BENCHMARK_DIR
from scripts.utils import setup_logging, setup_directories, get_datasets_to_run, download_and_prepare_dataset
from scripts.runners import get_runner


class Benchmark(ABC):
    def __init__(self, benchmark_type: str):
        self.benchmark_type = benchmark_type
        self.results: list[Dict[str, Any]] = []
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

    def run(self) -> None:
        """The main execution lifecycle."""
        import time
        start_time = time.time()

        if not self._run_setup():
            return

        self.print_parameters()
        self._process_datasets()
        self._handle_results()

        elapsed = time.time() - start_time
        self.logger.info(f"[*] Total Benchmark Time: {elapsed:.2f} seconds")
        self.logger.info(f"[*] Artifacts available in: {self.session_dir}")

    def _run_setup(self) -> bool:
        self.logger.info("=" * 10 + f"{' SETUP ':^30}" + "=" * 10)
        try:
            self.setup()
            return True
        except Exception as e:
            self.logger.error(f"[!] Setup aborted: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _process_datasets(self) -> None:
        self.logger.info("=" * 10 + f"{' PROCESSING ':^30}" + "=" * 10)
        for i, ds in enumerate(self.datasets_to_run, 1):
            url = ds["url"]
            filename = ds["filename"]
            short_name = ds.get("short_name", filename)

            try:
                dataset_path = download_and_prepare_dataset(url, filename, self.logger)

                if not dataset_path:
                    raise RuntimeError(f"Failed to download dataset {filename}.")

                self.logger.info(f"[{i}/{len(self.datasets_to_run)}] Benchmarking [{short_name}] ({self.args.runs} runs) ...")
                self.process(dataset_path, short_name)

            except Exception as e:
                self.logger.error(f"[!] Processing aborted for {filename}: {e}")
                self.logger.debug(traceback.format_exc())
                continue

    def _handle_results(self) -> None:
        if not self.results:
            self.logger.warning("[!] No results generated. Nothing to save.")
            return

        self.logger.info("=" * 10 + f"{' RESULTS ':^30}" + "=" * 10)
        try:
            self.print_table()
            self.finalize()
        except Exception as e:
            self.logger.error(f"[!] Error during table printing or plotting: {e}")
            self.logger.debug(traceback.format_exc())
            self._emergency_dump()

    def _emergency_dump(self) -> None:
        """Ultimate fallback: Dump raw dictionaries so hours of compute data are not lost."""
        fallback_path = Path.cwd() / f"EMERGENCY_DUMP_{self.timestamp}.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        self.logger.warning(f"[*] Saved raw fallback data to {fallback_path}")

    def _parse_arguments(self) -> argparse.Namespace:
        """Builds the parser, collects custom args, and parses them."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--runs", type=int, default=1)
        parser.add_argument("--group", choices=["all"] + list(DATASETS.keys()), default="all")
        parser.add_argument("--algorithm", nargs='+', help="Specific algorithms to run")
        parser.add_argument("--baseline", type=str, help="Algorithm for relative comparisons")
        parser.add_argument("--timeout", type=int, default=600,
                            help="Timeout in seconds for single algorithm execution (default: 600)")

        for p_name, p_data in PARAM_CONFIG.items():
            parser.add_argument(f"--{p_name}", type=type(p_data["default"]), default=p_data["default"])

        self.add_custom_args(parser)
        args = parser.parse_args()

        # Configure Active Algorithms
        if args.algorithm:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k in args.algorithm}
        else:
            self.active_algos = {k: v for k, v in ALGORITHMS.items() if k != "local"}

        if args.baseline and args.baseline not in ALGORITHMS:
            print(f"[!] The specified baseline '{args.baseline}' is not in the active algorithms list.")
            exit(1)

        return args

    def print_parameters(self) -> None:
        general_args = []
        for k, v in vars(self.args).items():
            if k not in PARAM_CONFIG:
                display_v = ", ".join(map(str, v)) if isinstance(v, list) else v
                general_args.append([k, display_v])

        self.logger.info("\n[*] General Parameters:")
        self.logger.info(tabulate(general_args, headers=["Argument", "Value"], tablefmt="simple"))

        for algo_name, algo_config in self.active_algos.items():
            template = algo_config.get('template', [])
            params = algo_config.get('params', {})

            if not template:
                self.logger.info(f"\n[*] Hyperparameters: [{algo_name}] -> (None required)")
                continue

            algo_data = []
            for p_key in template:
                if p_key in params:
                    algo_data.append([p_key, params[p_key], "FIXED"])
                else:
                    base_val = getattr(self.args, p_key, "N/A")
                    display_val = self.get_algo_param_display(p_key, base_val)
                    algo_data.append([p_key, display_val, "DYNAMIC"])

            self.logger.info(f"[*] Hyperparameters: [{algo_name}]")
            self.logger.info(tabulate(algo_data, headers=["Param", "Value", "State"], tablefmt="simple"))

        self.logger.info(f"\n[*] Datasets to Run ({len(self.datasets_to_run)}):")
        dataset_table = []
        for ds in self.datasets_to_run:
            filename = ds["filename"]
            short_name = ds.get("short_name", "N/A")
            meta = ds.get("meta", {})

            nodes = meta.get("nodes", "N/A")
            edges = meta.get("edges", "N/A")

            disp_nodes = f"{nodes:,}" if isinstance(nodes, int) else nodes
            disp_edges = f"{edges:,}" if isinstance(edges, int) else edges

            dataset_table.append([
                short_name,
                filename,
                meta.get("size", "N/A"),
                disp_nodes,
                disp_edges,
                meta.get("avg_degree", "N/A"),
            ])

        self.logger.info(
            tabulate(dataset_table, headers=["ID", "Dataset", "Size", "Nodes", "Edges", "Avg Deg"],
                     tablefmt="simple"))

    def execute_runner(self, algo_name: str, algo_config: dict, dataset_path: str, dataset_name: str,
                       resolved_params: dict):
        """A helper method to standardize runner execution across subclasses."""
        runner = get_runner(algo_name, self.logger, str(self.session_dir))

        if not runner.binary_exists():
            self.logger.warning(f"[!] Binary not found for {algo_name}. Skipping.")
            return None, None, None, None

        template = algo_config.get('template', [])

        return runner.run_multiple(
            dataset_path=dataset_path,
            base_output_name=f"{algo_name}_{dataset_name}_{self.timestamp}",
            runs=self.args.runs,
            parameters=resolved_params,
            template=template,
            timeout=self.args.timeout
        )

    def setup(self) -> None:
        self.datasets_to_run = get_datasets_to_run(self.args)
        setup_directories()

        self.logger.info(f"[*] Output Directory: {self.session_dir}")
        self.logger.info("[*] Compiling Configured Algorithms:")

        for algo_name, config in self.active_algos.items():
            runner = get_runner(algo_name, self.logger, str(self.session_dir))
            runner.build()

    def get_session_name(self) -> str:
        """Hook to allow subclasses to name the folder (e.g., sweep_c_2026...)"""
        return f"run_{self.timestamp}"

    def get_algo_param_display(self, p_key: str, default_val: Any) -> str:
        """Hook method. Allows subclasses to override parameter display formatting."""
        return str(default_val)

    @abstractmethod
    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def process(self, dataset_path: str, dataset_name: str) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass

    @abstractmethod
    def print_table(self) -> None:
        pass
