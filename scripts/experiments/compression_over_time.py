import re
import json
from pathlib import Path

from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment
from scripts.datasets import create_partial_dataset


class CompressionOverTime(Experiment):
    DEFAULT_CHECKPOINTS = [0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self):
        super().__init__("compression_time")

        # Regex to catch the interval outputs from the Mosso Java/C++ binaries
        self.log_pattern = re.compile(r"(\d+)\s*:\s*Elapsed time\s*:\s*([\d.]+)\s*:\s*ratio\s*:\s*([\d.]+)")

    def add_custom_args(self, parser):
        parser.add_argument(
            "--checkpoints", nargs="+", type=float,
            default=self.DEFAULT_CHECKPOINTS,
            help="Edge stream fractions to evaluate batch algorithms at (default: 0.2 0.4 0.6 0.8 1.0)",
        )

    def process(self, dataset_path: str, dataset_short_name: str) -> None:
        total_edges = DATASETS.get(dataset_short_name, {}).get("meta", {}).get("edges", 1)
        checkpoints = sorted(self.args.checkpoints)

        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)
            algo_type = algo_config.get("type", "mosso")

            if algo_type == "mags":
                self.logger.info(f"\t[*] Running Batch Algorithm: {algo_name}")

                for cp in checkpoints:
                    n_edges = int(total_edges * cp)

                    # Get or create the partial dataset (100% uses original file)
                    if cp >= 1.0:
                        partial_path = dataset_path
                    else:
                        partial_path = create_partial_dataset(dataset_path, cp, total_edges, self.logger)

                    # 1. Get raw data from base class
                    # Note: We pass a suffixed name (e.g., "YT_p20") so log files don't overwrite each other
                    raw_runs = self.execute_runner(algo_name, partial_path, f"{dataset_short_name}_p{int(cp * 100)}", params)
                    if not raw_runs:
                        continue

                    # 2. Decorate with checkpoint metadata
                    for row in raw_runs:
                        row["dataset"] = dataset_short_name  # Revert the suffix so it matches in DB
                        row["edges_processed"] = n_edges
                        row["fraction"] = cp
                        self.results.append(row)

            else:
                self.logger.info(f"\t[*] Running Incremental Algorithm: {algo_name}")

                # 1. Run once on 100% dataset
                raw_runs = self.execute_runner(algo_name, dataset_path, dataset_short_name, params)
                if not raw_runs:
                    continue

                # 2. Extract the intermediate points by parsing the logs
                for run_idx in range(1, self.args.runs + 1):
                    log_file = self.session_dir / "run_log" / f"{algo_name}_{dataset_short_name}_{self.timestamp}_run{run_idx}.log"

                    if not log_file.exists():
                        self.logger.warning(f"Could not find log file: {log_file}")
                        continue

                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            match = self.log_pattern.search(line)
                            if match:
                                edges_processed = int(match.group(1))

                                # Append intermediate interval point
                                self.results.append({
                                    "dataset": dataset_short_name,
                                    "algorithm": algo_name,
                                    "run": run_idx,
                                    "time": float(match.group(2)),
                                    "ratio": float(match.group(3)),
                                    "edges_processed": edges_processed,
                                    "fraction": edges_processed / total_edges,
                                    "parameters": json.dumps(params)
                                })

                # 3. Append the statistically complete final run at the 100% mark
                for row in raw_runs:
                    row["edges_processed"] = total_edges
                    row["fraction"] = 1.0
                    self.results.append(row)

def main():
    CompressionOverTime().run()


if __name__ == "__main__":
    CompressionOverTime().run()