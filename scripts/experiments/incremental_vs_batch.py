"""
Fair comparison of incremental (streaming) vs batch graph summarization algorithms.

Methodology based on Ko et al., "Incremental Lossless Graph Summarization" (KDD 2020):

  Figure 5 (Compression quality over time):
  - Splits edge streams into checkpoints (e.g., 20%, 40%, 60%, 80%, 100%)
  - At each checkpoint, runs ALL algorithms on the partial graph (or parses logs for streaming algos)
  - Compares compression ratio evolution

  Figure 4 (Speed - update cost):
  - Incremental update cost: time to process ONLY the new edges since last checkpoint
    (derived as time(cp) - time(prev_cp), since MoSSo streams edges sequentially)
  - Batch re-run cost: time to run from scratch on the full snapshot at each checkpoint
  - This shows MoSSo's real advantage: near-constant update cost vs growing re-run cost
"""

import re

import pandas as pd

from scripts.experiments.base_experiment import Experiment
from scripts.config import ALGORITHMS
from scripts.datasets import create_partial_dataset
import scripts.db as db


class IncrementalVsBatch(Experiment):

    DEFAULT_CHECKPOINTS = [0.2, 0.4, 0.6, 0.8, 1.0]
    _MOSSO_LOG_PATTERN = re.compile(r"(\d+)\s*:\s*Elapsed time\s*:\s*([\d.]+)\s*:\s*ratio\s*:\s*([\d.]+)")

    def __init__(self):
        super().__init__("incremental_vs_batch")

    def add_custom_args(self, parser):
        parser.add_argument(
            "--checkpoints", nargs="+", type=float,
            default=self.DEFAULT_CHECKPOINTS,
            help="Edge stream fractions to evaluate at (default: 0.2 0.4 0.6 0.8 1.0)",
        )

    def _create_partial_dataset(self, dataset_path: str, fraction: float, total_edges: int) -> str:
        return create_partial_dataset(dataset_path, fraction, total_edges, self.logger)

    def _parse_mosso_stdout(self, stdout: str):
        """Parses the interval logs from MoSSo stdout to extract time and ratio."""
        parsed = {}
        for line in stdout.splitlines():
            match = self._MOSSO_LOG_PATTERN.search(line)
            if match:
                edges = int(match.group(1))
                cum_time = float(match.group(2))
                ratio = float(match.group(3))
                parsed[edges] = {"cum_time": cum_time, "ratio": ratio}
        return parsed

    def process(self, dataset_path: str, ds: dict, dataset_name: str):
        meta = ds.get("meta", dataset_name)
        total_edges = meta["edges"]
        self.logger.info(f"\t[*] Total edges in stream: {total_edges:,}")

        checkpoints = sorted(self.args.checkpoints)

        # Initialize dictionary to hold row data for each checkpoint
        checkpoint_rows = {
            cp: {
                "Dataset": dataset_name,
                "Checkpoint": cp,
                "Edges": int(total_edges * cp),
            }
            for cp in checkpoints
        }

        for algo_name, algo_config in self.active_algos.items():
            resolved_params = self._resolve_algo_params(algo_config)

            is_batch = self._algo_type(algo_name) == "mags"

            if is_batch:
                # MAGS (Batch Algorithm): Needs to be run from scratch on partial datasets
                self.logger.info(f"\t[*] Running Batch Algorithm: {algo_name}")
                for cp in checkpoints:
                    n_edges = int(total_edges * cp)

                    if cp >= 1.0:
                        partial_path = dataset_path
                    else:
                        partial_path = self._create_partial_dataset(dataset_path, cp, total_edges)

                    t_avg, r_avg, _, _ = self.execute_runner(
                        algo_name=algo_name,
                        algo_config=algo_config,
                        dataset_path=partial_path,
                        dataset_name=f"{dataset_name}_p{int(cp * 100)}",
                        resolved_params=resolved_params,
                    )

                    checkpoint_rows[cp][f"Time_{algo_name}"] = t_avg
                    checkpoint_rows[cp][f"Ratio_{algo_name}"] = r_avg

                    if t_avg is not None:
                        self.logger.info(
                            f"\t=> {algo_name:<12} [CP {cp:.0%}] Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}"
                        )
            else:
                # MoSSo (Incremental Algorithm): Run once on 100% dataset, read log files
                self.logger.info(f"\t[*] Running Incremental Algorithm: {algo_name}")

                # execute_runner returns times/ratios, not the raw stdout strings
                t_avg, r_avg, _, _ = self.execute_runner(
                    algo_name=algo_name,
                    algo_config=algo_config,
                    dataset_path=dataset_path,
                    dataset_name=f"{dataset_name}_p100",
                    resolved_params=resolved_params,
                )

                if t_avg is None:
                    continue  # Execution failed

                # Find log files on disk
                base_name = f"{algo_name}_{dataset_name}_p100_{self.timestamp}"
                log_files = list(self.session_dir.rglob(f"{base_name}_run*.log"))

                if not log_files:
                    self.logger.warning(f"\t[!] Could not find log files for {base_name} to parse checkpoints.")
                    continue

                # Read all runs to average them out
                stdouts = []
                for log_file in log_files:
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            stdouts.append(f.read())
                    except Exception as e:
                        self.logger.error(f"\t[!] Failed to read log file {log_file}: {e}")

                # Aggregate parsed data across all runs to calculate averages
                aggregated_data = {}
                for out in stdouts:
                    run_data = self._parse_mosso_stdout(out)
                    for edges, metrics in run_data.items():
                        if edges not in aggregated_data:
                            aggregated_data[edges] = {"cum_time": [], "ratio": []}
                        aggregated_data[edges]["cum_time"].append(metrics["cum_time"])
                        aggregated_data[edges]["ratio"].append(metrics["ratio"])

                final_parsed = {}
                for edges, metrics in aggregated_data.items():
                    final_parsed[edges] = {
                        "cum_time": sum(metrics["cum_time"]) / len(metrics["cum_time"]),
                        "ratio": sum(metrics["ratio"]) / len(metrics["ratio"]),
                    }

                # Ensure the 100% mark contains the exact final time/ratio
                if total_edges not in final_parsed:
                    final_parsed[total_edges] = {"cum_time": t_avg, "ratio": r_avg}

                # Map extracted checkpoint data to rows
                for cp in checkpoints:
                    n_edges = int(total_edges * cp)

                    if final_parsed:
                        # Find the interval marker closest to this checkpoint's edge count
                        closest_edge = min(final_parsed.keys(), key=lambda k: abs(k - n_edges))

                        c_time = final_parsed[closest_edge]["cum_time"]
                        c_ratio = final_parsed[closest_edge]["ratio"]

                        checkpoint_rows[cp][f"Time_{algo_name}"] = c_time
                        checkpoint_rows[cp][f"Ratio_{algo_name}"] = c_ratio
                        self.logger.info(
                            f"\t=> {algo_name:<12} [CP {cp:.0%} mapped -> {closest_edge} edges] Time: {c_time:.3f}s | Ratio: {c_ratio:.5f}"
                        )
                    else:
                        checkpoint_rows[cp][f"Time_{algo_name}"] = None
                        checkpoint_rows[cp][f"Ratio_{algo_name}"] = None

        # Append fully populated rows to results sequentially
        for cp in checkpoints:
            self.results.append(checkpoint_rows[cp])

    def _detect_algos(self, df):
        return [c.replace("Time_", "") for c in df.columns if c.startswith("Time_")]

    def _algo_type(self, algo):
        return ALGORITHMS.get(algo, {}).get("type", "mosso")

    def finalize(self):
        df = pd.DataFrame(self.results)
        algos = self._detect_algos(df)

        if self.db_conn:
            for _, row in df.iterrows():
                for algo in algos:
                    t = row.get(f"Time_{algo}")
                    r = row.get(f"Ratio_{algo}")
                    if t is None or pd.isna(t):
                        continue
                    db.write_result(
                        self.db_conn,
                        algorithm=algo, dataset=row["Dataset"],
                        time=t, ratio=r,
                        checkpoint=float(row["Checkpoint"]),
                    )


if __name__ == "__main__":
    IncrementalVsBatch().run()
