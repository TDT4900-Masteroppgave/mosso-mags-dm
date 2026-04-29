import json

from scripts.config import DATASETS, ALGORITHMS
from scripts.experiments.base_experiment import Experiment

import pandas as pd
from rich.table import Table
from rich import box
from scripts import db


class IncrementalVsBatch(Experiment):
    def __init__(self):
        super().__init__("ivb")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics: list[dict] = []

        meta = DATASETS.get(dataset_short_name, {}).get("meta", {})
        total_edges = meta.get("edges", 1)

        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                params=params,
                dataset_short_name=dataset_short_name,
            )

            if t_list is None or r_list is None:
                continue

            for i, (t, r) in enumerate(zip(t_list, r_list)):
                is_incremental = ALGORITHMS.get(algo_name, {}).get("type", "") == "mosso"

                t_micros = t * 1_000_000

                final_t_micros = (t_micros / total_edges) if is_incremental else t_micros

                metrics.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    # Plot this column for Batch Algorithms
                    "time": t,
                    # Plot this column for Streaming Algorithms
                    "time_micros": final_t_micros,
                    "ratio": r,
                    **params,
                })

        return metrics

    def output(self):
        if not self.db_conn:
            return

        raw_df = db.read_results(self.db_conn)
        if raw_df.empty:
            return

        # Use the new pre-calculated column
        summary_df = raw_df.groupby(['dataset', 'algorithm'], as_index=False)[['time_micros']].mean()

        table = Table(title="Incremental vs Batch Execution Time", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Type", justify="center")
        table.add_column("Execution Time (µs)", justify="right")

        for _, row in summary_df.sort_values(by=["dataset", "algorithm"]).iterrows():
            t_val = f"{row['time_micros']:,.2f} µs" if pd.notna(row['time_micros']) else "N/A"

            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                t_val
            )

        self.console.print(table)

def main():
    IncrementalVsBatch().run()


if __name__ == "__main__":
    main()