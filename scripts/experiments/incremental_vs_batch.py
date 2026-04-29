import json

from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment


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
                t_micros = t * 1_000_000

                metrics.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    # Plot this column for Batch Algorithms
                    "time_micros": t_micros,
                    # Plot this column for Streaming Algorithms
                    "time_per_change": t_micros / total_edges,
                    "ratio": r,
                    "parameters": json.dumps(params),
                })

        return metrics

    def output(self):
        if not self.db_conn:
            return

        import pandas as pd
        from rich.table import Table
        from rich import box
        from scripts import db

        raw_df = db.read_results(self.db_conn)
        if raw_df.empty:
            return

        summary_df = raw_df.groupby(['dataset', 'algorithm'], as_index=False)[['time_micros', 'time_per_change']].mean()

        table = Table(title="Incremental vs Batch Execution Time", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Total Time (Batch)", justify="right")
        table.add_column("Time Per Change (Streaming)", justify="right")

        for _, row in summary_df.sort_values(by=["dataset", "algorithm"]).iterrows():
            t_batch = f"{row['time_micros']:,.0f} µs" if pd.notna(row['time_micros']) else "N/A"
            t_stream = f"{row['time_per_change']:,.2f} µs" if pd.notna(row['time_per_change']) else "N/A"

            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                t_batch,
                t_stream
            )

        self.console.print(table)

def main():
    IncrementalVsBatch().run()


if __name__ == "__main__":
    main()