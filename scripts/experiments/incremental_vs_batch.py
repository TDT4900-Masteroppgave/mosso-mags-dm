from scripts.config import DATASETS, ALGORITHMS
from scripts.experiments.base_experiment import Experiment

import pandas as pd
from rich.table import Table
from rich import box


class IncrementalVsBatch(Experiment):
    def __init__(self):
        super().__init__("ivb")

    def process(self) -> list[dict]:
        metrics: list[dict] = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)

            meta = DATASETS.get(short_name, {}).get("meta", {})
            total_edges = meta.get("edges", 1)

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)

                t_avg, r_avg, t_list, r_list = self.execute_runner(
                    algo_name=algo_name,
                    dataset_path=dataset_path,
                    params=params,
                    dataset_short_name=short_name,
                )

                if t_avg is None or r_avg is None:
                    continue

                for run, (t, r) in enumerate(zip(t_list, r_list)):
                    is_incremental = ALGORITHMS.get(algo_name, {}).get("type", "") == "mosso"
                    t_micros = t * 1_000_000
                    final_t_micros = (t_micros / total_edges) if is_incremental else t_micros

                    metrics.append({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "run": run + 1,
                        # Plot this column for Batch Algorithms
                        "time": t,
                        # Plot this column for Streaming Algorithms
                        "time_micros": final_t_micros,
                        "ratio": r,
                        **params,
                    })

        return metrics

    def output(self, df: pd.DataFrame):
        summary_df = df.groupby(['dataset', 'algorithm'], as_index=False)[['time_micros']].mean()

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

        self.logger.print(table)

def main():
    with IncrementalVsBatch() as exp:
        exp.run()


if __name__ == "__main__":
    main()