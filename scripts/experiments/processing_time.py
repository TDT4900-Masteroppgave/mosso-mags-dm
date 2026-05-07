import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS, ALGORITHMS
from scripts.experiments.base_experiment import Experiment


class ProcessingTime(Experiment):
    def __init__(self):
        super().__init__("processing_speed")

    def process(self) -> list[dict]:
        metrics = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)
            total_edges = DATASETS.get(short_name, {}).get("meta", {}).get("edges", 1)

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)
                res = self.execute_runner(algo_name, short_name, params)

                if res is None:
                    continue

                _, _, t_list, r_list = res
                is_inc = ALGORITHMS.get(algo_name, {}).get("type", "") == "mosso"

                for run, (t, r) in enumerate(zip(t_list, r_list)):
                    t_micros = t * 1_000_000
                    final_t_micros = (t_micros / total_edges) if is_inc else t_micros

                    metrics.append({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "run": run + 1,
                        "time": t,
                        "ratio": r,
                        "time_micros": final_t_micros,
                        **params
                    })

        return metrics

    def output(self, df: pd.DataFrame):
        table = Table(title="Incremental vs Batch Execution Time", box=box.SIMPLE, show_header=True,
                      header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Execution Time (µs)", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm'], as_index=False)[['time_micros']].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm"])

        for _, row in summary_df.iterrows():
            t_val = f"{row['time_micros']:,.2f} µs" if pd.notna(row['time_micros']) else "N/A"
            table.add_row(str(row['dataset']), str(row['algorithm']), t_val)

        self.logger.print(table)


def main():
    with ProcessingTime() as exp:
        exp.run()

if __name__ == "__main__":
    main()
