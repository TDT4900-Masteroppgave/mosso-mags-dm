import json

import pandas as pd
from rich import box
from rich.table import Table
from rich.console import Console

from scripts import db
from scripts.experiments.base_experiment import Experiment


class Benchmark(Experiment):
    def __init__(self):
        super().__init__("benchmark")

        if self.args.baseline and self.args.baseline not in self.args.algorithm:
            self.console.print(f"[bold red]Error: Baseline '{self.args.baseline}' must be included in algorithms list: {self.args.algorithm}[/]")

    def add_custom_args(self, parser):
        parser.add_argument("--baseline", type=str, help="Algorithm for relative comparisons")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics: list[dict] = []
        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                params=params,
                dataset_short_name=dataset_short_name,
            )

            for i, (t, r) in enumerate(zip(t_list, r_list)):
                metrics.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    "time": t,
                    "ratio": r,
                    "parameters": json.dumps(params),
                })

        return metrics

    def output(self):
        if not self.db_conn:
            self.logger.warning("No database connection. Skipping table generation.")
            return

        raw_df = db.read_results(self.db_conn)
        if raw_df.empty:
            return

        baseline_algo = getattr(self.args, 'baseline', None)

        global_avg_df = raw_df.groupby('algorithm', as_index=False)[['time', 'ratio']].mean()

        global_baselines = {}
        if baseline_algo and baseline_algo in global_avg_df['algorithm'].values:
            global_baselines = global_avg_df[global_avg_df['algorithm'] == baseline_algo].iloc[0].to_dict()

        def format_global_row(data_entry):
            t, r = data_entry.get('time', 0), data_entry.get('ratio', 0)
            t_str, r_str = f"{t:.3f}s", f"{r:.5f}"

            if global_baselines and data_entry['algorithm'] != baseline_algo:
                if t > 0 and global_baselines.get('time', 0) > 0:
                    t_str += f" [green]({global_baselines['time'] / t:.2f}x)[/green]"
                if global_baselines.get('ratio', 0) > 0:
                    r_str += f" [green]({r / global_baselines['ratio']:.2f}x)[/green]"

            return pd.Series({
                "Algorithm": data_entry['algorithm'],
                "Avg Time": t_str,
                "Avg Ratio": r_str
            })

        global_summary_df = global_avg_df.apply(format_global_row, axis=1).sort_values(by="Algorithm")

        avg_table = Table(title="Average Across All Datasets", box=box.SIMPLE, show_header=True,
                          header_style="bold green")
        for col in global_summary_df.columns:
            avg_table.add_column(str(col))

        for _, row in global_summary_df.iterrows():
            avg_table.add_row(*row.astype(str).tolist())

        self.console.print(avg_table)

        text_console = Console(width=250, color_system=None)
        with text_console.capture() as capture:
            text_console.print(avg_table)

        for line in capture.get().splitlines():
            if line.strip():
                self.logger.debug(line)


def main():
    Benchmark().run()


if __name__ == "__main__":
    main()
