import pandas as pd
from rich import box
from rich.table import Table

from scripts.experiments.base_experiment import Experiment


class Benchmark(Experiment):
    def __init__(self):
        super().__init__("benchmark")

        if self.args.baseline and self.args.baseline not in self.args.algorithm:
            self.logger.print(
                f"[bold red]Error: Baseline '{self.args.baseline}' must be included in algorithms list: {self.args.algorithm}[/]")

    def add_custom_args(self, parser):
        parser.add_argument("--baseline", type=str, help="Algorithm for relative comparisons")

    def process(self) -> list[dict]:
        metrics: list[dict] = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)

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
                    metrics.append({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "run": run + 1,
                        "time": t,
                        "ratio": r,
                        **params,
                    })
        return metrics

    def output(self, df: pd.DataFrame):
        baseline_algo = getattr(self.args, 'baseline', None)

        global_avg_df = df.groupby('algorithm', as_index=False)[['time', 'ratio']].mean()

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

        self.logger.print(avg_table)


def main():
    with Benchmark() as exp:
        exp.run()


if __name__ == "__main__":
    main()
