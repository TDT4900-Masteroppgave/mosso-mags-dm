import pandas as pd
from rich import box
from rich.table import Table
from scipy.stats import gmean

from scripts.experiments.base_experiment import Experiment

class Benchmark(Experiment):
    def __init__(self):
        super().__init__("benchmark")
        if self.args.baseline and self.args.baseline not in self.args.algorithm:
            self.logger.print(f"[bold red]Error: Baseline '{self.args.baseline}' must be in algorithms list.[/]")

    def add_custom_args(self, parser):
        parser.add_argument("--baseline", type=str)

    def process(self) -> None:
        for i, short_name in enumerate(self.datasets.keys(), 1):
            self._print_status(i, short_name)

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)

                algo_type = algo_config.get("type", None)
                dataset_path = self._get_dataset_path(short_name, algo_type)
                if not dataset_path:
                    continue

                res = self.execute_runner(dataset_path, short_name, algo_name,  params)
                if res is None:
                    continue

                _, _, t_list, r_list, _ = res

                for run, (t, r) in enumerate(zip(t_list, r_list)):
                    self.record_result({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "run": run + 1,
                        "time": t,
                        "ratio": r,
                        **params
                    })

    def output(self, df: pd.DataFrame):
        # Calculate Geometric Mean instead of arithmetic mean
        global_avg = df.groupby('algorithm', as_index=False)[['time', 'ratio']].agg(
            lambda x: gmean(x.dropna()) if not x.dropna().empty else float('nan')
        )

        baseline_vals = {}
        if self.args.baseline and self.args.baseline in global_avg['algorithm'].values:
            baseline_vals = global_avg[global_avg['algorithm'] == self.args.baseline].iloc[0].to_dict()

        # Updated titles and headers for GeoMean
        table = Table(title="Geometric Mean Across All Datasets", box=box.SIMPLE, show_header=True, header_style="bold green")
        table.add_column("Algorithm")
        table.add_column("GeoMean Time")
        table.add_column("GeoMean Ratio")

        for _, row in global_avg.sort_values(by="algorithm").iterrows():
            algo = row['algorithm']
            t_val = row['time']
            r_val = row['ratio']

            t_str = f"{t_val:.3f}s" if pd.notnull(t_val) else "-"
            r_str = f"{r_val:.5f}" if pd.notnull(r_val) else "-"

            if baseline_vals and algo != self.args.baseline:
                base_t = baseline_vals.get('time', 0)
                base_r = baseline_vals.get('ratio', 0)

                if pd.notnull(t_val) and t_val > 0 and pd.notnull(base_t) and base_t > 0:
                    t_str += f" [green]({base_t / t_val:.2f}x)[/green]"
                if pd.notnull(r_val) and r_val > 0 and pd.notnull(base_r) and base_r > 0:
                    r_str += f" [green]({r_val / base_r:.2f}x)[/green]"

            table.add_row(algo, t_str, r_str)

        self.logger.print(table)

def main():
    with Benchmark() as exp:
        exp.run()

if __name__ == "__main__":
    main()