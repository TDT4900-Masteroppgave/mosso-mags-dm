from rich import json

from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment


class ParameterSweep(Experiment):
    def __init__(self):
        super().__init__("sweep")
        self.all_times_dict, self.all_ratios_dict = {}, {}
        config = PARAM_CONFIG[self.args.param]
        self.sweep_values = list(range(*self.args.range)) if self.args.range else (
            self.args.values if self.args.values else config["bounds"])

    def add_custom_args(self, parser):
        parser.add_argument("--param", choices=list(PARAM_CONFIG.keys()), required=True)
        parser.add_argument("--range", type=int, nargs=3, required=True)

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics : list[dict] = []
        for val in self.sweep_values:
            self.logger.info(f"{self.args.param} = {val}:")
            for algo_name, algo_config in self.active_algos.items():
                if self.args.param not in algo_config.get('template', {}):
                    self.logger.info(f"=> Skipping {algo_name}: No such parameter {self.args.param}")
                    continue

                params = self._resolve_algo_params(algo_config)
                params.update({self.args.param: str(val)})

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
            return

        import pandas as pd
        from rich.table import Table
        from rich import box
        from scripts import db

        raw_df = db.read_results(self.db_conn)
        if raw_df.empty:
            return

        # Group by dataset, algorithm, and the parameter we swept over
        param = self.args.param
        summary_df = raw_df.groupby(['dataset', 'algorithm', param], as_index=False)[['time', 'ratio']].mean()

        table = Table(title=f"Parameter Sweep Summary: {param.upper()}", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column(f"Param: {param.upper()}", style="magenta", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Avg Ratio", justify="right")

        for _, row in summary_df.sort_values(by=["dataset", "algorithm", param]).iterrows():
            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                str(row[param]),
                f"{row['time']:.3f}s",
                f"{row['ratio']:.5f}"
            )

        self.console.print(table)

def main():
    ParameterSweep().run()


if __name__ == "__main__":
    ParameterSweep().run()
