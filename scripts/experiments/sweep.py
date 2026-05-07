import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment

class ParameterSweep(Experiment):
    def __init__(self):
        super().__init__("sweep")
        config = PARAM_CONFIG[self.args.param]
        if self.args.range:
            self.sweep_values = list(range(*self.args.range))
        elif self.args.values:
            self.sweep_values = self.args.values
        else:
            self.sweep_values = config["bounds"]

    def add_custom_args(self, parser):
        parser.add_argument("--param", choices=list(PARAM_CONFIG.keys()), required=True)
        parser.add_argument("--range", type=int, nargs=3, required=False)
        parser.add_argument("--values", type=int, nargs="+", required=False)

    def process(self) -> list[dict]:
        metrics = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)

            for val in self.sweep_values:
                self.logger.info(f"{self.args.param} = {val}:")

                for algo_name, algo_config in self.active_algos.items():
                    if self.args.param not in algo_config.get('template', {}):
                        self.logger.info(f"=> Skipping {algo_name}: No such parameter {self.args.param}")
                        continue

                    params = self._resolve_algo_params(algo_config)
                    params[self.args.param] = str(val)

                    res = self.execute_runner(algo_name, short_name, params)
                    if res is None:
                        continue

                    _, _, t_list, r_list = res
                    for run, (t, r) in enumerate(zip(t_list, r_list)):
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "run": run + 1,
                            "time": t,
                            "ratio": r,
                            "param_name": self.args.param,
                            "param": val,
                            **params
                        })
        return metrics

    def output(self, df: pd.DataFrame):
        title = f"Parameter Sweep Summary: {self.args.param.upper()}"
        table = Table(title=title, box=box.SIMPLE, show_header=True, header_style="bold yellow")

        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column(f"Param: {self.args.param.upper()}", style="magenta", justify="right")
        table.add_column("Avg Time", justify="right")
        table.add_column("Avg Ratio", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'param'], as_index=False)[['time', 'ratio']].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm", "param"])

        for _, row in summary_df.iterrows():
            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                str(row['param']),
                f"{row['time']:.3f}s",
                f"{row['ratio']:.5f}"
            )

        self.logger.print(table)

def main():
    with ParameterSweep() as exp:
        exp.run()

if __name__ == "__main__":
    main()