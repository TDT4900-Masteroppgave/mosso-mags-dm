import os
import pandas as pd
from tabulate import tabulate

from scripts.config import PARAM_CONFIG
from scripts.utils import format_dataframe_with_baseline
from scripts.plotter import plot_parameter_analysis
from scripts.benchmark import Benchmark


class ParameterSweepBenchmark(Benchmark):
    def __init__(self):
        super().__init__("sweep")
        config = PARAM_CONFIG[self.args.param]
        self.sweep_values = list(range(*self.args.range)) if self.args.range else (
            self.args.values if self.args.values else config["values"])

    def get_session_name(self):
        # This names the folder dynamically: e.g., output/benchmarks/sweep/sweep_c_2026...
        return f"sweep_{self.args.param}_{self.timestamp}"

    def add_custom_args(self, parser):
        parser.add_argument("--param", choices=list(PARAM_CONFIG.keys()), required=True)
        parser.add_argument("--range", type=int, nargs=3)

    def get_algo_param_display(self, p_key, default_val):
        if self.args.param == p_key:
            range_str = self.args.range if self.args.range else self.args.values
            return f"SWEEPING {range_str}"
        return default_val

    def process(self, dataset_path: str, dataset_name: str):
        for val in self.sweep_values:
            self.logger.info(f"--- Testing {self.args.param.upper()} = {val} ---")
            current_result = {"Dataset": dataset_name, self.args.param: val}

            for algo_name, algo_config in self.active_algos.items():
                params = algo_config.get('params', {})
                resolved_params = {
                    "interval": params.get('interval', self.args.interval)
                }
                for p_key in PARAM_CONFIG.keys():
                    current_fallback = val if self.args.param == p_key else getattr(self.args, p_key)
                    resolved_params[p_key] = params.get(p_key, current_fallback)

                t, r, _, _ = self.execute_runner(
                    algo_name=algo_name,
                    algo_config=algo_config,
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    resolved_params=resolved_params
                )
                if t is not None:
                    current_result[f"Time_{algo_name}"], current_result[f"Ratio_{algo_name}"] = t, r
                    self.logger.info(f"\t=> {algo_name: <12} Time: {t:.3f}s | Ratio: {r:.5f}")

            self.results.append(current_result)

    def print_table(self):
        df = pd.DataFrame(self.results)
        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]

        self.logger.info(f"--- SWEEP LOG ({self.args.param.upper()}) ---")
        display_df = format_dataframe_with_baseline(df, strategies, self.args.baseline)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
        for line in table_str.split('\n'):
            self.logger.info(line)

        self.logger.info(f"--- AVERAGES BY {self.args.param.upper()} ---")
        avg_df = df.groupby(self.args.param).mean(numeric_only=True).reset_index()
        display_avg = format_dataframe_with_baseline(avg_df, strategies, self.args.baseline)
        table_str = tabulate(display_avg, headers='keys', tablefmt='grid', showindex=False)
        for line in table_str.split('\n'):
            self.logger.info(line)

    def finalize(self):
        master_csv = os.path.join(self.session_dir, "results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(master_csv, index=False)

        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]

        table_str = f"--- SWEEP LOG ({self.args.param.upper()}) ---\n"
        display_df = format_dataframe_with_baseline(df, strategies, self.args.baseline)
        table_str += tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)

        avg_df = df.groupby(self.args.param).mean(numeric_only=True).reset_index()
        display_avg = format_dataframe_with_baseline(avg_df, strategies, self.args.baseline)
        table_str += f"\n\n--- AVERAGES BY {self.args.param.upper()} ---\n"
        table_str += tabulate(display_avg, headers='keys', tablefmt='grid', showindex=False)

        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_str)

        plot_parameter_analysis(master_csv, self.args.param, os.path.join(self.session_dir, "parameter_plot.pdf"))


if __name__ == "__main__":
    ParameterSweepBenchmark().run()
