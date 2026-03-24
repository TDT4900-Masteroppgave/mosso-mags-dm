import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from scripts.config import PARAM_CONFIG
from scripts.utils import format_dataframe_with_baseline
from scripts.plotter import plot_results, plot_runs_variance
from scripts.benchmark import Benchmark


class CompareBenchmark(Benchmark):
    def __init__(self):
        super().__init__("compare")
        self.all_times_dict, self.all_ratios_dict = {}, {}

    def add_custom_args(self, parser):
        parser.add_argument("--keep-summaries", action="store_true")

    def process(self, dataset_path: str, dataset_name: str):
        current_result = {"Dataset": dataset_name}

        for algo_name, algo_config in self.active_algos.items():
            resolved_params = {}
            params = algo_config.get('params', {})
            for p_key in PARAM_CONFIG.keys():
                resolved_params[p_key] = params.get(p_key, getattr(self.args, p_key))

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                algo_config=algo_config,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                resolved_params=resolved_params
            )

            if t_avg is not None:
                t_std = np.std(t_list) if len(t_list) > 1 else 0.0
                r_std = np.std(r_list) if len(r_list) > 1 else 0.0

                current_result[f"Time_{algo_name}"] = t_avg
                current_result[f"Ratio_{algo_name}"] = r_avg
                current_result[f"Time_std_{algo_name}"] = t_std
                current_result[f"Ratio_std_{algo_name}"] = r_std

                # Update the console logger to show the ± symbol
                if self.args.runs > 1:
                    self.logger.info(f"\t=> {algo_name: <12} Time: {t_avg:.3f}s ± {t_std:.3f}s | Ratio: {r_avg:.5f} ± {r_std:.5f}")
                    self.all_times_dict[algo_name], self.all_ratios_dict[algo_name] = t_list, r_list
                else:
                    self.logger.info(f"\t=> {algo_name: <12} Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}")

        self.results.append(current_result)

    def print_table(self):
        df = pd.DataFrame(self.results)
        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]

        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row['Dataset'] = 'AVERAGE'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        display_df = format_dataframe_with_baseline(df, strategies, self.args.baseline)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
        for line in table_str.split('\n'):
            self.logger.info(line)

    def finalize(self):
        csv_file = os.path.join(self.session_dir, "results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)

        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]
        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row['Dataset'] = 'AVERAGE'
        table_df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        display_df = format_dataframe_with_baseline(table_df, strategies, self.args.baseline)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)

        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_str)

        plot_results(csv_file, os.path.join(self.session_dir, "compare_plot.pdf"), self.logger)
        if self.args.runs > 1:
            plot_runs_variance("runs_variance_plot", self.all_times_dict, self.all_ratios_dict, self.session_dir)


if __name__ == "__main__":
    CompareBenchmark().run()
