import os
import pandas as pd
from tabulate import tabulate
from scipy.stats import qmc

from scripts.config import PARAM_CONFIG
from scripts.plotter import plot_pareto_front, get_pareto_front_2d
from scripts.benchmark import Benchmark

class LHSBenchmark(Benchmark):
    def __init__(self):
        super().__init__("lhs")
        # Dictionary to store the fixed LHS samples per algorithm
        self.algo_samples = {}

    def add_custom_args(self, parser):
        parser.add_argument("--samples", type=int, default=30, help="Number of LHS configurations to test")

    def get_algo_param_display(self, p_key, default_val):
        bounds = PARAM_CONFIG.get(p_key, {}).get('bounds')
        return f"BOUNDS {bounds}" if bounds else "FIXED"

    def generate_lhs_samples(self, algo_template):
        """Generates scaled LHS configurations for the active template."""
        if not algo_template:
            return []

        sampler = qmc.LatinHypercube(d=len(algo_template))
        sample = sampler.random(n=self.args.samples)

        configs = []
        for row in sample:
            cfg = {}
            for i, p_key in enumerate(algo_template):
                bounds = PARAM_CONFIG.get(p_key, {}).get("bounds")
                if bounds:
                    l_bound, u_bound = bounds
                    val = int(l_bound + row[i] * (u_bound - l_bound))
                    cfg[p_key] = val
                else:
                    cfg[p_key] = getattr(self.args, p_key)
            configs.append(cfg)
        return configs

    def process(self, dataset_path: str, dataset_name: str):

        for algo_name, algo_config in self.active_algos.items():
            template = algo_config.get('template', [])
            base_params = algo_config.get('params', {})

            # Generate the samples ONCE per algorithm and cache them
            if algo_name not in self.algo_samples:
                self.algo_samples[algo_name] = self.generate_lhs_samples(template) if template else [{}]

            # Retrieve the fixed configurations for this algorithm
            lhs_configs = self.algo_samples[algo_name]
            total_runs = len(lhs_configs)

            for run_idx, lhs_params in enumerate(lhs_configs, 1):
                resolved_params = {}
                for p_key in PARAM_CONFIG.keys():
                    resolved_params[p_key] = getattr(self.args, p_key)
                resolved_params.update(base_params)
                resolved_params.update(lhs_params)

                if template:
                    self.logger.info(f"\t[{algo_name}] Testing Config {run_idx}/{total_runs}: {lhs_params}")
                else:
                    self.logger.info(f"\t[{algo_name}] Testing Default Config (No template defined).")

                unique_run_name = f"{dataset_name}_lhs{run_idx}"

                t, r, _, _ = self.execute_runner(
                    algo_name=algo_name,
                    algo_config=algo_config,
                    dataset_path=dataset_path,
                    dataset_name=unique_run_name,
                    resolved_params=resolved_params
                )

                if t is not None:
                    res = {
                        "Dataset": dataset_name,
                        "Algorithm": algo_name,
                        "Time": t,
                        "Ratio": r
                    }
                    for p_key in template:
                        res[p_key] = resolved_params[p_key]

                    self.results.append(res)

    def _get_averaged_dataframe(self):
        """Helper to group results by Algorithm and Parameters, averaging Time and Ratio."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # We want to group by everything EXCEPT Dataset, Time, and Ratio
        group_cols = [col for col in df.columns if col not in ['Dataset', 'Time', 'Ratio']]

        # Calculate the mean Time and Ratio across all datasets for each specific config
        avg_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()

        # Add a placeholder dataset name so the plotter knows they belong to one global plot
        avg_df['Dataset'] = 'AVERAGE_ACROSS_DATASETS'
        return avg_df

    def print_table(self):
        raw_df = pd.DataFrame(self.results)

        for dataset_name, group_df in raw_df.groupby('Dataset'):
            pareto_df = get_pareto_front_2d(group_df, 'Time', 'Ratio')
            pareto_df = pareto_df.sort_values(by="Time", ascending=True)

            self.logger.info(f"\n--- PARETO OPTIMAL CONFIGURATIONS: {dataset_name} ---")
            table_str = tabulate(pareto_df, headers='keys', tablefmt='grid', showindex=False)
            for line in table_str.split('\n'):
                self.logger.info(line)

        avg_df = self._get_averaged_dataframe()
        if not avg_df.empty:
            pareto_df_avg = get_pareto_front_2d(avg_df, 'Time', 'Ratio')
            pareto_df_avg = pareto_df_avg.sort_values(by="Time", ascending=True)

            # Drop the placeholder column just for a cleaner console print
            if 'Dataset' in pareto_df_avg.columns:
                pareto_df_avg = pareto_df_avg.drop(columns=['Dataset'])

            self.logger.info("\n--- GLOBAL PARETO OPTIMAL CONFIGURATIONS (LHS AVERAGED) ---")
            table_str = tabulate(pareto_df_avg, headers='keys', tablefmt='grid', showindex=False)
            for line in table_str.split('\n'):
                self.logger.info(line)

    def finalize(self):
        raw_df = pd.DataFrame(self.results)
        avg_df = self._get_averaged_dataframe()

        raw_csv = os.path.join(self.session_dir, "lhs_raw_results.csv")
        raw_df.to_csv(raw_csv, index=False)

        avg_csv = os.path.join(self.session_dir, "lhs_averaged_results.csv")
        avg_df.to_csv(avg_csv, index=False)

        table_output = ""
        for dataset_name, group_df in raw_df.groupby('Dataset'):
            pareto_df = get_pareto_front_2d(group_df, 'Time', 'Ratio')
            pareto_df = pareto_df.sort_values(by="Time", ascending=True)
            table_output += f"\n--- PARETO OPTIMAL CONFIGURATIONS: {dataset_name} ---\n"
            table_output += tabulate(pareto_df, headers='keys', tablefmt='grid', showindex=False) + "\n"

        if not avg_df.empty:
            pareto_df_avg = get_pareto_front_2d(avg_df, 'Time', 'Ratio')
            pareto_df_avg = pareto_df_avg.sort_values(by="Time", ascending=True)
            if 'Dataset' in pareto_df_avg.columns:
                pareto_df_avg = pareto_df_avg.drop(columns=['Dataset'])
            table_output += "\n--- GLOBAL PARETO OPTIMAL CONFIGURATIONS (LHS AVERAGED) ---\n"
            table_output += tabulate(pareto_df_avg, headers='keys', tablefmt='grid', showindex=False) + "\n"

        # Save to text file
        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_output)

        combined_df = pd.concat([raw_df, avg_df], ignore_index=True)
        combined_csv = os.path.join(self.session_dir, "lhs_combined_for_plot.csv")
        combined_df.to_csv(combined_csv, index=False)

        plot_pareto_front(combined_csv, os.path.join(self.session_dir, "lhs_optimization_plot.pdf"))

if __name__ == "__main__":
    LHSBenchmark().run()