import os
import pandas as pd
from tabulate import tabulate
import optuna
import warnings

from scripts.config import PARAM_CONFIG
from scripts.plotter import get_pareto_front_2d
from scripts.benchmark import Benchmark
from scripts.plotter import plot_pareto_front

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


class BayesianOptimizationBenchmark(Benchmark):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--iterations", type=int, default=30, help="Total number of points to sample")
        parser.add_argument("--n-startup", type=int, default=10, help="Initial random explorations before AI kicks in")
        parser.add_argument("--jobs", type=int, default=1, help="Number of parallel threads to run (-1 uses all CPUs)")

    def process(self, dataset_path: str, dataset_name: str):
        for algo_name, algo_config in self.active_algos.items():
            template = algo_config.get('template', [])
            if not template:
                self.logger.info(f"\n[*] Skipping [{algo_name}]: No hyperparameters to optimize.")
                continue

            def objective(trial):
                resolved_params = {}

                for p_key in template:
                    bounds = PARAM_CONFIG.get(p_key, {}).get("bounds")
                    if bounds:
                        resolved_params[p_key] = trial.suggest_int(p_key, bounds[0], bounds[1])
                    else:
                        resolved_params[p_key] = PARAM_CONFIG.get(p_key, {}).get("default")

                avg_time, avg_ratio, _, _ = self.execute_runner(
                    algo_name, algo_config, dataset_path, dataset_name, resolved_params
                )

                if avg_time is None or avg_ratio is None:
                    raise optuna.exceptions.TrialPruned()

                result_entry = {
                    'Dataset': dataset_name,
                    'Algorithm': algo_name,
                    'Time': avg_time,
                    'Ratio': avg_ratio
                }
                result_entry.update(resolved_params)
                self.results.append(result_entry)

                return avg_time, avg_ratio

            db_path = os.path.join(self.session_dir, "optuna_study.db")
            study_name = f"{algo_name}_{dataset_name}"
            sampler = optuna.samplers.TPESampler(n_startup_trials=self.args.n_startup)

            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
                directions=["minimize", "minimize"],
                sampler=sampler,
                load_if_exists=True
            )

            self.logger.info(f"\n[*] Starting Optuna Search for [{algo_name}] on [{dataset_name}] ({self.args.iterations} trials)")
            study.optimize(objective, n_trials=self.args.iterations, n_jobs=self.args.jobs, show_progress_bar=True)

            self.logger.info(f"\n[*] Optuna search complete. Found {len(study.best_trials)} optimal trade-off configurations.")
            for i, trial in enumerate(study.best_trials):
                self.logger.info(f"    -> [Frontier {i}] Time: {trial.values[0]:.2f}s, Ratio: {trial.values[1]:.4f}, Params: {trial.params}")

    def _get_averaged_dataframe(self):
        if not self.results: return pd.DataFrame()
        df = pd.DataFrame(self.results)

        # Normalize Time and Ratio per Dataset (Scale: 0.0 to 1.0)
        # Divide by the maximum value observed in each dataset so massive graphs
        # don't mathematically overpower the small graphs.
        df['Time_Norm'] = df.groupby('Dataset')['Time'].transform(lambda x: x / x.max())
        df['Ratio_Norm'] = df.groupby('Dataset')['Ratio'].transform(lambda x: x / x.max())

        # Group by Hyperparameters and calculate the mean
        group_cols = [col for col in df.columns if col not in ['Dataset', 'Time', 'Ratio', 'Time_Norm', 'Ratio_Norm']]
        avg_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()

        # Preserve the raw averages for display, but overwrite the main
        # Time/Ratio columns with the Normalized values so the plotter uses them!
        avg_df['Raw_Time_Avg'] = avg_df['Time']
        avg_df['Raw_Ratio_Avg'] = avg_df['Ratio']

        avg_df['Time'] = avg_df['Time_Norm']    # Now represents "Normalized Time Score"
        avg_df['Ratio'] = avg_df['Ratio_Norm']  # Now represents "Normalized Ratio Score"

        avg_df = avg_df.drop(columns=['Time_Norm', 'Ratio_Norm'])
        avg_df['Dataset'] = 'GLOBAL_NORMALIZED_AVERAGE'

        return avg_df

    def print_table(self):
        avg_df = self._get_averaged_dataframe()
        if avg_df.empty: return

        # Calculate Pareto using the new Normalized 0-1 metrics
        pareto_df = get_pareto_front_2d(avg_df, 'Time', 'Ratio').sort_values(by=["Ratio", "Time"])

        # Rename columns for the terminal so it makes sense to the reader
        display_df = pareto_df.rename(columns={
            'Time': 'Norm_Time_Score',
            'Ratio': 'Norm_Ratio_Score'
        })

        # Rearrange columns so the Raw averages are next to the Normalized scores
        cols = ['Algorithm'] + [c for c in display_df.columns if c not in ['Dataset', 'Algorithm', 'Raw_Time_Avg', 'Raw_Ratio_Avg', 'Norm_Time_Score', 'Norm_Ratio_Score']] + ['Raw_Time_Avg', 'Norm_Time_Score', 'Raw_Ratio_Avg', 'Norm_Ratio_Score']
        display_df = display_df[cols]

        self.logger.info("\n--- OPTUNA OPTIMIZATION: GLOBAL PARETO FRONT (Normalized Scale 0.0 - 1.0) ---")
        self.logger.info(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    def finalize(self):
        if not self.results: return
        raw_df = pd.DataFrame(self.results)
        avg_df = self._get_averaged_dataframe()

        table_output = "--- OPTUNA SEARCH RESULTS (Priority: Ratio > Time) ---\n"
        sorted_avg_df = avg_df.sort_values(by=["Ratio", "Time"])
        table_output += tabulate(sorted_avg_df, headers='keys', tablefmt='grid')

        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_output)

        csv_file = os.path.join(self.session_dir, "optuna_combined_results.csv")
        pd.concat([raw_df, avg_df], ignore_index=True).to_csv(csv_file, index=False)

        try:
            plot_file = os.path.join(self.session_dir, "pareto_front.pdf")
            plot_pareto_front(csv_file, plot_file)
            self.logger.info(f"[*] Pareto front plot saved to: {plot_file}")
        except Exception as e:
            self.logger.error(f"[!] Failed to generate Pareto plot: {e}")

if __name__ == "__main__":
    BayesianOptimizationBenchmark().run()