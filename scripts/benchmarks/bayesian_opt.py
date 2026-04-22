import os
import pandas as pd
from tabulate import tabulate
import optuna
import warnings

from scripts.config import PARAM_CONFIG
from scripts.plotting import get_pareto_front_2d
from scripts.benchmark import Benchmark
import scripts.db as db

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


class BayesianOptimizationBenchmark(Benchmark):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--iterations", type=int, default=30, help="Total number of points to sample")
        parser.add_argument("--n-startup", type=int, default=10, help="Initial random explorations before AI kicks in")
        parser.add_argument("--jobs", type=int, default=1, help="Number of parallel threads to run (-1 uses all CPUs)")

    def process(self, dataset_path: str, ds: dict, dataset_name: str):
        for algo_name, algo_config in self.active_algos.items():
            template = algo_config.get('template', [])
            if not template:
                self.logger.info(f"\n[*] Skipping [{algo_name}]: No hyperparameters to optimize.")
                continue

            def objective(objective_trial):
                resolved_params = {}
                for p_key in template:
                    bounds = PARAM_CONFIG.get(p_key, {}).get("bounds")
                    if bounds:
                        resolved_params[p_key] = objective_trial.suggest_int(p_key, bounds[0], bounds[1])
                    else:
                        resolved_params[p_key] = PARAM_CONFIG.get(p_key, {}).get("default")

                # Fill remaining params with defaults
                full_params = self._resolve_algo_params(algo_config, resolved_params)

                avg_time, avg_ratio, _, _, _ = self.execute_runner(
                    algo_name, algo_config, dataset_path, dataset_name, full_params
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

            # Scope the DB to this session to avoid accumulating stale trials across runs
            db_path = os.path.join(self.session_dir, "optuna_study.db")
            study_name = f"{algo_name}_{dataset_name}"
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=self.args.n_startup,
                seed=self.args.seed,
            )

            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}",
                directions=["minimize", "minimize"],
                sampler=sampler,
                load_if_exists=False,  # fresh study per session for reproducibility
            )

            self.logger.info(f"\n[*] Starting Optuna Search for [{algo_name}] on [{dataset_name}] ({self.args.iterations} trials)")
            study.optimize(objective, n_trials=self.args.iterations, n_jobs=self.args.jobs, show_progress_bar=True)

            self.logger.info(f"\n[*] Optuna search complete. Found {len(study.best_trials)} optimal trade-off configurations.")
            for i, trial in enumerate(study.best_trials):
                self.logger.info(f"    -> [Frontier {i}] Time: {trial.values[0]:.2f}s, Ratio: {trial.values[1]:.4f}, Params: {trial.params}")

    def _get_averaged_dataframe(self):
        if not self.results: return pd.DataFrame()
        df = pd.DataFrame(self.results)

        df['Time_Norm'] = df.groupby('Dataset')['Time'].transform(lambda x: x / x.max())
        df['Ratio_Norm'] = df.groupby('Dataset')['Ratio'].transform(lambda x: x / x.max())

        group_cols = [col for col in df.columns if col not in ['Dataset', 'Time', 'Ratio', 'Time_Norm', 'Ratio_Norm']]
        avg_df = df.groupby(group_cols, dropna=False).mean(numeric_only=True).reset_index()

        avg_df['Raw_Time_Avg'] = avg_df['Time']
        avg_df['Raw_Ratio_Avg'] = avg_df['Ratio']

        avg_df['Time'] = avg_df['Time_Norm']
        avg_df['Ratio'] = avg_df['Ratio_Norm']

        avg_df = avg_df.drop(columns=['Time_Norm', 'Ratio_Norm'])
        avg_df['Dataset'] = 'GLOBAL_NORMALIZED_AVERAGE'

        return avg_df

    def print_table(self):
        avg_df = self._get_averaged_dataframe()
        if avg_df.empty: return

        self.logger.info("\n--- OPTUNA OPTIMIZATION: PARETO FRONTS (Normalized Scale 0.0 - 1.0) ---")

        for algo in avg_df['Algorithm'].unique():
            algo_df = avg_df[avg_df['Algorithm'] == algo]
            pareto_df = get_pareto_front_2d(algo_df, 'Time', 'Ratio').sort_values(by=["Ratio", "Time"])

            display_df = pareto_df.rename(columns={
                'Time': 'Norm_Time_Score',
                'Ratio': 'Norm_Ratio_Score'
            })

            cols = ['Algorithm'] + [c for c in display_df.columns if c not in ['Dataset', 'Algorithm', 'Raw_Time_Avg', 'Raw_Ratio_Avg', 'Norm_Time_Score', 'Norm_Ratio_Score']] + ['Raw_Time_Avg', 'Norm_Time_Score', 'Raw_Ratio_Avg', 'Norm_Ratio_Score']
            display_df = display_df[cols]

            self.logger.info(f"\n[*] Pareto Front for: {algo}")
            self.logger.info(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    def finalize(self):
        if not self.results: return
        raw_df = pd.DataFrame(self.results)
        avg_df = self._get_averaged_dataframe()

        table_output = "--- OPTUNA SEARCH RESULTS ---\n"

        for algo in avg_df['Algorithm'].unique():
            algo_df = avg_df[avg_df['Algorithm'] == algo]
            pareto_df = get_pareto_front_2d(algo_df, 'Time', 'Ratio').sort_values(by=["Ratio", "Time"])

            table_output += f"\n[*] PARETO FRONT: {algo}\n"
            table_output += tabulate(pareto_df, headers='keys', tablefmt='grid', showindex=False) + "\n"

        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_output)

        csv_file = os.path.join(self.session_dir, "optuna_combined_results.csv")
        pd.concat([raw_df, avg_df], ignore_index=True).to_csv(csv_file, index=False)

        # Write per-trial results to unified DB
        if self.db_conn:
            param_cols = [c for c in raw_df.columns
                          if c not in {"Dataset", "Algorithm", "Time", "Ratio"}]
            for trial_i, (_, row) in enumerate(raw_df.iterrows(), 1):
                params = {c: row[c] for c in param_cols if not pd.isna(row.get(c))}
                db.write_result(
                    self.db_conn,
                    algorithm=row["Algorithm"], dataset=row["Dataset"],
                    time=row.get("Time"), ratio=row.get("Ratio"),
                    trial=trial_i,
                    params=params,
                )


if __name__ == "__main__":
    BayesianOptimizationBenchmark().run()
