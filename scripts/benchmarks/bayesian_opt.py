import os
import pandas as pd
import optuna
import warnings

from scripts.config import PARAM_CONFIG
from scripts.experiment import Experiment
import scripts.db as db

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


class BayesianOptimization(Experiment):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--iterations", type=int, default=30, help="Total number of points to sample")
        parser.add_argument("--n-startup", type=int, default=10, help="Initial random explorations before AI kicks in")
        parser.add_argument("--jobs", type=int, default=1, help="Number of parallel threads to run (-1 uses all CPUs)")

    def process(self, dataset_path: str, dataset_short_name: str):
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

                avg_time, avg_ratio, _, _, = self.execute_runner(
                    algo_name, algo_config, dataset_path, dataset_short_name, full_params
                )

                if avg_time is None or avg_ratio is None:
                    raise optuna.exceptions.TrialPruned()

                result_entry = {
                    'Dataset': dataset_short_name,
                    'Algorithm': algo_name,
                    'Time': avg_time,
                    'Ratio': avg_ratio
                }
                result_entry.update(resolved_params)
                self.results.append(result_entry)

                return avg_time, avg_ratio

            # Scope the DB to this session to avoid accumulating stale trials across runs
            db_path = os.path.join(self.session_dir, "optuna_study.db")
            study_name = f"{algo_name}_{dataset_short_name}"
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

            self.logger.info(f"\n[*] Starting Optuna Search for [{algo_name}] on [{dataset_short_name}] ({self.args.iterations} trials)")
            study.optimize(objective, n_trials=self.args.iterations, n_jobs=self.args.jobs, show_progress_bar=True)

            self.logger.info(f"\n[*] Optuna search complete. Found {len(study.best_trials)} optimal trade-off configurations.")
            for i, trial in enumerate(study.best_trials):
                self.logger.info(f"    -> [Frontier {i}] Time: {trial.values[0]:.2f}s, Ratio: {trial.values[1]:.4f}, Params: {trial.params}")

    def finalize(self):
        raw_df = pd.DataFrame(self.results)

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
    BayesianOptimization().run()
