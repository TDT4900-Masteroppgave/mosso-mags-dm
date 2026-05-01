import optuna
import pandas as pd
from rich.panel import Panel

from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment

class Bayesian(Experiment):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials per algorithm.")
        parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs. Set to -1 to use all cores.")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics: list[dict] = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Setup SQLite Storage for Optuna Dashboard
        db_path = self.session_dir / "optuna_study.db"
        storage_url = f"sqlite:///{db_path}"

        self.logger.info(f"[dim]Optuna Dashboard: [bold green]optuna-dashboard {storage_url}[/bold green][/dim]")
        for algo_name, algo_config in self.active_algos.items():
            self.logger.print(f"[bold cyan]Starting Bayesian Optimization: {algo_name} on {dataset_short_name}[/]")

            def objective(trial):
                params = self._resolve_algo_params(algo_config)

                for param_name, conf in PARAM_CONFIG.items():
                    if param_name in params:
                        val_type = conf["type"]
                        bounds = conf.get("bounds")
                        if bounds:
                            if val_type == int:
                                params[param_name] = str(trial.suggest_int(param_name, bounds[0], bounds[1]))
                            elif val_type == float:
                                params[param_name] = str(trial.suggest_float(param_name, bounds[0], bounds[1]))

                t_avg, r_avg, t_list, r_list = self.execute_runner(
                    algo_name=algo_name,
                    dataset_path=dataset_path,
                    params=params,
                    dataset_short_name=dataset_short_name
                )

                if t_list is None or r_list is None:
                    raise optuna.exceptions.TrialPruned()

                for i, (t, r) in enumerate(zip(t_list, r_list)):
                    metrics.append({
                        "dataset": dataset_short_name,
                        "algorithm": algo_name,
                        "run": i + 1,
                        "time": t,
                        "ratio": r,
                        "trial": trial.number + 1,
                        **params,
                    })
                return t_avg, r_avg

            study_name = f"{algo_name}_{dataset_short_name}"
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=["minimize", "minimize"],
                load_if_exists=True
            )

            study.optimize(
                objective,
                n_trials=self.args.trials,
                n_jobs=self.args.jobs,
                show_progress_bar=True
            )
        return metrics

    def output(self, df: pd.DataFrame):
        display_df = df.copy()
        for ds in display_df['dataset'].unique():
            for algo in display_df['algorithm'].unique():
                df_sub = display_df[(display_df['algorithm'] == algo) & (display_df['dataset'] == ds)].copy()
                if df_sub.empty or 'trial' not in df_sub.columns: continue

                self.logger.print(f"[bold magenta]Optimization Summary:[/] {algo} on {ds}")
                db_path = self.session_dir / "optuna_study.db"
                storage_url = f"sqlite:///{db_path}"
                study_name = f"{algo}_{ds}"

                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                    pareto_size = len(study.best_trials)

                    best_time = df_sub['time'].min()
                    best_ratio = df_sub['ratio'].min()

                    self.logger.print(Panel(
                        f"[bold green]Optuna Search Statistics[/]\n"
                        f"Completed Trials: {completed}\n"
                        f"Pruned (Failed) Trials: {pruned}\n"
                        f"Configurations on Pareto Front: {pareto_size}\n"
                        f"────────\n"
                        f"[bold green]Best Bounds Discovered[/]\n"
                        f"Fastest Time: {best_time:.3f}s\n"
                        f"Smallest Ratio: {best_ratio:.5f}",
                        border_style="green"
                    ))
                except Exception as e:
                    self.logger.debug(f"Could not load study for summary: {e}")


def main():
    with Bayesian() as exp:
        exp.run()

if __name__ == "__main__":
    main()