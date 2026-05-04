import optuna
import pandas as pd
from rich.panel import Panel

from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment

class Bayesian(Experiment):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials per algorithm.")
        parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs. Set to -1 to use all cores.")
        parser.add_argument("--strict-baseline", action="store_true", help="Prune trials that are slower than kdd20-mosso.")

    def _calculate_baseline(self) -> dict[str, float]:
        baseline_times = {}
        baseline_config = self.active_algos["kdd20-mosso"]
        baseline_params = self._resolve_algo_params(baseline_config)

        for ds in self.datasets_to_run:
            short_name, dataset_path = self._get_dataset(ds)
            t_avg, _, _, _ = self.execute_runner(
                algo_name="kdd20-mosso",
                dataset_path=dataset_path,
                params=baseline_params,
                dataset_short_name=short_name
            )
            if t_avg:
                baseline_times[short_name] = t_avg
        return baseline_times

    def process(self) -> list[dict]:
        metrics: list[dict] = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Setup SQLite Storage for Optuna
        db_path = self.session_dir / "optuna_study.db"
        storage_url = f"sqlite:///{db_path}"
        self.logger.info(f"[dim]Optuna Dashboard: [bold green]optuna-dashboard {storage_url}[/bold green][/dim]")

        # Calculate Baseline Times
        baseline_times = {}
        if self.args.strict_baseline and "kdd20-mosso" in self.active_algos:
            self.logger.print("[bold yellow]Calculating Baseline Times (kdd20-mosso)[/bold yellow]")
            baseline_times = self._calculate_baseline()
        else:
            self.logger.warning("Baseline 'kdd20-mosso' not found in active algorithms. No time constraints will be applied.")

        for algo_name, algo_config in self.active_algos.items():
            if algo_name == "kdd20-mosso":
                continue # Don't optimize the baseline

            study_name = f"{algo_name}"
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=["minimize"],
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            )
            self.logger.print(f"[bold cyan]Starting Bayesian Optimization: {algo_name}[/]")

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

                trial_ratios = []

                for i, ds in enumerate(self.datasets_to_run, 1):
                    short_name, dataset_path = self._get_dataset(ds)
                    self._print_status(i, short_name)

                    t_avg, r_avg, t_list, r_list = self.execute_runner(
                        algo_name=algo_name,
                        dataset_path=dataset_path,
                        params=params,
                        dataset_short_name=short_name
                    )

                    if t_avg is None or r_avg is None:
                        raise optuna.exceptions.TrialPruned()

                    # Strict Time Pruning
                    if self.args.strict_baseline and short_name in baseline_times:
                        # Allow a 10% buffer for standard variance
                        max_allowed_time = baseline_times[short_name] * 1.10
                        if t_avg > max_allowed_time:
                            self.logger.info(f"[yellow]Trial {trial.number} pruned: Exceeded baseline time on {short_name} ({t_avg:.3f}s > {max_allowed_time:.3f}s)[/yellow]")
                            raise optuna.exceptions.TrialPruned()

                    for run, (t, r) in enumerate(zip(t_list, r_list)):
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "run": run + 1,
                            "time": t,
                            "ratio": r,
                            "trial": trial.number + 1,
                            **params,
                        })

                    trial_ratios.append(r_avg)

                    # Report ratio to pruner to kill bad compression early
                    trial.report(r_avg, step=i)
                    if trial.should_prune():
                        self.logger.info(f"[yellow]Trial {trial.number} pruned: Poor compression ratio on {short_name}[/yellow]")
                        raise optuna.exceptions.TrialPruned()

                joint_ratio = sum(trial_ratios) / len(trial_ratios)

                return joint_ratio

            study.optimize(
                objective,
                n_trials=self.args.trials,
                n_jobs=self.args.jobs,
                show_progress_bar=True
            )
        return metrics

    def output(self, df: pd.DataFrame):
        display_df = df.copy()
        for algo in display_df['algorithm'].unique():
            df_sub = display_df[display_df['algorithm'] == algo].copy()
            if df_sub.empty or 'trial' not in df_sub.columns: continue

            self.logger.print(f"[bold magenta]Optimization Summary:[/] {algo} (Global)")
            db_path = self.session_dir / "optuna_study.db"
            storage_url = f"sqlite:///{db_path}"

            study_name = f"{algo}"

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