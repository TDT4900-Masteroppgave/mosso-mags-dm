import optuna
import pandas as pd
from rich.panel import Panel

from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment

class Bayesian(Experiment):
    def __init__(self):
        super().__init__("bayesian")

    def add_custom_args(self, parser):
        parser.add_argument("--trials", type=int, default=100)
        parser.add_argument("--jobs", type=int, default=1)

    def process(self) -> list[dict]:
        metrics = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage_url = f"sqlite:///{self.session_dir / 'optuna_study.db'}"

        self.logger.info(f"[dim]Optuna Dashboard: [bold green]optuna-dashboard {storage_url}[/bold green][/dim]")

        for algo_name, algo_config in self.active_algos.items():
            study = optuna.create_study(
                study_name=algo_name,
                storage=storage_url,
                directions=["minimize", "minimize"],
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            )

            study.set_user_attr("algorithm_name", algo_name)
            study.set_user_attr("algorithm_type", algo_config.get("type", "unknown"))
            study.set_user_attr("git_branch", algo_config.get("branch", "local"))
            study.set_user_attr("datasets_optimized", [ds['short_name'] for ds in self.datasets_to_run])
            study.set_user_attr("time_normalization", "seconds_per_edge")
            study.set_user_attr("objective_1", "Minimized Avg Normalized Time")
            study.set_user_attr("objective_2", "Minimized Avg Compression Ratio")

            self.logger.print(f"[bold cyan]Starting Bayesian Optimization: {algo_name}[/]")

            def objective(trial):
                params = self._resolve_algo_params(algo_config)

                for p_name, conf in PARAM_CONFIG.items():
                    if p_name in params and "bounds" in conf:
                        bounds = conf["bounds"]
                        step = conf.get("step")

                        if conf["type"] == int:
                            params[p_name] = str(trial.suggest_int(p_name, bounds[0], bounds[1], step=step or 1))
                        else:
                            params[p_name] = str(trial.suggest_float(p_name, bounds[0], bounds[1], step=step))

                trial_times = []
                trial_ratios = []

                for ds in self.datasets_to_run:
                    short_name, dataset_path = self._get_dataset(ds)
                    res = self.execute_runner(algo_name, dataset_path, short_name, params)

                    if not res:
                        raise optuna.exceptions.TrialPruned()

                    t_avg, r_avg, t_list, r_list = res
                    trial.set_user_attr(f"raw_time_{short_name}", f"{t_avg:.4f}s")
                    trial.set_user_attr(f"ratio_{short_name}", f"{r_avg:.5f}")

                    for i, (t, r) in enumerate(zip(t_list, r_list)):
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "run": i + 1,
                            "time": t,
                            "ratio": r,
                            "trial": trial.number + 1,
                            **params
                        })

                    edges = ds.get("meta", {}).get("edges", 1)
                    trial_times.append(t_avg / edges)
                    trial_ratios.append(r_avg)

                avg_time = sum(trial_times) / len(trial_times)
                avg_ratio = sum(trial_ratios) / len(trial_ratios)
                return avg_time, avg_ratio

            study.optimize(objective, n_trials=self.args.trials, n_jobs=self.args.jobs, show_progress_bar=True)

        return metrics

    def output(self, df: pd.DataFrame):
        for algo in df['algorithm'].unique():
            df_sub = df[df['algorithm'] == algo]
            if df_sub.empty or 'trial' not in df_sub.columns:
                continue

            try:
                storage_url = f"sqlite:///{self.session_dir / 'optuna_study.db'}"
                study = optuna.load_study(study_name=algo, storage=storage_url)

                completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
                pruned = len([t for t in study.trials if t.state.name == 'PRUNED'])
                pareto = len(study.best_trials)
                best_t = df_sub['time'].min()
                best_r = df_sub['ratio'].min()

                summary = (
                    f"[bold green]Optuna Search Statistics[/]\n"
                    f"Completed: {completed}\n"
                    f"Pruned: {pruned}\n"
                    f"Pareto Front: {pareto}\n"
                    f"────────\n"
                    f"[bold green]Best Bounds[/]\n"
                    f"Fastest: {best_t:.3f}s\n"
                    f"Smallest Ratio: {best_r:.5f}"
                )
                self.logger.print(Panel(summary, border_style="green"))
            except Exception as e:
                self.logger.debug(f"Summary load error: {e}")

def main():
    with Bayesian() as exp:
        exp.run()

if __name__ == "__main__":
    main()