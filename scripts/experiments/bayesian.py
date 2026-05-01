import json
import optuna
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich import box

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

        for algo_name, algo_config in self.active_algos.items():
            self.logger.debug(f"Starting Bayesian Optimization: {algo_name} on {dataset_short_name}")
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
                    dataset_short_name=dataset_short_name,
                )

                if t_list is None:
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

            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(
                objective,
                n_trials=self.args.trials,
                n_jobs=self.args.jobs,
                show_progress_bar=True
            )

        return metrics

    def output(self, df: pd.DataFrame):
        df.columns = [c.lower() for c in df.columns]

        tunable_params = [p for p in PARAM_CONFIG.keys() if p in df.columns]

        if not tunable_params:
            self.logger.print("[bold red]Error:[/] No parameter columns found. Check the saving process.")
            return

        full_df = df.copy()

        for p in tunable_params:
            full_df[p] = pd.to_numeric(full_df[p], errors='coerce')

        for ds in full_df['dataset'].unique():
            for algo in full_df['algorithm'].unique():
                df_sub = full_df[(full_df['algorithm'] == algo) & (full_df['dataset'] == ds)].copy()
                if df_sub.empty or 'trial' not in df_sub.columns: continue

                agg_df = df_sub.groupby('trial', as_index=False).mean(numeric_only=True)
                if len(agg_df) < 1: continue

                # --- Knee Point Logic ---
                t_min, t_max = agg_df['time'].min(), agg_df['time'].max()
                r_min, r_max = agg_df['ratio'].min(), agg_df['ratio'].max()
                t_range, r_range = (t_max - t_min), (r_max - r_min)

                agg_df['t_norm'] = (agg_df['time'] - t_min) / (t_range if t_range > 0 else 1.0)
                agg_df['r_norm'] = (agg_df['ratio'] - r_min) / (r_range if r_range > 0 else 1.0)
                agg_df['dist'] = np.sqrt(agg_df['t_norm']**2 + agg_df['r_norm']**2)
                knee = agg_df.loc[agg_df['dist'].idxmin()]

                best_trial_data = df_sub[df_sub['trial'] == knee['trial']].iloc[0]

                # FIX: Convert the slice to a dictionary to ensure native Python types for JSON
                best_p_dict = best_trial_data[tunable_params].to_dict()

                best_p_str = json.dumps(best_p_dict)

                self.logger.print(f"\n[bold magenta]Optimization Results:[/] {algo} on {ds}")
                self.logger.print(Panel(
                    f"[bold green]Best Trade-off (Trial {int(knee['trial'])})[/]\n"
                    f"Time: {knee['time']:.3f}s | Ratio: {knee['ratio']:.5f}\n"
                    f"Params: {best_p_str}",
                    border_style="green"
                ))

                tunable_params = [p for p in PARAM_CONFIG.keys() if p in df_sub.columns]
                if tunable_params:
                    corr_table = Table(
                        title="Parameter Impact Profile (Correlation)",
                        box=box.SIMPLE,
                        header_style="bold yellow",
                        caption="[dim]Green means an increase in the parameter improves the metric (lowers time/ratio).\nRed means an increase worsens the metric.[/dim]"
                    )
                    corr_table.add_column("Parameter", style="cyan")
                    corr_table.add_column("Impact on Time", justify="right")
                    corr_table.add_column("Impact on Ratio", justify="right")

                    for p in tunable_params:
                        if df_sub[p].nunique() > 1: # Only measure params that actually varied
                            t_corr = df_sub[p].corr(df_sub['time'])
                            r_corr = df_sub[p].corr(df_sub['ratio'])

                            # Format nicely: Positive correlation to a "lower is better" metric is BAD (Red)
                            t_fmt = f"[red]{t_corr:+.2f}[/]" if t_corr > 0 else f"[green]{t_corr:+.2f}[/]"
                            r_fmt = f"[red]{r_corr:+.2f}[/]" if r_corr > 0 else f"[green]{r_corr:+.2f}[/]"

                            corr_table.add_row(p, t_fmt, r_fmt)

                    self.logger.print(corr_table)


def main():
<<<<<<< HEAD
    with BayesianOpt() as exp:
=======
    with Bayesian() as exp:
>>>>>>> main
        exp.run()

if __name__ == "__main__":
    main()