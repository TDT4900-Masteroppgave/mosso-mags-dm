"""Cross-dataset generalization: take Pareto-optimal configs from a prior Bayesian
study tuned on dataset A, evaluate them on datasets B, C, ... without retuning.

Detects hyperparameter overfitting: if --tune-on configs underperform on test
datasets relative to the Bayesian-tuned configs *for those datasets*, the
improvement may be dataset-specific rather than algorithmic.

Required:
  --bayesian-session <path>   Prior bayesian session containing optuna_study.db
  --tune-on <DS>              Dataset short_name whose Pareto front to extract
  --dataset <DS ...>          Test datasets (use base --dataset / --group flags)
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from rich import box
from rich.table import Table

from scripts.experiments.base_experiment import Experiment


KNEE_LABELS = ("min_time", "min_ratio", "knee")


def _pareto_picks(study: optuna.Study) -> dict[str, optuna.trial.FrozenTrial]:
    pareto = [t for t in study.best_trials if t.values is not None]
    if not pareto:
        return {}

    times = np.array([t.values[0] for t in pareto])
    ratios = np.array([t.values[1] for t in pareto])

    min_time_trial = pareto[int(np.argmin(times))]
    min_ratio_trial = pareto[int(np.argmin(ratios))]

    if len(pareto) <= 2 or min_time_trial.number == min_ratio_trial.number:
        return {"min_time": min_time_trial, "min_ratio": min_ratio_trial,
                "knee": min_ratio_trial}

    t_norm = (times - times.min()) / (times.max() - times.min() + 1e-12)
    r_norm = (ratios - ratios.min()) / (ratios.max() - ratios.min() + 1e-12)
    p1 = np.array([t_norm[np.argmin(times)], r_norm[np.argmin(times)]])
    p2 = np.array([t_norm[np.argmin(ratios)], r_norm[np.argmin(ratios)]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec) + 1e-12
    distances = []
    for i in range(len(pareto)):
        pt = np.array([t_norm[i], r_norm[i]])
        cross = abs(np.cross(line_vec, pt - p1))
        distances.append(cross / line_len)
    knee_idx = int(np.argmax(distances))

    return {"min_time": min_time_trial, "min_ratio": min_ratio_trial, "knee": pareto[knee_idx]}


class Generalization(Experiment):
    def __init__(self):
        super().__init__("generalization")

    def add_custom_args(self, parser):
        parser.add_argument("--bayesian-session", type=Path, required=True,
                            help="Path to prior bayesian session dir (containing optuna_study.db).")
        parser.add_argument("--tune-on", type=str, required=True,
                            help="Dataset short_name whose Pareto front to extract.")
        parser.add_argument("--picks", nargs="+", choices=KNEE_LABELS, default=list(KNEE_LABELS),
                            help="Which Pareto picks to evaluate.")

    def _load_picks(self, algo_name: str) -> dict[str, dict]:
        db_path = self.args.bayesian_session / "optuna_study.db"
        if not db_path.exists():
            self.logger.warning(f"[!] {db_path} not found")
            return {}
        storage_url = f"sqlite:///{db_path}"
        study_name = f"{algo_name}_{self.args.tune_on}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
        except Exception as e:
            self.logger.warning(f"[!] Could not load study {study_name}: {e}")
            return {}
        picks = _pareto_picks(study)
        return {label: t.params for label, t in picks.items() if label in self.args.picks}

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        if dataset_short_name == self.args.tune_on:
            self.logger.info(f"=> Skipping tune-on dataset {dataset_short_name} (in-sample).")
            return []

        metrics: list[dict] = []
        for algo_name, algo_config in self.active_algos.items():
            picks = self._load_picks(algo_name)
            if not picks:
                self.logger.info(f"=> No picks for {algo_name}; skipping.")
                continue

            for label, tuned_params in picks.items():
                params = self._resolve_algo_params(algo_config)
                for k, v in tuned_params.items():
                    if k in params:
                        params[k] = str(v)

                self.logger.print(f"[cyan]{algo_name}/{label}[/] on {dataset_short_name} | tuned_on={self.args.tune_on}")
                result = self.execute_runner(
                    algo_name=algo_name, dataset_path=dataset_path,
                    params=params, dataset_short_name=dataset_short_name,
                )
                if result is None:
                    continue
                _, _, t_list, r_list = result
                if t_list is None or r_list is None:
                    continue
                for i, (t, r) in enumerate(zip(t_list, r_list)):
                    metrics.append({
                        "dataset": dataset_short_name,
                        "algorithm": algo_name,
                        "tune_on": self.args.tune_on,
                        "pick": label,
                        "run": i + 1,
                        "time": t,
                        "ratio": r,
                        **params,
                    })
        return metrics

    def output(self, df: pd.DataFrame):
        if df.empty:
            return
        agg = df.groupby(["algorithm", "pick", "dataset"], as_index=False)[["time", "ratio"]].mean()
        table = Table(title=f"Generalization: tuned on {self.args.tune_on}, evaluated elsewhere",
                      box=box.SIMPLE, header_style="bold magenta")
        for col in ("Algorithm", "Pick", "Dataset", "Avg Time", "Avg Ratio"):
            table.add_column(col)
        for _, r in agg.sort_values(["algorithm", "pick", "dataset"]).iterrows():
            table.add_row(r["algorithm"], r["pick"], r["dataset"],
                          f"{r['time']:.3f}s", f"{r['ratio']:.5f}")
        self.logger.print(table)


def main():
    with Generalization() as exp:
        exp.run()


if __name__ == "__main__":
    main()
