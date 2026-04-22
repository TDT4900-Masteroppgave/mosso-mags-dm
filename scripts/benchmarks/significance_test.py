import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tabulate import tabulate

from scripts.benchmark import Benchmark
from scripts.plotting import get_pareto_front_2d
from scripts.runners import get_runner
import scripts.db as db
from scripts.stats import (
    bootstrap_ci,
    cliffs_delta,
    holm_bonferroni,
    interp_time_at_ratio,
    knee_point,
    paired_wilcoxon,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


class SignificanceTestBenchmark(Benchmark):
    def __init__(self):
        super().__init__("significance")
        self.knee_log: list[dict] = []

    def add_custom_args(self, parser):
        parser.add_argument(
            "--bo-session", action="append", required=True,
            help="Path to a BO session directory containing optuna_study.db. Repeatable."
        )
        parser.add_argument(
            "--reps", type=int, default=15,
            help="Repetitions per (strategy, dataset) cell. Each is one JVM invocation."
        )
        parser.add_argument(
            "--baseline-algo", type=str, default="kdd20-mosso",
            help="Algorithm name used as the comparison reference."
        )
        parser.add_argument(
            "--alpha", type=float, default=0.05,
            help="Significance threshold for Holm-Bonferroni."
        )

    def _study_storages(self) -> list[str]:
        storages = []
        for sess in self.args.bo_session:
            db = Path(sess) / "optuna_study.db"
            if not db.exists():
                self.logger.warning(f"[!] No optuna_study.db in {sess}, skipping.")
                continue
            storages.append(f"sqlite:///{db}")
        if not storages:
            raise RuntimeError("No usable BO sessions found.")
        return storages

    def _load_study(self, algo_name: str, dataset_name: str):
        """Try each provided BO session; return the first study found, or None."""
        study_name = f"{algo_name}_{dataset_name}"
        for storage in self._study_storages():
            try:
                return optuna.load_study(study_name=study_name, storage=storage)
            except KeyError:
                continue
            except Exception as e:
                self.logger.debug(f"[!] Failed to load {study_name} from {storage}: {e}")
                continue
        return None

    def _knee_config(self, study, algo_name: str, dataset_name: str):
        """Returns (chosen_params dict, knee_time, knee_ratio) or None if unavailable."""
        df = study.trials_dataframe()
        if df.empty or "values_0" not in df.columns or "values_1" not in df.columns:
            return None
        df = df[df["state"] == "COMPLETE"].dropna(subset=["values_0", "values_1"])
        if df.empty:
            return None
        df = df.rename(columns={"values_0": "Time", "values_1": "Ratio"})
        pareto = get_pareto_front_2d(df, "Time", "Ratio").sort_values("Time").reset_index(drop=True)
        if pareto.empty:
            return None

        idx = knee_point(pareto["Time"].tolist(), pareto["Ratio"].tolist())
        row = pareto.iloc[idx]
        params = {c.replace("params_", ""): row[c] for c in pareto.columns if c.startswith("params_")}
        # Optuna stores ints as float; cast back where the param config expects an int.
        for k, v in list(params.items()):
            if isinstance(v, float) and v.is_integer():
                params[k] = int(v)
        return params, float(row["Time"]), float(row["Ratio"])

    def _execute_reps(self, algo_name, algo_config, dataset_path, dataset_name, params):
        """Run N independent JVM invocations and return raw lists."""
        runner = get_runner(algo_name, self.logger, str(self.session_dir))
        if not runner.binary_exists():
            self.logger.warning(f"[!] Binary missing for {algo_name}; skipping.")
            return [], []
        template = algo_config.get("template", [])
        # Ensure interval is present for mosso-type algos that need it.
        merged = {**params}
        if "interval" in template and "interval" not in merged:
            merged["interval"] = self.args.interval

        _, _, times, ratios, _ = runner.run_multiple(
            dataset_path=dataset_path,
            base_output_name=f"{algo_name}_{dataset_name}_{self.timestamp}",
            runs=self.args.reps,
            parameters=merged,
            template=template,
        )
        return times or [], ratios or []

    def process(self, dataset_path, ds, dataset_name):
        if self.args.baseline_algo not in self.active_algos:
            self.logger.error(
                f"[!] Baseline '{self.args.baseline_algo}' not in active algos; pass it via --algorithm."
            )
            return

        for algo_name, algo_config in self.active_algos.items():
            study = self._load_study(algo_name, dataset_name)
            if study is None:
                self.logger.warning(f"[!] No BO study for {algo_name}/{dataset_name}; skipping cell.")
                continue
            knee = self._knee_config(study, algo_name, dataset_name)
            if knee is None:
                self.logger.warning(f"[!] No completed trials for {algo_name}/{dataset_name}; skipping.")
                continue
            params, k_time, k_ratio = knee
            self.logger.info(
                f"\t[{algo_name}] knee on {dataset_name}: time={k_time:.3f} ratio={k_ratio:.5f} params={params}"
            )
            self.knee_log.append({
                "Dataset": dataset_name, "Algorithm": algo_name,
                "Knee_Time": k_time, "Knee_Ratio": k_ratio, **{f"param_{k}": v for k, v in params.items()}
            })

            times, ratios = self._execute_reps(algo_name, algo_config, dataset_path, dataset_name, params)
            for i, (t, r) in enumerate(zip(times, ratios), 1):
                self.results.append({
                    "Dataset": dataset_name,
                    "Algorithm": algo_name,
                    "Rep": i,
                    "Time": t,
                    "Ratio": r,
                    **{f"param_{k}": v for k, v in params.items()},
                })

    def _build_summary(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        baseline = self.args.baseline_algo
        non_baseline = sorted(a for a in raw_df["Algorithm"].unique() if a != baseline)

        # Per (algo, dataset) medians for the paired test.
        med = raw_df.groupby(["Algorithm", "Dataset"])[["Time", "Ratio"]].median().reset_index()

        baseline_time_med = dict(zip(med[med["Algorithm"] == baseline]["Dataset"],
                                     med[med["Algorithm"] == baseline]["Time"]))
        baseline_ratio_med = dict(zip(med[med["Algorithm"] == baseline]["Dataset"],
                                      med[med["Algorithm"] == baseline]["Ratio"]))

        baseline_time_pool = raw_df[raw_df["Algorithm"] == baseline]["Time"].tolist()
        baseline_ratio_pool = raw_df[raw_df["Algorithm"] == baseline]["Ratio"].tolist()

        raw_p_time, raw_p_ratio = {}, {}
        rows = []
        for algo in non_baseline:
            sub = med[med["Algorithm"] == algo]
            s_time = dict(zip(sub["Dataset"], sub["Time"]))
            s_ratio = dict(zip(sub["Dataset"], sub["Ratio"]))

            _, p_time = paired_wilcoxon(s_time, baseline_time_med)
            _, p_ratio = paired_wilcoxon(s_ratio, baseline_ratio_med)
            raw_p_time[algo], raw_p_ratio[algo] = p_time, p_ratio

            algo_time_pool = raw_df[raw_df["Algorithm"] == algo]["Time"].tolist()
            algo_ratio_pool = raw_df[raw_df["Algorithm"] == algo]["Ratio"].tolist()
            d_time, mag_time = cliffs_delta(algo_time_pool, baseline_time_pool)
            d_ratio, mag_ratio = cliffs_delta(algo_ratio_pool, baseline_ratio_pool)

            shared = sorted(set(s_time) & set(baseline_time_med))
            if shared:
                t_ratio_arr = np.array([s_time[k] / baseline_time_med[k] for k in shared])
                r_ratio_arr = np.array([s_ratio[k] / baseline_ratio_med[k] for k in shared])
                t_rel = float(np.median(t_ratio_arr))
                r_rel = float(np.median(r_ratio_arr))
            else:
                t_rel = r_rel = float("nan")

            t_ci_lo, t_ci_hi = bootstrap_ci(algo_time_pool, seed=self.args.seed)
            r_ci_lo, r_ci_hi = bootstrap_ci(algo_ratio_pool, seed=self.args.seed)

            rows.append({
                "Strategy": algo,
                "Datasets": len(shared),
                "Median Time / Baseline": t_rel,
                "Time CI (lo)": t_ci_lo,
                "Time CI (hi)": t_ci_hi,
                "p_time (raw)": p_time,
                "Cliff's d_time": d_time,
                "d_time mag": mag_time,
                "Median Ratio / Baseline": r_rel,
                "Ratio CI (lo)": r_ci_lo,
                "Ratio CI (hi)": r_ci_hi,
                "p_ratio (raw)": p_ratio,
                "Cliff's d_ratio": d_ratio,
                "d_ratio mag": mag_ratio,
            })

        adj_time = holm_bonferroni(raw_p_time, alpha=self.args.alpha)
        adj_ratio = holm_bonferroni(raw_p_ratio, alpha=self.args.alpha)

        for row in rows:
            algo = row["Strategy"]
            ap_t, sig_t = adj_time.get(algo, (float("nan"), False))
            ap_r, sig_r = adj_ratio.get(algo, (float("nan"), False))
            row["p_time (Holm)"] = ap_t
            row["p_ratio (Holm)"] = ap_r
            row["Verdict_Time"] = self._verdict(row["Median Time / Baseline"], sig_t)
            row["Verdict_Ratio"] = self._verdict(row["Median Ratio / Baseline"], sig_r)

        # Column order
        col_order = [
            "Strategy", "Datasets",
            "Median Time / Baseline", "Time CI (lo)", "Time CI (hi)",
            "p_time (raw)", "p_time (Holm)", "Cliff's d_time", "d_time mag", "Verdict_Time",
            "Median Ratio / Baseline", "Ratio CI (lo)", "Ratio CI (hi)",
            "p_ratio (raw)", "p_ratio (Holm)", "Cliff's d_ratio", "d_ratio mag", "Verdict_Ratio",
        ]
        return pd.DataFrame(rows)[col_order]

    def _iso_ratio_table(self) -> pd.DataFrame:
        """For each dataset, interpolate each algorithm's Pareto front to find the time
        it would need to match the baseline's knee-point compression ratio exactly.

        Speedup_vs_Baseline > 1  → algorithm is faster at equal compression.
        Speedup_vs_Baseline < 1  → algorithm is slower at equal compression.
        Pct_Slower > 0           → percentage overhead vs baseline at that ratio.
        """
        baseline = self.args.baseline_algo
        datasets = sorted({e["Dataset"] for e in self.knee_log})
        rows = []

        for dataset_name in datasets:
            b_study = self._load_study(baseline, dataset_name)
            if b_study is None:
                continue
            b_knee = self._knee_config(b_study, baseline, dataset_name)
            if b_knee is None:
                continue
            _, b_time, b_ratio = b_knee

            rows.append({
                "Dataset": dataset_name,
                "Algorithm": baseline,
                "Baseline_Knee_Ratio": b_ratio,
                "Baseline_Knee_Time_s": b_time,
                "Iso_Time_s": b_time,
                "Speedup_vs_Baseline": 1.0,
                "Pct_Slower": 0.0,
                "Note": "baseline",
            })

            for algo_name in self.active_algos:
                if algo_name == baseline:
                    continue
                study = self._load_study(algo_name, dataset_name)
                if study is None:
                    continue
                df = study.trials_dataframe()
                if df.empty or "values_0" not in df.columns:
                    continue
                df = df[df["state"] == "COMPLETE"].dropna(subset=["values_0", "values_1"])
                if df.empty:
                    continue
                df = df.rename(columns={"values_0": "Time", "values_1": "Ratio"})
                pareto = get_pareto_front_2d(df, "Time", "Ratio")
                if pareto.empty:
                    continue

                iso_time = interp_time_at_ratio(pareto, b_ratio)
                if iso_time is None:
                    continue

                speedup = b_time / iso_time
                pct_slower = (iso_time / b_time - 1.0) * 100.0
                direction = "faster" if speedup > 1.0 else "slower"
                rows.append({
                    "Dataset": dataset_name,
                    "Algorithm": algo_name,
                    "Baseline_Knee_Ratio": b_ratio,
                    "Baseline_Knee_Time_s": b_time,
                    "Iso_Time_s": iso_time,
                    "Speedup_vs_Baseline": speedup,
                    "Pct_Slower": pct_slower,
                    "Note": f"{direction} by {abs(pct_slower):.1f}%",
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def _verdict(rel_median: float, significant: bool) -> str:
        if np.isnan(rel_median):
            return "n/a"
        if not significant:
            return "n.s."
        # rel = strategy / baseline; <1 means strategy smaller (faster / better compression)
        return "better*" if rel_median < 1.0 else "worse*"

    def print_table(self):
        raw_df = pd.DataFrame(self.results)
        if raw_df.empty:
            self.logger.warning("[!] No results to summarize.")
            return
        summary = self._build_summary(raw_df)

        self.logger.info("\n--- KNEE POINTS PICKED ---")
        knee_df = pd.DataFrame(self.knee_log)
        self.logger.info(tabulate(knee_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

        iso_df = self._iso_ratio_table()
        if not iso_df.empty:
            self.logger.info(
                f"\n--- ISO-RATIO: time to match {self.args.baseline_algo} knee-point compression ---"
            )
            self.logger.info(tabulate(iso_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

        self.logger.info(
            f"\n--- SIGNIFICANCE vs {self.args.baseline_algo} "
            f"(α={self.args.alpha}, Wilcoxon paired across datasets, Holm-Bonferroni) ---"
        )
        self.logger.info(tabulate(summary, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    def finalize(self):
        raw_df = pd.DataFrame(self.results)
        if raw_df.empty:
            return

        raw_csv = self.session_dir / "results_raw.csv"
        raw_df.to_csv(raw_csv, index=False)

        summary = self._build_summary(raw_df)
        summary_csv = self.session_dir / "summary.csv"
        summary.to_csv(summary_csv, index=False)

        knee_csv = self.session_dir / "knee_points.csv"
        pd.DataFrame(self.knee_log).to_csv(knee_csv, index=False)

        iso_df = self._iso_ratio_table()
        if not iso_df.empty:
            iso_df.to_csv(self.session_dir / "iso_ratio.csv", index=False)

        with open(self.session_dir / "table_results.txt", "w") as f:
            f.write(f"Baseline: {self.args.baseline_algo}\n")
            f.write(f"Reps per cell: {self.args.reps}\n")
            f.write(f"Alpha: {self.args.alpha}\n\n")
            f.write("--- KNEE POINTS ---\n")
            f.write(tabulate(pd.DataFrame(self.knee_log), headers="keys",
                             tablefmt="grid", showindex=False, floatfmt=".4f") + "\n\n")
            if not iso_df.empty:
                f.write(f"--- ISO-RATIO: time to match {self.args.baseline_algo} knee-point compression ---\n")
                f.write(tabulate(iso_df, headers="keys",
                                 tablefmt="grid", showindex=False, floatfmt=".4f") + "\n\n")
            f.write(f"--- SIGNIFICANCE vs {self.args.baseline_algo} ---\n")
            f.write(tabulate(summary, headers="keys", tablefmt="grid",
                             showindex=False, floatfmt=".4f") + "\n")

        # Write raw results to DB
        if self.db_conn:
            param_cols = [c for c in raw_df.columns if c.startswith("param_")]
            for _, row in raw_df.iterrows():
                params = {c.replace("param_", ""): row[c] for c in param_cols if not pd.isna(row.get(c))}
                db.write_result(
                    self.db_conn,
                    algorithm=row["Algorithm"], dataset=row["Dataset"],
                    time=row.get("Time"), ratio=row.get("Ratio"),
                    rep=int(row.get("Rep", 0)),
                    params=params,
                )


if __name__ == "__main__":
    SignificanceTestBenchmark().run()
