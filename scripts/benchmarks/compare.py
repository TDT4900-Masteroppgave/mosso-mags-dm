import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from scripts.utils import format_dataframe_with_baseline
from scripts.benchmark import Benchmark
from scripts.stats import bootstrap_ci
import scripts.db as db


class CompareBenchmark(Benchmark):
    def __init__(self):
        super().__init__("compare")
        self.all_times_dict, self.all_ratios_dict = {}, {}
        self.raw_runs: list[dict] = []

    def add_custom_args(self, parser):
        parser.add_argument("--keep-summaries", action="store_true")
        parser.add_argument("--verify", action="store_true",
                            help="Run losslessness check on each algorithm output.")

    def process(self, dataset_path: str, ds: dict, dataset_name: str):
        current_result = {"Dataset": dataset_name}
        ds_meta = ds.get("meta", {})
        current_result["Edges"] = ds_meta.get("edges")

        for algo_name, algo_config in self.active_algos.items():
            resolved_params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list, m_list = self.execute_runner(
                algo_name=algo_name,
                algo_config=algo_config,
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                resolved_params=resolved_params
            )

            if t_avg is not None:
                t_std = np.std(t_list) if len(t_list) > 1 else 0.0
                r_std = np.std(r_list) if len(r_list) > 1 else 0.0
                m_avg = np.mean(m_list) if m_list else None
                m_std = np.std(m_list) if len(m_list) > 1 else 0.0

                # Bootstrap CIs (only meaningful when runs > 1)
                t_lo, t_hi = bootstrap_ci(t_list, seed=self.args.seed) if len(t_list) > 1 else (None, None)
                r_lo, r_hi = bootstrap_ci(r_list, seed=self.args.seed) if len(r_list) > 1 else (None, None)

                current_result[f"Time_{algo_name}"] = t_avg
                current_result[f"Ratio_{algo_name}"] = r_avg
                current_result[f"Time_std_{algo_name}"] = t_std
                current_result[f"Ratio_std_{algo_name}"] = r_std
                current_result[f"Time_ci_lo_{algo_name}"] = t_lo
                current_result[f"Time_ci_hi_{algo_name}"] = t_hi
                current_result[f"Ratio_ci_lo_{algo_name}"] = r_lo
                current_result[f"Ratio_ci_hi_{algo_name}"] = r_hi
                if m_avg is not None:
                    current_result[f"Memory_avg_{algo_name}"] = m_avg
                    current_result[f"Memory_std_{algo_name}"] = m_std

                if self.args.runs > 1:
                    ci_str = f" CI=[{t_lo:.3f},{t_hi:.3f}]" if t_lo is not None else ""
                    self.logger.info(f"\t=> {algo_name: <12} Time: {t_avg:.3f}s ± {t_std:.3f}s{ci_str} | Ratio: {r_avg:.5f} ± {r_std:.5f}")
                    self.all_times_dict[algo_name] = t_list
                    self.all_ratios_dict[algo_name] = r_list
                else:
                    mem_str = f" | Memory: {m_avg:.1f}MB" if m_avg is not None else ""
                    self.logger.info(f"\t=> {algo_name: <12} Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}{mem_str}")

                # Store raw per-run data for DB
                for rep_i, (t_r, r_r) in enumerate(zip(t_list, r_list), 1):
                    m_r = m_list[rep_i - 1] if m_list and rep_i - 1 < len(m_list) else None
                    self.raw_runs.append({
                        "algo": algo_name, "dataset": dataset_name,
                        "rep": rep_i, "time": t_r, "ratio": r_r, "memory_mb": m_r,
                    })

                if self.args.verify:
                    self._run_verify(algo_name, dataset_path, dataset_name)

        self.results.append(current_result)

    def _run_verify(self, algo_name: str, dataset_path: str, dataset_name: str):
        """Spot-check losslessness if --verify is set."""
        try:
            from scripts.correctness import verify_lossless
            summary_path = self.summaries_dir / f"{algo_name}_{dataset_name}_{self.timestamp}"
            if not summary_path.exists():
                self.logger.warning(f"[verify] No summary found for {algo_name}/{dataset_name}; skipping.")
                return
            result = verify_lossless(str(summary_path), dataset_path, self.logger)
            self.logger.info(f"[verify] {algo_name}/{dataset_name}: {result}")
        except Exception as e:
            self.logger.warning(f"[verify] Check failed for {algo_name}/{dataset_name}: {e}")

    def print_table(self):
        df = pd.DataFrame(self.results)
        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_") and not col.startswith("Time_ci")]

        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row['Dataset'] = 'AVERAGE'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        display_df = format_dataframe_with_baseline(df, strategies, self.args.baseline)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
        for line in table_str.split('\n'):
            self.logger.info(line)

    def finalize(self):
        csv_file = os.path.join(self.session_dir, "results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)

        strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_") and not col.startswith("Time_ci")]
        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row['Dataset'] = 'AVERAGE'
        table_df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        display_df = format_dataframe_with_baseline(table_df, strategies, self.args.baseline)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)

        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_str)

        # Write raw per-run data to DB
        if self.db_conn:
            for row in self.raw_runs:
                db.write_result(
                    self.db_conn,
                    algorithm=row["algo"], dataset=row["dataset"],
                    time=row["time"], ratio=row["ratio"], memory_mb=row["memory_mb"],
                    rep=row["rep"],
                )


if __name__ == "__main__":
    CompareBenchmark().run()
