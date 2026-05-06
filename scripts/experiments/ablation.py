"""Ablation study: turn each new improvement parameter to a neutralizing value
and measure the delta vs the full strategy.

Each algorithm runs once with `--ablate` set to "off" (every listed param at its
neutralizing value simultaneously, isolating the strategy's overall contribution),
then once per individual ablated parameter (others at full config), then once
"full" with all parameters at their tuned/default values.

Specify ablations as `--ablate <param>=<value>` (repeatable).
Common neutralizers:
  --ablate b=10       (top-b → uniform-like)
  --ablate h=4        (hashes → base MoSSo count)
  --ablate cap=240    (cap → effectively unlimited)
  --ablate thr_end=0  (threshold disabled)
  --ablate T=1        (single partition pass)
"""
import pandas as pd
from rich import box
from rich.table import Table

from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment


class Ablation(Experiment):
    def __init__(self):
        super().__init__("ablation")

    def add_custom_args(self, parser):
        parser.add_argument("--ablate", action="append", default=[],
                            metavar="PARAM=VALUE",
                            help="Param to neutralize and its degenerate value. Repeatable.")
        parser.add_argument("--combined-only", action="store_true",
                            help="Only run full vs all-ablated-together (skip per-param ablations).")

    def _parse_ablations(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in self.args.ablate:
            if "=" not in spec:
                self.logger.warning(f"Bad --ablate spec '{spec}', expected PARAM=VALUE")
                continue
            k, v = spec.split("=", 1)
            k = k.strip()
            if k not in PARAM_CONFIG:
                self.logger.warning(f"Unknown param '{k}' in --ablate")
                continue
            out[k] = v.strip()
        return out

    def _run_variant(self, algo_name: str, algo_config: dict, dataset_path: str,
                     dataset_short_name: str, label: str, overrides: dict[str, str]) -> list[dict]:
        params = self._resolve_algo_params(algo_config)
        params.update(overrides)
        result = self.execute_runner(
            algo_name=algo_name, dataset_path=dataset_path,
            params=params, dataset_short_name=dataset_short_name,
        )
        if result is None:
            return []
        _, _, t_list, r_list = result
        if t_list is None or r_list is None:
            return []
        rows = []
        for i, (t, r) in enumerate(zip(t_list, r_list)):
            rows.append({
                "dataset": dataset_short_name,
                "algorithm": algo_name,
                "variant": label,
                "run": i + 1,
                "time": t,
                "ratio": r,
                **params,
            })
        return rows

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        ablations = self._parse_ablations()
        if not ablations:
            self.logger.warning("[!] No --ablate flags given; nothing to ablate.")
            return []

        metrics: list[dict] = []
        for algo_name, algo_config in self.active_algos.items():
            template = set(algo_config.get("template", []))
            applicable = {k: v for k, v in ablations.items() if k in template}
            if not applicable:
                self.logger.info(f"=> Skipping {algo_name}: none of {list(ablations.keys())} in its template.")
                continue

            self.logger.print(f"[bold cyan]Ablating {algo_name} on {dataset_short_name}[/] "
                              f"| params={list(applicable.keys())}")

            metrics.extend(self._run_variant(algo_name, algo_config, dataset_path,
                                             dataset_short_name, "full", overrides={}))
            metrics.extend(self._run_variant(algo_name, algo_config, dataset_path,
                                             dataset_short_name, "all_ablated", overrides=applicable))

            if not self.args.combined_only and len(applicable) > 1:
                for p, v in applicable.items():
                    metrics.extend(self._run_variant(algo_name, algo_config, dataset_path,
                                                     dataset_short_name, f"no_{p}", overrides={p: v}))
        return metrics

    def output(self, df: pd.DataFrame):
        if df.empty:
            return
        means = df.groupby(["dataset", "algorithm", "variant"], as_index=False)[["time", "ratio"]].mean()
        full = means[means["variant"] == "full"][["dataset", "algorithm", "time", "ratio"]] \
            .rename(columns={"time": "time_full", "ratio": "ratio_full"})
        merged = means.merge(full, on=["dataset", "algorithm"], how="left")
        merged["Δtime"] = merged["time"] - merged["time_full"]
        merged["Δratio"] = merged["ratio"] - merged["ratio_full"]
        merged["%Δtime"] = (merged["Δtime"] / merged["time_full"]) * 100
        merged["%Δratio"] = (merged["Δratio"] / merged["ratio_full"]) * 100

        table = Table(title="Ablation Deltas (vs 'full' variant)", box=box.SIMPLE,
                      header_style="bold yellow",
                      caption="[dim]Positive Δratio means ablation hurt compression. "
                              "Positive Δtime means ablation slowed it.[/]")
        for col in ("Dataset", "Algorithm", "Variant", "Time", "Ratio", "Δtime", "Δratio", "%Δtime", "%Δratio"):
            table.add_column(col)
        for _, r in merged.sort_values(["dataset", "algorithm", "variant"]).iterrows():
            if r["variant"] == "full":
                continue
            table.add_row(
                r["dataset"], r["algorithm"], r["variant"],
                f"{r['time']:.3f}s", f"{r['ratio']:.5f}",
                f"{r['Δtime']:+.3f}", f"{r['Δratio']:+.5f}",
                f"{r['%Δtime']:+.1f}%", f"{r['%Δratio']:+.1f}%",
            )
        self.logger.print(table)


def main():
    with Ablation() as exp:
        exp.run()


if __name__ == "__main__":
    main()
