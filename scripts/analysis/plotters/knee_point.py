# scripts/analysis/analyzers/knee_point.py
import pandas as pd
import numpy as np
from pathlib import Path

from rich import box
from rich.table import Table
from rich.console import Console
from .base_plotter import Plotter, register
from scripts.config import PARAM_CONFIG

console = Console()

@register
class KneePointPlotter(Plotter):
    analyzer_id = "bayesian_study"
    description = "Optimal Knee Point Summary (CSV/Console)"

    def __init__(self):
        super().__init__()
        self.generates_data = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        time_weight = 0.305
        ratio_weight = 1.0 - time_weight

        results = []
        groups = data['dataset'].unique() if 'dataset' in data.columns else ["Average"]

        for ds in groups:
            for algo in algos:
                df_sub = data[(data['algorithm'] == algo) & (data['dataset'] == ds)].copy() if 'dataset' in data.columns else data[data['algorithm'] == algo].copy()
                if df_sub.empty: continue

                df_sub = df_sub.sort_values(by=['time', 'ratio'])
                pareto_mask = []
                best_ratio = float('inf')

                for r in df_sub['ratio']:
                    if r < best_ratio:
                        best_ratio = r
                        pareto_mask.append(True)
                    else:
                        pareto_mask.append(False)

                pareto_front = df_sub[pareto_mask].copy()
                if pareto_front.empty: continue

                times, ratios = pareto_front['time'].values, pareto_front['ratio'].values
                t_range = (times.max() - times.min()) or 1.0
                r_range = (ratios.max() - ratios.min()) or 1.0

                t_norm = (times - times.min()) / t_range
                r_norm = (ratios - ratios.min()) / r_range

                # ... inside the inner loop, right after finding knee_row ...
                distances = np.sqrt(time_weight * (t_norm ** 2) + ratio_weight * (r_norm ** 2))
                knee_row = pareto_front.iloc[np.argmin(distances)]

                # Consolidate and FORMAT parameters
                params = {}
                for k in PARAM_CONFIG.keys():
                    if k in knee_row.index and pd.notna(knee_row[k]):
                        val = knee_row[k]
                        # If it's effectively an integer (like c=120.0), print without decimals
                        if isinstance(val, float) and val.is_integer():
                            params[k] = int(val)
                        # Otherwise format to 2 decimal places
                        elif isinstance(val, float):
                            params[k] = round(val, 2)
                        else:
                            params[k] = val

                results.append({
                    "Dataset": ds,
                    "Algorithm": algo,
                    "Knee Time": knee_row["time"],
                    "Knee Ratio": knee_row["ratio"],
                    "Parameters": ", ".join([f"{k}={v}" for k, v in params.items()])
                })

        # Generate Console Output
        table = Table(box=box.SIMPLE, header_style="bold cyan")
        table.add_column("Dataset"); table.add_column("Algorithm"); table.add_column("Knee Time"); table.add_column("Knee Ratio"); table.add_column("Params")
        for r in results:
            table.add_row(str(r["Dataset"]), str(r["Algorithm"]), f"{r['Knee Time']:.5f}", f"{r['Knee Ratio']:.5f}", r["Parameters"])
        console.print(f"\n[bold magenta]Knee Point Summary (W_t={time_weight:.2f}, W_r={ratio_weight:.2f})[/]")
        console.print(table)

        # Generate CSV Output
        results_df = pd.DataFrame(results)
        out_path = out_dir / f"knee_points_{context}_{ts}.csv"
        results_df.to_csv(out_path, index=False)

        return [out_path]