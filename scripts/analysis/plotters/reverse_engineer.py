# scripts/analysis/analyzers/reverse_engineer.py
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from .base_plotter import Plotter, register

console = Console()

@register
class ReverseEngineerPlotter(Plotter):
    analyzer_id = "reverse_engineer"
    description = "Reverse-Engineer Baseline Implied Weights"

    def __init__(self):
        super().__init__()
        self.generates_data = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        # The exact baseline and parameters we want to reverse-engineer
        target_algo = "kdd20-mosso"
        target_c = 128.0
        target_e = 2.0

        if target_algo not in algos:
            console.print(f"[red]Please select {target_algo} to run this analysis.[/red]")
            return []

        groups = data['dataset'].unique() if 'dataset' in data.columns else ["Average"]

        for ds in groups:
            df_sub = data[(data['algorithm'] == target_algo) & (data['dataset'] == ds)].copy() if 'dataset' in data.columns else data[data['algorithm'] == target_algo].copy()
            if df_sub.empty: continue

            # 1. Build the Pareto Front (exactly as we do in KneePointAnalyzer)
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

            # 2. Normalize the metrics (0.0 to 1.0)
            times, ratios = pareto_front['time'].values, pareto_front['ratio'].values
            t_range = (times.max() - times.min()) or 1.0
            r_range = (ratios.max() - ratios.min()) or 1.0

            t_norm = (times - times.min()) / t_range
            r_norm = (ratios - ratios.min()) / r_range

            # 3. Sweep every possible Time Weight from 0.00 to 1.00
            matching_weights = []

            for w_t in np.arange(0.0, 1.001, 0.001): # Checking 1,000 different weights
                w_r = 1.0 - w_t

                # Calculate distances to ideal (0,0) for this specific weight
                distances = np.sqrt(w_t * (t_norm ** 2) + w_r * (r_norm ** 2))
                knee_idx = np.argmin(distances)
                knee_row = pareto_front.iloc[knee_idx]

                # Check if this weight selected our target parameters
                if knee_row.get('c') == target_c and knee_row.get('e') == target_e:
                    matching_weights.append(w_t)

            # 4. Output the Results
            if matching_weights:
                min_w = min(matching_weights)
                max_w = max(matching_weights)
                median_w = np.median(matching_weights)

                console.print(Panel(
                    f"[bold green]Target Found for {target_algo} ({ds})![/]\n\n"
                    f"Parameters [cyan]c={target_c}, e={target_e}[/] are selected as the optimal Knee Point when:\n"
                    f"Time Weight (W_t) is between: [bold yellow]{min_w:.3f}[/] and [bold yellow]{max_w:.3f}[/]\n\n"
                    f"[bold]Recommended Implied Weight:[/] [bold magenta]W_t = {median_w:.3f}, W_r = {1.0 - median_w:.3f}[/]",
                    title="Reverse-Engineering Complete", border_style="green"
                ))
            else:
                console.print(Panel(
                    f"[red]Target Not Found![/]\n"
                    f"Parameters c={target_c}, e={target_e} are NEVER selected as the optimal knee point mathematically, "
                    f"meaning they are likely off the extreme edges or completely dominated.",
                    title="Reverse-Engineering Failed", border_style="red"
                ))

        return []