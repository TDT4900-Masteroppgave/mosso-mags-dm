# scripts/analysis/analyzers/marginal_utility.py
import pandas as pd
from pathlib import Path
from rich.table import Table
from rich.console import Console
from rich import box

from .base_plotter import Plotter, register
from scripts.config import PARAM_CONFIG

console = Console()

@register
class MarginalUtilityPlotter(Plotter):
    analyzer_id = "marginal_utility"
    description = "Diminishing Returns / Marginal Cost Analysis"

    def __init__(self):
        super().__init__()
        self.generates_data = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        results = []
        groups = data['dataset'].unique() if 'dataset' in data.columns else ["Average"]

        for ds in groups:
            for algo in algos:
                df_sub = data[(data['algorithm'] == algo) & (data['dataset'] == ds)].copy() if 'dataset' in data.columns else data[data['algorithm'] == algo].copy()
                if df_sub.empty: continue

                # 1. Bygg Pareto-fronten først (sortert etter tid)
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
                if pareto_front.empty or len(pareto_front) < 2: continue

                # 2. Sorter fra dårligst kompresjon (høy ratio) til best kompresjon (lav ratio)
                # Dette gjør at vi kan se "kostnaden" av å bevege oss til et bedre punkt
                pareto_front = pareto_front.sort_values(by='ratio', ascending=False).reset_index(drop=True)

                # 3. Regn ut prosentvis endring mellom punktene
                # pct_change() ser på forrige rad.
                pareto_front['pct_change_time'] = pareto_front['time'].pct_change()
                pareto_front['pct_change_ratio'] = pareto_front['ratio'].pct_change().abs() # Vi tar abs() siden ratio går ned (blir bedre)

                # 4. Regn ut Marginal Cost (Hvor mye tid (i %) koster 1% kompresjon?)
                pareto_front['marginal_cost'] = pareto_front['pct_change_time'] / pareto_front['pct_change_ratio']

                # 5. Finn grensen for Diminishing Returns
                # Vi leter etter det punktet hvor marginal cost "eksploderer" (for eksempel over 5.0)
                # Hvis vi ikke finner et ekstremt hopp, velger vi det punktet med høyest økning før grafen flater ut

                table = Table(box=box.SIMPLE, header_style="bold cyan", title=f"Diminishing Returns: {algo} ({ds})")
                table.add_column("Ratio", justify="right")
                table.add_column("Time", justify="right")
                table.add_column("Δ Ratio %", justify="right")
                table.add_column("Δ Time %", justify="right")
                table.add_column("Marginal Cost", justify="right", style="bold red")
                table.add_column("Params")

                for i, row in pareto_front.iterrows():
                    params = {k: row[k] for k in PARAM_CONFIG.keys() if k in row.index}
                    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])

                    if i == 0:
                        table.add_row(f"{row['ratio']:.5f}", f"{row['time']:.6f}", "-", "-", "-", param_str)
                    else:
                        cost_str = f"{row['marginal_cost']:.2f}x"
                        # Marker store hopp visuelt!
                        if row['marginal_cost'] > 5.0:
                            cost_str = f"[bold yellow]{cost_str} (SPIKE!)[/]"

                        table.add_row(
                            f"{row['ratio']:.5f}",
                            f"{row['time']:.6f}",
                            f"{row['pct_change_ratio']*100:.2f}%",
                            f"{row['pct_change_time']*100:.2f}%",
                            cost_str,
                            param_str
                        )

                console.print(table)
                results.append(pareto_front)

        # Lagre rådata til fil så dere har den til oppgaven
        if results:
            final_df = pd.concat(results)
            out_path = out_dir / f"marginal_utility_{context}_{ts}.csv"
            final_df.to_csv(out_path, index=False)
            return [out_path]
        return []