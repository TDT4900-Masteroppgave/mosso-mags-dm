# scripts/analysis/analyzers/pareto_front.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .base_plotter import Plotter, register

@register
class ParetoFrontPlotter(Plotter):
    analyzer_id = "pareto_front"
    description = "Pareto Frontier (PNG)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1", "#D90429", "#EF233C"]
        palette = dict(zip(algos, [academic_colors[i % len(academic_colors)] for i in range(len(algos))]))

        fig, ax = plt.subplots(figsize=(8, 6))

        for algo in algos:
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            algo_data = algo_data.sort_values(by=["ratio", "time"])
            pareto_front = []
            min_time = float('inf')

            for _, row in algo_data.iterrows():
                if row["time"] < min_time:
                    pareto_front.append(row)
                    min_time = row["time"]

            if pareto_front:
                pareto_df = pd.DataFrame(pareto_front)
                ax.plot(
                    pareto_df["ratio"], pareto_df["time"],
                    color=palette[algo], linestyle="-", linewidth=3, zorder=4, label=algo
                )

        time_label = options.get("time_label", "Time (s)")
        ax.set_title(f"Pareto Frontier - {context.title()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Compression Ratio (lower is better)", fontsize=12, fontweight="bold")
        ax.set_ylabel(time_label, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper right", title="Algorithms")

        fig.tight_layout()

        # Save and return artifact path
        out_path = out_dir / f"pareto_front_{context}_{ts}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return [out_path]