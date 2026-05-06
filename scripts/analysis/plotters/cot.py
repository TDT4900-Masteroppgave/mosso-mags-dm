# scripts/analysis/analyzers/cot_line_analyzer.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base_plotter import Plotter, register

@register
class CotLinePlotter(Plotter):
    analyzer_id = "cot_line"
    description = "Progression Line Chart (Compression over time)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        if "change_ratio" not in data.columns:
            raise ValueError("Missing 'change_ratio' column. Please re-run the COT experiment.")

        title_prefix = context.title()
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        n_algos = len(data["algorithm"].unique())

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.lineplot(
            data=data,
            x="change_ratio",
            y="ratio",
            hue="algorithm",
            style="algorithm",
            markers=True,
            dashes=False,
            palette=academic_colors[:n_algos],
            linewidth=2.5,
            markersize=10,
            ax=ax
        )

        ax.set_title(f"Compression Over Time - {title_prefix}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Fraction of Edge Stream", fontweight="bold")
        ax.set_ylabel("Compression Ratio (lower is better)", fontweight="bold")

        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])

        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()

        out_path = out_dir / f"cot_line_{context}_{ts}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return [out_path]