# scripts/analysis/analyzers/bar_analyzer.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base_plotter import Plotter, register

@register
class BarPlotter(Plotter):
    analyzer_id = "bar_chart"
    description = "Bar chart (Time & Ratio)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        time_label = options.get("time_label", "Time (seconds)")
        title_prefix = context.title()

        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        colors = [academic_colors[i % len(academic_colors)] for i in range(len(algos))]

        has_multiple_ds = "dataset" in data.columns and len(data["dataset"].unique()) > 1
        x_col = "dataset" if has_multiple_ds else "algorithm"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=data, x=x_col, y="ratio", hue="algorithm", ax=axes[0], palette=colors, edgecolor="black", linewidth=1.2)
        axes[0].set_title(f"Compression Ratio - {title_prefix}", fontweight="bold")
        axes[0].set_ylabel("Ratio", fontweight="bold")

        sns.barplot(data=data, x=x_col, y="time", hue="algorithm", ax=axes[1], palette=colors, edgecolor="black", linewidth=1.2)
        axes[1].set_title(f"{time_label} - {title_prefix}", fontweight="bold")
        axes[1].set_ylabel(time_label, fontweight="bold")

        if axes[1].get_legend(): axes[1].get_legend().remove()
        if not has_multiple_ds and axes[0].get_legend(): axes[0].get_legend().remove()

        for ax in axes:
            ax.set_xlabel("Dataset" if has_multiple_ds else "Algorithm", fontweight="bold")
            ax.tick_params(axis="x", rotation=0 if has_multiple_ds else 30)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()

        out_path = out_dir / f"bar_chart_{context}_{ts}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return [out_path]