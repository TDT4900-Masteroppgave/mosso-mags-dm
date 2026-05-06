# scripts/analysis/analyzers/ivb_bar_analyzer.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .base_plotter import Plotter, register

@register
class IvbBarPlotter(Plotter):
    analyzer_id = "ivb_bar"
    description = "Logarithmic Bar Chart (Streaming vs Batch)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        title_prefix = context.title()
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        colors = [academic_colors[i % len(academic_colors)] for i in range(len(algos))]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Avoid SettingWithCopyWarning by explicitly copying if we modify
        plot_data = data.set_index("algorithm")

        plot_data[["time_micros"]].plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=1.2, legend=False)
        ax.set_title(f"Streaming vs Batch - {title_prefix}", fontweight="bold")
        ax.set_ylabel("Execution Time (µs)", fontweight="bold")
        ax.set_xlabel("Algorithm", fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()

        out_path = out_dir / f"ivb_bar_{context}_{ts}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return [out_path]