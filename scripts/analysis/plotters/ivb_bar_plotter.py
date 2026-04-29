import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis.plotters.plotter import Plotter, register

@register
class IvbBarPlotter(Plotter):
    plot_id = "ivb_bar"
    description = "Logarithmic Bar Chart (Streaming vs Batch)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        colors = [academic_colors[i % len(academic_colors)] for i in range(len(algos))]

        fig, ax = plt.subplots(figsize=(8, 6))
        data = data.set_index("algorithm")

        data[["time_micros"]].plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=1.2, legend=False)
        ax.set_title(f"Streaming vs Batch - {title_prefix}", fontweight="bold")
        ax.set_ylabel("Execution Time (µs)", fontweight="bold")
        ax.set_xlabel("Algorithm", fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        return fig