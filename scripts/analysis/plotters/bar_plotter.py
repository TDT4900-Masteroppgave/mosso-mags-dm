import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.analysis.plotters.plotter import Plotter, register

@register
class BarPlotter(Plotter):
    plot_id = "bar"
    description = "Bar chart (Time & Ratio)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        colors = [academic_colors[i % len(academic_colors)] for i in range(len(algos))]

        # Detect if we are plotting a Combined dataset view, or a single/average view
        has_multiple_ds = "dataset" in data.columns and len(data["dataset"].unique()) > 1
        x_col = "dataset" if has_multiple_ds else "algorithm"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # We use Seaborn's 'hue' to automatically group the bars!
        sns.barplot(data=data, x=x_col, y="ratio", hue="algorithm", ax=axes[0], palette=colors, edgecolor="black", linewidth=1.2)
        axes[0].set_title(f"Compression Ratio - {title_prefix}", fontweight="bold")
        axes[0].set_ylabel("Ratio", fontweight="bold")

        sns.barplot(data=data, x=x_col, y="time", hue="algorithm", ax=axes[1], palette=colors, edgecolor="black", linewidth=1.2)
        axes[1].set_title(f"{time_label} - {title_prefix}", fontweight="bold")
        axes[1].set_ylabel(time_label, fontweight="bold")

        # Clean up redundant legends
        if axes[1].get_legend(): axes[1].get_legend().remove()
        if not has_multiple_ds and axes[0].get_legend(): axes[0].get_legend().remove()

        for ax in axes:
            ax.set_xlabel("Dataset" if has_multiple_ds else "Algorithm", fontweight="bold")
            ax.tick_params(axis="x", rotation=0 if has_multiple_ds else 30)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        return fig