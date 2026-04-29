import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.analysis.plotters.plotter import Plotter, register

@register
class ScatterPlotter(Plotter):
    plot_id = "scatter"
    description = "Trade-off Scatter Plot (Pareto Frontier)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        n_algos = len(algos)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.scatterplot(
            data=data, x="time", y="ratio", hue="algorithm", style="algorithm",
            s=250, ax=ax, palette=academic_colors[:n_algos], edgecolor="black", linewidth=1.2
        )

        for _, row in data.iterrows():
            ax.text(
                row["time"] * 1.05, row["ratio"], row["algorithm"],
                horizontalalignment='left', size='10', color='#111111', weight='semibold'
            )

        ax.set_title(f"Pareto Frontier - {title_prefix}", fontsize=14, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Compression Ratio (lower is better)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)

        ax.text(0.05, 0.05, 'Optimal Region\n(Fast & Compact) $\\swarrow$',
                transform=ax.transAxes, color='#4A536B', alpha=0.6,
                fontsize=11, fontweight='bold', va='bottom', ha='left')

        fig.tight_layout()
        return fig