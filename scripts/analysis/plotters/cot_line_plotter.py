import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.analysis.plotters.plotter import Plotter, register

@register
class CotLinePlotter(Plotter):
    plot_id = "cot_line"
    description = "Progression Line Chart (Compression over time)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        if "change_ratio" not in data.columns:
            raise ValueError("Missing 'change_ratio' column. Please re-run the COT experiment.")

        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        n_algos = len(data["algorithm"].unique())

        fig, ax = plt.subplots(figsize=(8, 6))

        # Seaborn naturally connects the dots for our time-series lines
        sns.lineplot(
            data=data,
            x="change_ratio",
            y="ratio",
            hue="algorithm",
            style="algorithm", # Gives different markers (circle, square, X) to each line
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

        # Format X-axis to show percentages exactly like MoSSo Figure 5
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])

        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()

        return fig