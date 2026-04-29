import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.analysis.plotters.plotter import Plotter, register

@register
class SweepLinePlotter(Plotter):
    plot_id = "sweep_line"
    description = "Parameter Sensitivity Line Charts (Time & Ratio)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        if "param" not in data.columns:
            raise ValueError("Missing 'param' column. Please re-run the Sweep experiment.")

        # Grab the dynamic parameter name we passed from analyze.py
        param_name = options.get("param_name", "Parameter").upper()

        academic_colors = ["#2B2D42", "#8D99AE", "#CCCCCC", "#4A536B", "#111111", "#A8B2C1"]
        n_algos = len(data["algorithm"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Chart A: Execution Time vs Param
        sns.lineplot(
            data=data, x="param", y="time", hue="algorithm", style="algorithm",
            markers=True, dashes=False, palette=academic_colors[:n_algos],
            linewidth=2.5, markersize=10, ax=axes[0]
        )
        axes[0].set_title(f"Execution Time vs {param_name}", fontweight="bold", fontsize=14)
        axes[0].set_xlabel(f"{param_name} Value", fontweight="bold")
        axes[0].set_ylabel(time_label, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.7)

        # Chart B: Compression Ratio vs Param
        sns.lineplot(
            data=data, x="param", y="ratio", hue="algorithm", style="algorithm",
            markers=True, dashes=False, palette=academic_colors[:n_algos],
            linewidth=2.5, markersize=10, ax=axes[1], legend=False
        )
        axes[1].set_title(f"Compression Ratio vs {param_name}", fontweight="bold", fontsize=14)
        axes[1].set_xlabel(f"{param_name} Value", fontweight="bold")
        axes[1].set_ylabel("Compression Ratio (lower is better)", fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.7)

        fig.suptitle(f"Parameter Sensitivity - {title_prefix}", fontweight="bold", fontsize=16)
        fig.tight_layout()

        return fig