import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scripts.analysis.plotters.plotter import Plotter, register

@register
class BoScatterPlotter(Plotter):
    plot_id = "bo_scatter"
    description = "Pareto Optimization Cloud (Optuna Trials)"

    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        if "trial" not in data.columns:
            raise ValueError("Missing 'trial' column. Please re-run the BO experiment.")

        # We will create one subplot per algorithm to clearly see their distinct parameter spaces
        n_algos = len(algos)
        fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 6), squeeze=False)
        axes = axes.flatten()

        for idx, algo in enumerate(algos):
            ax = axes[idx]
            algo_data = data[data["algorithm"] == algo].copy()

            if algo_data.empty:
                ax.set_visible(False)
                continue

            # 1. Plot the "Cloud" of all trials
            sns.scatterplot(
                data=algo_data, x="time", y="ratio",
                color="#8D99AE", alpha=0.6, s=100, ax=ax, edgecolor="w"
            )

            # 2. Calculate and highlight the Knee Point mathematically
            if len(algo_data) > 1:
                t_min, t_max = algo_data['time'].min(), algo_data['time'].max()
                r_min, r_max = algo_data['ratio'].min(), algo_data['ratio'].max()

                algo_data['t_norm'] = (algo_data['time'] - t_min) / (t_max - t_min + 1e-9)
                algo_data['r_norm'] = (algo_data['ratio'] - r_min) / (r_max - r_min + 1e-9)
                algo_data['distance'] = np.sqrt(algo_data['t_norm']**2 + algo_data['r_norm']**2)

                knee_row = algo_data.loc[algo_data['distance'].idxmin()]

                # Plot the Knee Point as a giant Red Star
                ax.scatter(
                    knee_row["time"], knee_row["ratio"],
                    color="#D90429", marker="*", s=500, edgecolor="black", zorder=5, label="Optimal Knee Point"
                )

                ax.annotate(
                    f"Trial {int(knee_row['trial'])}",
                    (knee_row["time"], knee_row["ratio"]),
                    xytext=(10, 10), textcoords='offset points',
                    fontweight='bold', color='#D90429'
                )
                ax.legend(loc="upper right")

            ax.set_title(f"{algo} Parameter Space - {title_prefix}", fontsize=14, fontweight="bold")
            ax.set_xlabel(time_label, fontsize=12, fontweight="bold")
            ax.set_ylabel("Compression Ratio (lower is better)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.7)

            # Add optimal direction watermark
            ax.text(0.05, 0.05, 'Optimal Region $\\swarrow$',
                    transform=ax.transAxes, color='#4A536B', alpha=0.6,
                    fontsize=12, fontweight='bold', va='bottom', ha='left')

        fig.tight_layout()
        return fig