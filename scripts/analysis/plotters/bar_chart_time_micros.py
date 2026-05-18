from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_micros"
    description = "Bar chart comparing Execution Time (Batch vs Streaming)"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        fig, ax = plt.subplots(figsize=(6, 3.5))

        palette = [self.get_algo_style(algo)["color"] for algo in algos]
        ordered_datasets = self.get_dataset_order(data)

        sns.barplot(
            data=data,
            x="dataset",
            y="time_micros",
            hue="algorithm",
            hue_order=algos, # Enforce order to match our palette
            order=ordered_datasets,
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=1.2,
            ax=ax
        )

        plt.xlabel("")
        ax.set_yscale("log", base=10)
        ax.set_ylim(bottom=1)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        plt.ylabel("Execution Time\n(microseconds)", fontsize=14, style='italic')

        plt.legend(
            title="",
            bbox_to_anchor=(0.5, 1.15),
            loc='upper center',
            ncol=len(algos),
            frameon=False
        )

        plt.tight_layout()

        suffix = "_".join(algos)
        png_path = out_dir / f"execution_time_{suffix}.png"

        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

        return [png_path]