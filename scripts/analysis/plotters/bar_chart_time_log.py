from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_log"
    description = "Bar chart comparing execution time logarithmic scale"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()

        fig, ax = plt.subplots(figsize=(6, 3.5))

        palette = [self.get_algo_style(algo)["color"] for algo in algos]

        sns.barplot(
            data=data,
            x="dataset",
            y="time",
            hue="algorithm",
            hue_order=algos,
            order=self.get_dataset_order(data),
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=1.2,
            ax=ax
        )

        plt.xlabel("")
        ax.set_yscale("log", base=10)
        plt.ylabel("time (microseconds)", fontsize=14, style='italic')

        plt.legend(
            title="",
            bbox_to_anchor=(0.5, 1.15),
            loc='upper center',
            ncol=len(algos),
            frameon=False
        )

        plt.tight_layout()

        suffix = "_".join(algos)
        png_path = out_dir / f"runtime_bar_chart_{suffix}.png"
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

        return [png_path]