from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_compression"
    description = "Bar chart comparing compression"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        self.set_chart_theme()

        fig, ax = plt.subplots(figsize=(6, 3.5))

        num_algos = len(algos)
        palette = ["#FFFFFF", "#DDDDDD", "#888888", "#000000"]
        if num_algos > len(palette):
            palette = sns.color_palette("Greys", num_algos)

        sns.barplot(
            data=data,
            x="dataset",
            y="ratio",
            hue="algorithm",
            palette=palette,
            errorbar=None,
            edgecolor="black",
            linewidth=1.2,
            ax=ax
        )

        # Apply hatches manually to the containers
        hatches = ['', 'xx', '//', '\\\\', '..', '*']
        for i, bar_group in enumerate(ax.containers):
            hatch = hatches[i % len(hatches)]
            for bar in bar_group:
                bar.set_hatch(hatch)

        plt.xlabel("") # Omit x-label to match paper style

        # Italicized y-label
        plt.ylabel("relative size", fontsize=14, style='italic')

        plt.legend(
            title="",
            bbox_to_anchor=(0.5, 1.15),
            loc='upper center',
            ncol=num_algos,
            frameon=False
        )

        plt.tight_layout()

        png_path = out_dir / f"compression_bar_chart_{ts}.png"
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

        return [png_path]