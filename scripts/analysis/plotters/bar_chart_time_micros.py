from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.stats import gmean
from rich.console import Console
from rich.table import Table

from scripts.analysis.plotters.base_plotter import Plotter, register


def _annotate_pairwise_ratios(ax, data: pd.DataFrame, algos: list[str], ordered_datasets: list[str]) -> None:
    if len(algos) < 2:
        return

    # Pivot data to get means for all active algorithms
    values = data.pivot_table(
        index="dataset",
        columns="algorithm",
        values="time_micros",
        aggfunc="mean",
    )

    global_y_max = 1

    for idx, dataset in enumerate(ordered_datasets):
        if dataset not in values.index:
            continue

        # Extract only the values for the algorithms currently being plotted
        dataset_vals = values.loc[dataset, [a for a in algos if a in values.columns]].dropna()
        if dataset_vals.empty or len(dataset_vals) < 2:
            continue

        # Dynamically find the absolute min and max among ALL plotted algorithms
        low = dataset_vals.min()
        high = dataset_vals.max()

        if low <= 0 or high <= 0:
            continue

        magnitude = math.log10(high / low)
        global_y_max = max(global_y_max, high)

        # Identify which specific bars are the lowest and highest to find their positions
        low_algo = dataset_vals.idxmin()
        high_algo = dataset_vals.idxmax()

        # Seaborn positions grouped bars sequentially around the central index integer (0, 1, 2...)
        n_bars = len(algos)
        bar_width = 0.8 / n_bars  # Seaborn's default total group width is 0.8

        pos_low = idx - 0.4 + (algos.index(low_algo) * bar_width) + (bar_width / 2)
        pos_high = idx - 0.4 + (algos.index(high_algo) * bar_width) + (bar_width / 2)

        # FIX: Align the arrow directly with the center of the lowest value bar
        arrow_x = pos_low
        label_x = arrow_x - 0.04  # Shift text slightly left of the vertical arrow line

        # Draw clean horizontal anchor guidelines from the bar tops to the arrow line
        ax.hlines(high, min(pos_high, arrow_x), max(pos_high, arrow_x), colors="black", linestyles=(0, (1, 3)), linewidth=0.6)
        ax.hlines(low, min(pos_low, arrow_x), max(pos_low, arrow_x), colors="black", linestyles=(0, (1, 3)), linewidth=0.6)

        # Draw the main comparison arrow directly over the low bar
        ax.annotate(
            "",
            xy=(arrow_x, high),
            xytext=(arrow_x, low),
            arrowprops={
                "arrowstyle": "<->",
                "color": "black",
                "lw": 0.9,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            annotation_clip=False,
        )

        # Display the order of magnitude difference factor
        ax.text(
            label_x,
            (low * high) ** 0.5,  # Balanced geometric mean positioning for log-scale vertical centering
            rf"$10^{{{magnitude:.1f}}}$",
            rotation=90,
            ha="right",
            va="center",
            fontsize=8,
            color="black",
            )

    # Set the y-limit comfortably above the absolute highest global value
    ax.set_ylim(top=global_y_max * 10)


@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_micros"
    description = "Bar chart comparing Execution Time (Batch vs Streaming) with stats table"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()
        suffix = "_".join(algos)
        ordered_datasets = self.get_dataset_order(data)

        # 1. Generate the statistics tables using rich
        console = Console(record=True)
        metadata = self.get_dataset_metadata()

        # Group and calculate metrics
        stats_df = data.groupby(["dataset", "algorithm"]).agg(
            time_mean=('time_micros', 'mean'),
            time_std=('time_micros', 'std')
        ).reset_index()

        # --- TABLE 1: Dataset Statistics ---
        table_ds = Table(title="Execution Time Statistics (Microseconds)", show_header=True, show_lines=True)
        table_ds.add_column("Dataset", style="cyan")
        table_ds.add_column("Nodes", style="dim")
        table_ds.add_column("Edges", style="dim")
        table_ds.add_column("Algorithm", style="magenta")
        table_ds.add_column("Time Mean")
        table_ds.add_column("Time StdDev")

        for ds in ordered_datasets:
            ds_data = stats_df[stats_df["dataset"] == ds]
            if ds_data.empty: continue

            ds_meta = metadata.get(ds, {})
            nodes = str(ds_meta.get('nodes', '-'))
            edges = str(ds_meta.get('edges', '-'))

            for _, row in ds_data.iterrows():
                algo = str(row['algorithm'])
                if algo not in algos: continue

                t_mean = f"{row['time_mean']:.3g}" if pd.notnull(row['time_mean']) else "-"
                t_std = f"{row['time_std']:.3g}" if pd.notnull(row['time_std']) else "-"

                table_ds.add_row(ds, nodes, edges, algo, t_mean, t_std)

        console.print(table_ds)
        console.print("\n")

        # --- TABLE 2: Geometric Means Summary ---
        table_gm = Table(title="Geometric Means Summary", show_header=True)
        table_gm.add_column("Algorithm", style="magenta")
        table_gm.add_column("Time Mean (GeoMean)")
        for algo in algos:
            algo_data = stats_df[stats_df['algorithm'] == algo]
            if algo_data.empty: continue

            t_mean_vals = algo_data['time_mean'].dropna()

            t_mean_gm = gmean(t_mean_vals) if not t_mean_vals.empty else np.nan

            table_gm.add_row(
                algo,
                f"{t_mean_gm:.3g}" if pd.notnull(t_mean_gm) else "-",
            )

        console.print(table_gm)

        # Save tables to text file (maintains styling characters)
        txt_path = out_dir / f"combined_stats.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console.export_text())

        # 2. Generate the plot
        fig, ax = plt.subplots(figsize=(6.5, 3.8))

        palette = [self.get_algo_style(algo)["color"] for algo in algos]
        ordered_datasets = self.get_dataset_order(data)

        sns.barplot(
            data=data,
            x="dataset",
            y="time_micros",
            hue="algorithm",
            hue_order=algos,
            order=ordered_datasets,
            palette=palette,
            estimator=np.mean,
            errorbar=None,
            edgecolor="white",
            linewidth=0.8,
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_yscale("log", base=10)
        ax.set_ylim(bottom=1)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        # Execute repaired dynamic annotation engine
        _annotate_pairwise_ratios(ax, data, algos, ordered_datasets)

        ax.set_ylabel("Execution Time\n(microseconds)", fontsize=12, style='italic')

        sns.despine(ax=ax, top=True, right=True)

        ax.legend(
            title="",
            bbox_to_anchor=(0.5, 1.15),
            loc='upper center',
            ncol=len(algos),
            frameon=False,
            fontsize=9
        )

        plt.tight_layout()

        png_path = out_dir / f"execution_time_{suffix}.png"
        pdf_path = out_dir / f"execution_time_{suffix}.pdf"
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        return [png_path, pdf_path, txt_path]