from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from scripts.analysis.plotters.base_plotter import Plotter, register


def _annotate_pairwise_ratios(ax, data: pd.DataFrame, algos: list[str], ordered_datasets: list[str]) -> None:
    if len(algos) < 2:
        return

    # Pivot data to get medians for all active algorithms
    values = data.pivot_table(
        index="dataset",
        columns="algorithm",
        values="time_micros",
        aggfunc="median",
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
        suffix = "_".join(algos)

        # 1. Generate the statistics base DataFrame
        raw_stats = data.groupby(["dataset", "algorithm"])["time_micros"].agg(
            Mean='mean',
            Median='median',
            Min='min',
            Max='max',
            StdDev='std',
            Count='count'
        ).reset_index()

        # Build the table text string dynamically for both terminal and disk storage
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"STATISTICS SUMMARY FOR ALGORITHMS: {', '.join(algos)}")
        report_lines.append("=" * 80)

        for algo in algos:
            algo_stats = raw_stats[raw_stats["algorithm"] == algo].copy()
            if algo_stats.empty:
                continue

            report_lines.append(f"\n▶ Algorithm: {algo}")
            report_lines.append(f"{'Dataset':<15} {'Mean':<10} {'Med.':<10} {'Min':<10} {'Max':<10} {'Std.':<10}")
            report_lines.append("-" * 65)

            # Append rows for each dataset
            for _, row in algo_stats.iterrows():
                report_lines.append(
                    f"{row['dataset']:<15} {row['Mean']:.2e} {row['Median']:.2e} "
                    f"{row['Min']:.2e} {row['Max']:.2e} {row['StdDev']:.2e}"
                )

            report_lines.append("-" * 65)

            # Option A: Geometric Mean
            geom_mean_mean = np.exp(np.mean(np.log(algo_stats['Mean'])))
            geom_mean_med = np.exp(np.mean(np.log(algo_stats['Median'])))
            report_lines.append(f"{'Opt A: Geom Mean':<15} {geom_mean_mean:.2e} {geom_mean_med:.2e} {'--':<10} {'--':<10} {'--':<10}")

            # Option B: Global Pool metrics
            algo_data = data[data["algorithm"] == algo]
            global_min = algo_data['time_micros'].min()
            global_max = algo_data['time_micros'].max()

            n_minus_1 = algo_stats['Count'] - 1
            sum_n_minus_1 = np.sum(n_minus_1)
            if sum_n_minus_1 > 0:
                pooled_variance = np.sum(n_minus_1 * (algo_stats['StdDev'] ** 2)) / sum_n_minus_1
                pooled_std = np.sqrt(pooled_variance)
            else:
                pooled_std = 0.0

            report_lines.append(f"{'Opt B: Global':<15} {'--':<10} {'--':<10} {global_min:.2e} {global_max:.2e} {pooled_std:.2e}")

        report_lines.append("=" * 80 + "\n")

        # Combine the string matrix
        full_report_text = "\n".join(report_lines)

        # Print to terminal
        print(full_report_text)

        # Save exact text printout layout to file
        txt_path = out_dir / f"execution_time_stats_{suffix}.txt"
        txt_path.write_text(full_report_text, encoding="utf-8")

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
            estimator=np.median,
            errorbar=None,
            edgecolor="black",
            linewidth=1.2,
            ax=ax
        )

        plt.xlabel("")
        ax.set_yscale("log", base=10)
        ax.set_ylim(bottom=1)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        # Execute repaired dynamic annotation engine
        _annotate_pairwise_ratios(ax, data, algos, ordered_datasets)

        plt.ylabel("Execution Time\n(microseconds)", fontsize=14, style='italic')

        plt.legend(
            title="",
            bbox_to_anchor=(0.5, 1.15),
            loc='upper center',
            ncol=len(algos),
            frameon=False
        )

        plt.tight_layout()

        png_path = out_dir / f"execution_time_{suffix}.png"
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

        return [png_path, txt_path]