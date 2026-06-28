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

from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register


RATIO_ANNOTATION_SIZE = 18.0


def _annotate_pairwise_ratios(ax, data: pd.DataFrame, algos: list[str], ordered_datasets: list[str], fontsize: float) -> None:
    if len(algos) < 2:
        return

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

        dataset_vals = values.loc[dataset, [a for a in algos if a in values.columns]].dropna()
        if dataset_vals.empty or len(dataset_vals) < 2:
            continue

        low = dataset_vals.min()
        high = dataset_vals.max()

        if low <= 0 or high <= 0:
            continue

        magnitude = math.log10(high / low)
        global_y_max = max(global_y_max, high)

        low_algo = dataset_vals.idxmin()
        high_algo = dataset_vals.idxmax()

        n_bars = len(algos)
        bar_width = 0.8 / n_bars

        pos_low = idx - 0.4 + (algos.index(low_algo) * bar_width) + (bar_width / 2)
        pos_high = idx - 0.4 + (algos.index(high_algo) * bar_width) + (bar_width / 2)

        arrow_x = pos_low
        label_x = arrow_x - 0.04

        ax.hlines(high, min(pos_high, arrow_x), max(pos_high, arrow_x), colors="black", linestyles=(0, (1, 3)))
        ax.hlines(low, min(pos_low, arrow_x), max(pos_low, arrow_x), colors="black", linestyles=(0, (1, 3)))

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

        ax.text(
            label_x,
            (low * high) ** 0.5,
            rf"$10^{{{magnitude:.1f}}}$",
            rotation=90,
            ha="right",
            va="center",
            color="black",
            fontsize=fontsize,
            )

    ax.set_ylim(top=global_y_max * 10)


@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_micros"
    description = "Bar chart comparing Execution Time (Batch vs Streaming) with stats table"
    FIGURE_PROFILE = FigureProfile(
        figsize=(10.8, 6.075),
        font_size=35.0,
        label_size=30.0,
        tick_size=30.0,
        legend_size=28.0,
        line_width=3.5,
        marker_size=10.0,
        extra_rc={
            "axes.linewidth": 2.0,
            "xtick.major.size": 7.0,
            "ytick.major.size": 7.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
        },
    )

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        profile = self.use_figure_profile(self.FIGURE_PROFILE, options, profile_name="time_micros_bar")
        ordered_datasets = self.get_dataset_order(data)

        metadata = self.get_dataset_metadata()

        stats_df = data.groupby(["dataset", "algorithm"]).agg(
            time_mean=('time_micros', 'mean'),
            time_std=('time_micros', 'std'),
            time_cv=('time_micros', self.cv),
        ).reset_index()

        table_ds = Table(title="Execution Time Statistics (Microseconds)", show_header=True, show_lines=True)
        table_ds.add_column("Dataset", style="cyan")
        table_ds.add_column("Nodes", style="dim")
        table_ds.add_column("Edges", style="dim")
        table_ds.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        table_ds.add_column("Time Mean")
        table_ds.add_column("Time StdDev")
        table_ds.add_column("Time CV (%)", style="green")

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
                t_cv = self.format_percent(row['time_cv'])

                table_ds.add_row(ds, nodes, edges, algo, t_mean, t_std, t_cv)

        table_gm = Table(title="Geometric Means and Stability Summary", show_header=True)
        table_gm.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        table_gm.add_column("Time Mean (GeoMean)")
        table_gm.add_column("Median Time CV (%)")
        table_gm.add_column("IQR Time CV (%)")
        table_gm.add_column("Max Time CV (%)")
        for algo in algos:
            algo_data = stats_df[stats_df['algorithm'] == algo]
            if algo_data.empty: continue

            t_mean_vals = algo_data['time_mean'].dropna()

            t_mean_gm = gmean(t_mean_vals) if not t_mean_vals.empty else np.nan
            time_cv_vals = algo_data['time_cv'].dropna()

            if time_cv_vals.empty:
                cv_summary = ["-", "-", "-"]
            else:
                cv_summary = [
                    self.format_percent(time_cv_vals.median()),
                    self.format_percent(time_cv_vals.quantile(0.75) - time_cv_vals.quantile(0.25)),
                    self.format_percent(time_cv_vals.max()),
                ]

            table_gm.add_row(
                algo,
                f"{t_mean_gm:.3g}" if pd.notnull(t_mean_gm) else "-",
                *cv_summary,
            )

        txt_path = out_dir / f"combined_stats.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            file_console = Console(file=f, width=220)
            file_console.print(table_ds)
            file_console.print()
            file_console.print(table_gm)

        fig, ax = self.create_figure(profile)

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
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_yscale("log", base=10)
        ax.set_ylim(bottom=1)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        _annotate_pairwise_ratios(ax, data, algos, ordered_datasets, RATIO_ANNOTATION_SIZE)

        ax.set_ylabel("Execution Time\n(microseconds)", style='italic')

        self.despine(ax, top=True, right=True)

        self.add_centered_legend(
            ax,
            y=1.17,
            loc='upper center',
            ncol=min(len(algos), 4),
        )

        fig.tight_layout()

        pdf_path = out_dir / f"execution_time.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        return [pdf_path, txt_path]
