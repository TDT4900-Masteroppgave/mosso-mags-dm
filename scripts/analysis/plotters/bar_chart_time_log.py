from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
from rich.console import Console
from rich.table import Table

from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register


BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.5


@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_log"
    description = "Bar chart comparing time (logarithmic scale)"
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
        profile = self.use_figure_profile(self.FIGURE_PROFILE, options, profile_name="paper_bar")

        ordered_datasets = self.get_dataset_order(data)
        metadata = self.get_dataset_metadata()

        stats_df = data.groupby(["dataset", "algorithm"]).agg(
            time_mean=("time", "mean"),
            time_std=("time", "std"),
            time_cv=("time", self.cv),
        ).reset_index()

        table_ds = Table(title="Runtime Statistics (Log Scale Plotter)", show_header=True, show_lines=True)
        table_ds.add_column("Dataset", style="cyan")
        table_ds.add_column("Nodes", style="dim")
        table_ds.add_column("Edges", style="dim")
        table_ds.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        table_ds.add_column("Time Mean (s)")
        table_ds.add_column("Time StdDev (s)")
        table_ds.add_column("Time CV (%)", style="green")

        for ds in ordered_datasets:
            ds_data = stats_df[stats_df["dataset"] == ds]
            if ds_data.empty:
                continue

            ds_meta = metadata.get(ds, {})
            nodes = str(ds_meta.get("nodes", "-"))
            edges = str(ds_meta.get("edges", "-"))

            for _, row in ds_data.iterrows():
                algo = str(row["algorithm"])
                if algo not in algos:
                    continue

                t_mean = f"{row['time_mean']:.3f}".rstrip("0").rstrip(".") if pd.notnull(row["time_mean"]) else "-"
                t_std = f"{row['time_std']:.3f}".rstrip("0").rstrip(".") if pd.notnull(row["time_std"]) else "-"
                t_cv = self.format_percent(row["time_cv"])
                table_ds.add_row(ds, nodes, edges, algo, t_mean, t_std, t_cv)

        table_gm = Table(title="Geometric Means and Stability Summary", show_header=True)
        table_gm.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        table_gm.add_column("Time GeoMean (s)")
        table_gm.add_column("Median Time CV (%)")
        table_gm.add_column("IQR Time CV (%)")
        table_gm.add_column("Max Time CV (%)")

        for algo in algos:
            algo_data = stats_df[stats_df["algorithm"] == algo]
            if algo_data.empty:
                continue

            t_vals = algo_data["time_mean"].dropna()
            t_gm = gmean(t_vals) if not t_vals.empty else float("nan")
            time_cv_vals = algo_data["time_cv"].dropna()

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
                f"{t_gm:.2f}" if pd.notnull(t_gm) else "-",
                *cv_summary,
            )

        txt_path = out_dir / f"time_compression_log_stats.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            file_console = Console(file=f, width=220)
            file_console.print(table_ds)
            file_console.print()
            file_console.print(table_gm)

        fig, ax = self.create_figure(profile)

        palette = [self.get_algo_style(algo)["color"] for algo in algos]

        sns.barplot(
            data=data,
            x="dataset",
            y="time",
            hue="algorithm",
            hue_order=algos,
            order=ordered_datasets,
            palette=palette,
            errorbar=None,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            ax=ax
        )

        ax.set_yscale("log", base=10)

        self.style_paper_bar_chart(ax, ylabel="time (seconds)", algos=algos)
        fig.tight_layout(pad=0.4)

        pdf_path = out_dir / "runtime_bar_chart_log.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        return [pdf_path, txt_path]
