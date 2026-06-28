from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean

from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register
from rich.console import Console
from rich.table import Table


BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.5


@register
class CombinedBarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_compression"
    description = "Bar charts comparing time and compression ratio with stats"
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
        artifacts = []

        metadata = self.get_dataset_metadata()

        has_time = "time" in data.columns
        has_ratio = "ratio" in data.columns

        agg_dict = {}
        if has_time:
            agg_dict['time'] = ['mean', 'std', self.cv]
        if has_ratio:
            agg_dict['ratio'] = ['mean', 'std', self.cv]

        stats_df = data.groupby(["dataset", "algorithm"]).agg(agg_dict)
        stats_df.columns = [f"{col[0]}_{col[1]}" for col in stats_df.columns]
        stats_df = stats_df.reset_index()

        table_ds = Table(title="Compression & Timing Statistics", show_header=True, show_lines=True)
        table_ds.add_column("Dataset", style="cyan")
        table_ds.add_column("Nodes", style="dim")
        table_ds.add_column("Edges", style="dim")
        table_ds.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        if has_time:
            table_ds.add_column("Time Mean (s)")
            table_ds.add_column("Time StdDev (s)")
            table_ds.add_column("Time CV (%)", style="green")
        if has_ratio:
            table_ds.add_column("Ratio Mean")
            table_ds.add_column("Ratio StdDev")
            table_ds.add_column("Ratio CV (%)", style="green")

        for ds in self.get_dataset_order(data):
            ds_data = stats_df[stats_df["dataset"] == ds]
            if ds_data.empty: continue

            ds_meta = metadata.get(ds, {})
            nodes = str(ds_meta.get('nodes', '-'))
            edges = str(ds_meta.get('edges', '-'))

            for _, row in ds_data.iterrows():
                algo = str(row['algorithm'])
                if algo not in algos: continue

                row_data = [ds, nodes, edges, algo]

                if has_time:
                    t_mean = f"{row['time_mean']:.3f}".rstrip('0').rstrip('.') if pd.notnull(row.get('time_mean')) else "-"
                    t_std = f"{row['time_std']:.3f}".rstrip('0').rstrip('.') if pd.notnull(row.get('time_std')) else "-"
                    t_cv = f"{row['time_cv']:.2f}%" if pd.notnull(row.get('time_cv')) else "-"
                    row_data.extend([t_mean, t_std, t_cv])

                if has_ratio:
                    r_mean = f"{row['ratio_mean']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('ratio_mean')) else "-"
                    r_std = f"{row['ratio_std']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('ratio_std')) else "-"
                    r_cv = f"{row['ratio_cv']:.4f}%" if pd.notnull(row.get('ratio_cv')) else "-"
                    row_data.extend([r_mean, r_std, r_cv])

                table_ds.add_row(*row_data)

        table_gm = Table(title="Geometric Means and Stability Summary", show_header=True)
        table_gm.add_column("Algorithm", style="magenta", no_wrap=True, min_width=18)
        if has_time:
            table_gm.add_column("Time GeoMean (s)")
            table_gm.add_column("Median Time CV (%)")
            table_gm.add_column("IQR Time CV (%)")
            table_gm.add_column("Max Time CV (%)")
        if has_ratio:
            table_gm.add_column("Ratio (GeoMean)")
            table_gm.add_column("Median Ratio CV (%)")
            table_gm.add_column("IQR Ratio CV (%)")
            table_gm.add_column("Max Ratio CV (%)")

        for algo in algos:
            algo_data = stats_df[stats_df['algorithm'] == algo]
            row_data = [algo]

            if has_time:
                t_vals = algo_data['time_mean'].dropna()
                t_gm = gmean(t_vals) if not t_vals.empty else np.nan
                row_data.append(f"{t_gm:.2f}" if pd.notnull(t_gm) else "-")

                time_cv_vals = algo_data['time_cv'].dropna()
                if time_cv_vals.empty:
                    row_data.extend(["-", "-", "-"])
                else:
                    median_cv = time_cv_vals.median()
                    iqr_cv = time_cv_vals.quantile(0.75) - time_cv_vals.quantile(0.25)
                    max_cv = time_cv_vals.max()
                    row_data.extend([
                        self.format_percent(median_cv),
                        self.format_percent(iqr_cv),
                        self.format_percent(max_cv),
                    ])

            if has_ratio:
                r_vals = algo_data['ratio_mean'].dropna()
                r_gm = gmean(r_vals) if not r_vals.empty else np.nan
                row_data.append(f"{r_gm:.4f}" if pd.notnull(r_gm) else "-")

                ratio_cv_vals = algo_data['ratio_cv'].dropna()
                if ratio_cv_vals.empty:
                    row_data.extend(["-", "-", "-"])
                else:
                    median_cv = ratio_cv_vals.median()
                    iqr_cv = ratio_cv_vals.quantile(0.75) - ratio_cv_vals.quantile(0.25)
                    max_cv = ratio_cv_vals.max()
                    row_data.extend([
                        self.format_percent(median_cv),
                        self.format_percent(iqr_cv),
                        self.format_percent(max_cv),
                    ])

            table_gm.add_row(*row_data)

        txt_path = out_dir / "time_compression_stats.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            file_console = Console(file=f, width=220)
            file_console.print(table_ds)
            file_console.print()
            file_console.print(table_gm)
        artifacts.append(txt_path)

        palette = [self.get_algo_style(algo)["color"] for algo in algos]
        ordered_datasets = self.get_dataset_order(data)

        if has_time:
            fig_time, ax_time = self.create_figure(profile)
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
                ax=ax_time
            )
            self.style_paper_bar_chart(ax_time, ylabel="time (seconds)", algos=algos)
            fig_time.tight_layout(pad=0.4)

            pdf_path_time = out_dir / "runtime_bar_chart.pdf"
            fig_time.savefig(pdf_path_time, format="pdf", bbox_inches='tight')
            plt.close(fig_time)
            artifacts.extend([pdf_path_time])

        if has_ratio:
            fig_comp, ax_comp = self.create_figure(profile)
            sns.barplot(
                data=data, x="dataset", y="ratio", hue="algorithm",
                hue_order=algos, order=ordered_datasets,
                palette=palette, errorbar=None,
                edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH,
                ax=ax_comp
            )

            self.style_paper_bar_chart(ax_comp, ylabel="relative size", algos=algos)
            fig_comp.tight_layout(pad=0.4)

            pdf_path_comp = out_dir / "compression_bar_chart.pdf"
            fig_comp.savefig(pdf_path_comp, format="pdf", bbox_inches='tight')
            plt.close(fig_comp)
            artifacts.extend([pdf_path_comp])

        return artifacts
