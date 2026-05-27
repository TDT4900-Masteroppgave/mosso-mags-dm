from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean

from scripts.analysis.plotters.base_plotter import Plotter, register
from rich.console import Console
from rich.table import Table

@register
class CombinedBarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_compression"
    description = "Bar charts comparing time and compression ratio with stats"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()
        artifacts = []

        # --- Metadata Setup ---
        metadata = self.get_dataset_metadata()

        has_time = "time" in data.columns
        has_ratio = "ratio" in data.columns

        # ==========================================
        # 1. GENERATE UNIFIED STATISTICS TABLES
        # ==========================================
        agg_dict = {}
        if has_time:
            agg_dict['time'] = ['mean', 'std']
        if has_ratio:
            agg_dict['ratio'] = ['mean', 'std']

        # Pre-compute metrics grouping by dataset and algorithm
        stats_df = data.groupby(["dataset", "algorithm"]).agg(agg_dict)
        stats_df.columns = [f"{col[0]}_{col[1]}" for col in stats_df.columns]
        stats_df = stats_df.reset_index()

        console = Console(record=True)

        # --- TABLE 1: Dataset Statistics ---
        table_ds = Table(title="Compression & Timing Statistics", show_header=True, show_lines=True)
        table_ds.add_column("Dataset", style="cyan")
        table_ds.add_column("Nodes", style="dim")
        table_ds.add_column("Edges", style="dim")
        table_ds.add_column("Algorithm", style="magenta")
        if has_time:
            table_ds.add_column("Time Mean")
            table_ds.add_column("Time StdDev")
        if has_ratio:
            table_ds.add_column("Ratio Mean")
            table_ds.add_column("Ratio StdDev")

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
                    t_mean = f"{row['time_mean']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('time_mean')) else "-"
                    t_std = f"{row['time_std']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('time_std')) else "-"
                    row_data.extend([t_mean, t_std])

                if has_ratio:
                    r_mean = f"{row['ratio_mean']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('ratio_mean')) else "-"
                    r_std = f"{row['ratio_std']:.6f}".rstrip('0').rstrip('.') if pd.notnull(row.get('ratio_std')) else "-"
                    row_data.extend([r_mean, r_std])

                table_ds.add_row(*row_data)

        console.print(table_ds)
        console.print("\n")

        # --- TABLE 2: Geometric Means Summary ---
        table_gm = Table(title="Geometric Means Summary", show_header=True)
        table_gm.add_column("Algorithm", style="magenta")
        if has_time:
            table_gm.add_column("Time (GeoMean)")
        if has_ratio:
            table_gm.add_column("Ratio (GeoMean)")

        for algo in algos:
            algo_data = stats_df[stats_df['algorithm'] == algo]
            row_data = [algo]

            if has_time:
                t_vals = algo_data['time_mean'].dropna()
                t_gm = gmean(t_vals) if not t_vals.empty else np.nan
                row_data.append(f"{t_gm:.6f}".rstrip('0').rstrip('.') if pd.notnull(t_gm) else "-")

            if has_ratio:
                r_vals = algo_data['ratio_mean'].dropna()
                r_gm = gmean(r_vals) if not r_vals.empty else np.nan
                row_data.append(f"{r_gm:.6f}".rstrip('0').rstrip('.') if pd.notnull(r_gm) else "-")

            table_gm.add_row(*row_data)

        console.print(table_gm)

        # Save both tables to a single text file
        txt_path = out_dir / "combined_stats.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console.export_text())
        artifacts.append(txt_path)

        # ==========================================
        # 2. GENERATE BAR CHARTS (Time & Ratio)
        # ==========================================
        palette = [self.get_algo_style(algo)["color"] for algo in algos]

        if has_time:
            fig_time, ax_time = plt.subplots(figsize=(6, 3.5))
            sns.barplot(
                data=data, x="dataset", y="time", hue="algorithm",
                hue_order=algos, order=self.get_dataset_order(data),
                palette=palette, errorbar=None, edgecolor="black",
                linewidth=1.2, ax=ax_time
            )
            total_bars_time = len(data["dataset"].unique()) * len(algos)
            if total_bars_time <= 20:
                for container in ax_time.containers:
                    labels = [f"{val:.4f}".rstrip('0').rstrip('.') if '.' in f"{val:.4f}" else str(val) for val in container.datavalues]
                    ax_time.bar_label(container, labels=labels, padding=3, fontsize=8, color='#4A5568')

            plt.xlabel("")
            plt.ylabel("time (microseconds)", fontsize=14, style='italic')
            plt.legend(title="", bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=len(algos), frameon=False)
            plt.tight_layout()

            png_path_time = out_dir / "runtime_bar_chart.png"
            pdf_path_time = out_dir / "runtime_bar_chart.pdf"
            plt.savefig(png_path_time, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path_time, format="pdf", bbox_inches='tight')
            plt.close(fig_time)
            artifacts.extend([png_path_time, pdf_path_time])

        if has_ratio:
            fig_comp, ax_comp = plt.subplots(figsize=(6, 3.5))
            sns.barplot(
                data=data, x="dataset", y="ratio", hue="algorithm",
                hue_order=algos, order=self.get_dataset_order(data),
                palette=palette, errorbar=None, edgecolor="white",
                linewidth=0.8, ax=ax_comp
            )

            ax_comp.set_xlabel("")
            ax_comp.set_ylabel("relative size", fontsize=12, style='italic')

            total_bars_comp = len(data["dataset"].unique()) * len(algos)
            if total_bars_comp <= 20:
                for container in ax_comp.containers:
                    labels = [f"{val:.4f}".rstrip('0').rstrip('.') if '.' in f"{val:.4f}" else str(val) for val in container.datavalues]
                    ax_comp.bar_label(container, labels=labels, padding=3, fontsize=8, color='#4A5568')

            sns.despine(ax=ax_comp, top=True, right=True)
            ax_comp.legend(title="", bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=len(algos), frameon=False, fontsize=9)
            plt.tight_layout()

            png_path_comp = out_dir / "compression_bar_chart.png"
            pdf_path_comp = out_dir / "compression_bar_chart.pdf"
            plt.savefig(png_path_comp, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path_comp, format="pdf", bbox_inches='tight')
            plt.close(fig_comp)
            artifacts.extend([png_path_comp, pdf_path_comp])

        return artifacts