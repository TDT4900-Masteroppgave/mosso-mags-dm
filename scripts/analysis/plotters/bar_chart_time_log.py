from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class BarChartPlotter(Plotter):
    plotter_id = "bar_chart_time_log"
    description = "Bar chart comparing time (logarithmic scale)"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()

        suffix = "_".join(algos)

        # 1. Generate and display the statistics table
        stats_df = data.groupby(["dataset", "algorithm"])["time"].agg(
            Mean='mean',
            Median='median',
            Min='min',
            Max='max',
            StdDev='std'
        ).reset_index()

        for col in ['Mean', 'Median', 'Min', 'Max', 'StdDev']:
            stats_df[col] = stats_df[col].apply(
                lambda x: f"{x:.6f}".rstrip('0').rstrip('.') if pd.notnull(x) and '.' in f"{x:.6f}" else x
            )

        from rich.console import Console
        from rich.table import Table
        
        console = Console(record=True)
        table = Table(title="Runtime Statistics (Log Scale Plotter)", show_header=True)
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="magenta")
        table.add_column("Mean")
        table.add_column("Median")
        table.add_column("Min")
        table.add_column("Max")
        table.add_column("StdDev")

        for _, row in stats_df.iterrows():
            table.add_row(
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row["Mean"]),
                str(row["Median"]),
                str(row["Min"]),
                str(row["Max"]),
                str(row["StdDev"])
            )
        
        console.print(table)
        
        txt_path = out_dir / f"runtime_bar_chart_log_stats_{suffix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(console.export_text())

        # 2. Generate the plot
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
            edgecolor="white",
            linewidth=0.8,
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_yscale("log", base=10)
        ax.set_ylabel("time (microseconds)", fontsize=12, style='italic')

        total_bars = len(data["dataset"].unique()) * len(algos)
        if total_bars <= 20:
            for container in ax.containers:
                labels = [f"{val:.4f}".rstrip('0').rstrip('.') if '.' in f"{val:.4f}" else str(val) for val in container.datavalues]
                ax.bar_label(container, labels=labels, padding=3, fontsize=8, color='#4A5568')

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

        png_path = out_dir / f"runtime_bar_chart_log_{suffix}.png"
        pdf_path = out_dir / f"runtime_bar_chart_log_{suffix}.pdf"
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        return [png_path, pdf_path, txt_path]