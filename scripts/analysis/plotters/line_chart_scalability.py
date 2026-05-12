import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class LineChartScalabilityPlotter(Plotter):
    plotter_id = "line_chart_scalability"
    description = "Line chart for scalability"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        self.set_chart_theme()

        data["edges_evaluated"] = pd.to_numeric(data["edges_evaluated"], errors='coerce')
        data["accumulated_time_sec"] = pd.to_numeric(data["accumulated_time_sec"], errors='coerce')

        generated_files = []
        datasets = data["dataset"].unique()

        for ds in datasets:
            ds_data = data[data["dataset"] == ds].dropna(subset=["edges_evaluated", "accumulated_time_sec"])
            if ds_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 3.5))

            num_algos = len(ds_data["algorithm"].unique())
            markers = ['s', 'o', '^', 'D', 'X', 'v']

            sns.lineplot(
                data=ds_data,
                x="edges_evaluated",
                y="accumulated_time_sec",
                hue="algorithm",
                style="algorithm",
                markers=markers[:num_algos],
                dashes=False, # Keep all lines solid
                palette=["black"] * num_algos, # Force all lines to black
                linewidth=1.2,
                markersize=8,
                markeredgecolor="black",
                markerfacecolor="none", # Empty markers
                errorbar=None,
                ax=ax
            )

            min_edges = ds_data["edges_evaluated"].min()
            min_time = ds_data["accumulated_time_sec"].min()

            linear_x = sorted(ds_data["edges_evaluated"].unique())
            linear_y = [float(min_time) * (float(x) / float(min_edges)) for x in linear_x]

            ax.plot(
                linear_x,
                linear_y,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Linear ($O(|E|)$)",
                zorder=1
            )

            plt.ylabel("time (seconds)", fontsize=14, style='italic')
            plt.xlabel("Number of Changes", fontsize=14)

            ax.set_yscale("log", base=2)
            ax.set_xscale("log", base=2)

            def format_power_of_2(val, _):
                if val <= 0: return ""
                p = int(round(math.log2(val)))
                return f"$2^{{{p}}}$"

            formatter = plt.FuncFormatter(format_power_of_2)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            dynamic_ticks = sorted(ds_data["edges_evaluated"].unique())
            ax.set_xticks(dynamic_ticks)
            plt.xticks(rotation=0)

            plt.legend(
                title="",
                bbox_to_anchor=(0.5, 1.15),
                loc='upper center',
                ncol=num_algos + 1,
                frameon=False
            )
            plt.tight_layout()

            # Save as high-res PNG
            png_path = out_dir / f"scalability_line_{ds}_{ts}.png"
            plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
            plt.close()

            generated_files.append(png_path)

        return generated_files