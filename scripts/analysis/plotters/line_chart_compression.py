from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class LineChartCompressionPlotter(Plotter):
    plotter_id = "line_chart_compression"
    description = "Line chart for streaming compression over time"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        self.set_chart_theme()

        data["change_ratio"] = pd.to_numeric(data["change_ratio"], errors='coerce')

        generated_files = []
        datasets = data["dataset"].unique()

        paper_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for ds in datasets:
            ds_data = data[data["dataset"] == ds]

            fig, ax = plt.subplots(figsize=(6, 3.5))

            markers = ['s', 'o', '^', 'D', 'X', 'v']

            # Use specific colors if you want to perfectly mimic the paper,
            # or stick to your grayscale/black theme. We'll use black/gray here.
            colors = sns.color_palette("dark")

            # Iterate and plot individually to handle the Batch vs Streaming logic
            for i, algo in enumerate(algos):
                algo_data = ds_data[ds_data["algorithm"] == algo].sort_values("change_ratio")
                if algo_data.empty:
                    continue


                is_streaming = algo_data["is_streaming"].iloc[0]

                if is_streaming:
                    smoothed_ratio = algo_data["ratio"].rolling(window=3, min_periods=1).mean()

                    ax.plot(
                        algo_data["change_ratio"],
                        smoothed_ratio,
                        label=algo,
                        color="red" if "mosso" in algo.lower() else colors[i % len(colors)],
                        linestyle="-",
                        linewidth=1.5,
                        marker=""
                    )
                else:
                    ax.plot(
                        algo_data["change_ratio"],
                        algo_data["ratio"],
                        label=algo,
                        color=colors[i % len(colors)],
                        linestyle="", # Explicitly no connecting line
                        marker=markers[i % len(markers)],
                        markersize=8,
                        markeredgecolor=colors[i % len(colors)],
                        markerfacecolor="none",
                        mew=1.2 # Marker edge width
                    )

            plt.ylabel("Compression Ratio", fontsize=14)
            plt.xlabel("Ratio of Processed Changes", fontsize=14)

            def format_checkpoint(x, _):
                return f"{x:.1f}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_checkpoint))
            ax.set_xticks(paper_ticks)
            ax.set_xlim(-0.05, 1.05) # Add a tiny padding so 0.0 and 1.0 markers aren't cut off

            plt.legend(
                title="",
                bbox_to_anchor=(0.5, 1.15),
                loc='upper center',
                ncol=len(algos),
                frameon=False
            )

            plt.tight_layout()

            # Save as high-res PNG
            png_path = out_dir / f"compression_line_{ds}_{ts}.png"
            plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
            plt.close()

            generated_files.append(png_path)

        return generated_files