from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class LineChartCompressionPlotter(Plotter):
    plotter_id = "line_chart_compression"
    description = "Line chart for streaming compression over time"

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()

        data["change_ratio"] = pd.to_numeric(data["change_ratio"], errors='coerce')

        generated_files = []
        datasets = data["dataset"].unique()

        paper_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for ds in datasets:
            ds_data = data[data["dataset"] == ds]

            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6, 3.5))

            for algo in algos:
                algo_data = ds_data[ds_data["algorithm"] == algo].sort_values("change_ratio")
                if algo_data.empty:
                    continue

                style = self.get_algo_style(algo)
                color = style["color"]
                marker = style["marker"]

                raw_streaming_val = str(algo_data["is_streaming"].iloc[0]).strip().lower()
                is_streaming = raw_streaming_val in ["1", "1.0", "true"]

                if is_streaming:
                    smoothed_ratio = algo_data["ratio"].rolling(window=3, min_periods=1).mean()

                    ax.plot(
                        algo_data["change_ratio"],
                        smoothed_ratio,
                        label=algo,
                        color=color,
                        linestyle="-",
                        linewidth=1.5,
                        marker=""
                    )

                else:
                    ax.plot(
                        algo_data["change_ratio"],
                        algo_data["ratio"],
                        label=algo,
                        color=color,
                        linestyle="",
                        marker=marker,
                        markersize=6,
                        markeredgecolor=color,
                        markerfacecolor="none",
                        mew=1.2
                    )

            ax.set_ylabel("Compression Ratio", fontsize=12, style='italic')
            ax.set_xlabel("Ratio of Processed Changes", fontsize=12, style='italic')

            def format_checkpoint(x, _):
                return f"{x:.1f}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_checkpoint))
            ax.set_xticks(paper_ticks)
            ax.set_xlim(-0.05, 1.05)

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

            png_path = out_dir / f"compression_line_{ds}.png"
            pdf_path = out_dir / f"compression_line_{ds}.pdf"
            plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close()

            generated_files.extend([png_path, pdf_path])

        return generated_files