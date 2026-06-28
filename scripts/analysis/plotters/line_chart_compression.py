from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register


@register
class LineChartCompressionPlotter(Plotter):
    plotter_id = "line_chart_compression"
    description = "Line chart for streaming compression over time"
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
        profile = self.use_figure_profile(self.FIGURE_PROFILE, options, profile_name="compression_line")

        data["change_ratio"] = pd.to_numeric(data["change_ratio"], errors='coerce')

        generated_files = []
        datasets = data["dataset"].unique()

        paper_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for ds in datasets:
            ds_data = data[data["dataset"] == ds]

            fig, ax = self.create_figure(profile)

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
                        markeredgecolor=color,
                        markerfacecolor="none",
                    )

            ax.set_ylabel("Relative Size", style='italic')
            ax.set_xlabel("Ratio of Processed Changes", style='italic')

            def format_checkpoint(x, _):
                return f"{x:.1f}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_checkpoint))
            ax.set_xticks(paper_ticks)
            ax.set_xlim(-0.05, 1.05)

            self.despine(ax, top=True, right=True)

            self.add_centered_legend(
                ax,
                y=1.25,
                loc='upper center',
                ncol=min(len(algos), 4),
            )

            fig.tight_layout()

            pdf_path = out_dir / f"compression_line_{ds}.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)

            generated_files.extend([pdf_path])

        return generated_files
