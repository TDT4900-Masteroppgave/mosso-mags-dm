import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

from scripts.analysis.plotters.base_plotter import Plotter, register


@register
class LineChartSweep(Plotter):
    plotter_id = "line_chart_sweep"
    description = "Parameter sweep visualization (1 plot per algorithm, lines = datasets)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> \
            list[Path]:
        self.set_chart_theme()
        generated_files = []

        if "param_name" not in data.columns:
            Console().print("[bold red][Warning] 'param_name' column missing.[/bold red]")
            return generated_files

        param_name = data["param_name"].iloc[0]
        datasets = data["dataset"].unique()
        algorithms = [a for a in algos if a in data["algorithm"].values]

        time_label = options.get("time_label", "time (seconds)").lower()

        metrics = [
            ("time", time_label),
            ("ratio", "relative size")
        ]

        # Use the established theme colors and markers from base_plotter
        colors = self.THEME_COLORS
        markers = self.THEME_MARKERS

        for algo in algorithms:
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            # Group by the actual numeric 'param' column
            summary_data = algo_data.groupby(['dataset', 'param'], as_index=False)[['time', 'ratio']].mean()

            # Extract unique parameter steps to force clean X-axis ticks
            sweep_vals = sorted(summary_data["param"].unique())

            for metric, ylabel in metrics:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                plotted_ds = 0

                for ds in datasets:
                    ds_data = summary_data[summary_data["dataset"] == ds].sort_values(by="param")
                    if ds_data.empty: continue

                    # Derive a consistent style index based on dataset name
                    if ds in self.dataset_order:
                        style_idx = self.dataset_order.index(ds)
                    else:
                        style_idx = sum(ord(c) for c in ds)

                    color = colors[style_idx % len(colors)]
                    mk = markers[style_idx % len(markers)]

                    # Plot the line for this dataset
                    ax.plot(
                        ds_data["param"], ds_data[metric],
                        label=ds, marker=mk, markersize=6, color=color,
                        linestyle="-", linewidth=1.5, alpha=0.9, zorder=3
                    )
                    plotted_ds += 1

                ax.set_xlabel(f"parameter: {param_name}", fontsize=14, style='italic')
                ax.set_ylabel(ylabel, fontsize=14, style='italic')

                # Force X-ticks to exactly match the sweep values being tested
                ax.set_xticks(sweep_vals)

                # Consistent Grid Styling
                ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
                ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.4)

                # Consistent Legend Styling (Bottom row, compact)
                if plotted_ds > 0:
                    ax.legend(
                        title="",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.18),
                        ncol=min(5, plotted_ds),  # Pack datasets into a single row
                        handlelength=1.5,  # Slightly longer line for marker visibility
                        handletextpad=0.4,  # Tight text gap
                        columnspacing=1.0,  # Tight column gap
                        frameon=False,
                        fontsize=9
                    )

                fig.tight_layout()

                # Save specific file depending on the metric
                out_path = out_dir / f"sweep_{algo}_{param_name}_{metric}.png"
                fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
                plt.close(fig)

                generated_files.append(out_path)

        return generated_files