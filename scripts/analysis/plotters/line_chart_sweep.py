import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
from rich.console import Console
from rich.table import Table
from matplotlib.ticker import MaxNLocator

from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register


def format_param_tick(value) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


@register
class LineChartSweep(Plotter):
    plotter_id = "line_chart_sweep"
    description = "Parameter sweep visualization (1 plot per algorithm, lines = datasets)"
    FIGURE_PROFILE = FigureProfile(
        figsize=(10.8, 6.075),
        font_size=38.0,
        label_size=35.0,
        tick_size=20.0,
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

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        options = options or {}

        profile = self.use_figure_profile(self.FIGURE_PROFILE, options, profile_name="sweep_line")
        generated_files = []

        if "param_name" not in data.columns:
            return generated_files

        param_name = data["param_name"].iloc[0]
        datasets = data["dataset"].unique()
        algorithms = [a for a in algos if a in data["algorithm"].values]

        time_label = options.get("time_label", "time (seconds)").lower()

        metrics = [
            ("time", time_label),
            ("ratio", "relative size")
        ]

        colors = self.THEME_COLORS
        markers = self.THEME_MARKERS

        sweep_stats = []

        for algo in algorithms:
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            algo_data["param_value"] = pd.to_numeric(algo_data["param"], errors="coerce")
            if algo_data["param_value"].isna().any():
                ordered_params = list(dict.fromkeys(algo_data["param"].tolist()))
                order_map = {param: idx for idx, param in enumerate(ordered_params)}
                algo_data["param_value"] = algo_data["param"].map(order_map)

            summary_data = algo_data.groupby(['dataset', 'param_value'], as_index=False)[['time', 'ratio']].mean()

            sweep_vals = sorted(summary_data["param_value"].unique())
            x_positions = {value: idx for idx, value in enumerate(sweep_vals)}
            summary_data["param_position"] = summary_data["param_value"].map(x_positions)

            for metric, ylabel in metrics:
                fig, ax = self.create_figure(profile)
                plotted_ds = 0

                for ds in datasets:
                    ds_data = summary_data[summary_data["dataset"] == ds].sort_values(by="param_value")
                    if ds_data.empty: continue

                    for _, row in ds_data.iterrows():
                        sweep_stats.append({
                            "Algorithm": algo,
                            "Dataset": ds,
                            "Metric": metric,
                            "Parameter": row["param_value"],
                            "Value": row[metric]
                        })

                    if ds in self.dataset_order:
                        style_idx = self.dataset_order.index(ds)
                    else:
                        style_idx = sum(ord(c) for c in ds)

                    color = colors[style_idx % len(colors)]
                    mk = markers[style_idx % len(markers)]

                    ax.plot(
                        ds_data["param_position"], ds_data[metric],
                        label=ds, marker=mk, color=color,
                        linestyle="-", alpha=0.9, zorder=3
                    )

                    plotted_ds += 1

                ax.set_xlabel(f"parameter: {param_name}", style='italic')
                ax.set_ylabel(ylabel, style='italic')
                if metric == "ratio":
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))

                max_x_tick_labels = options.get("sweep_max_x_tick_labels", 10)
                if len(sweep_vals) > max_x_tick_labels:
                    tick_step = math.ceil(len(sweep_vals) / max_x_tick_labels)
                    tick_indices = list(range(0, len(sweep_vals), tick_step))
                    if tick_indices[-1] != len(sweep_vals) - 1:
                        tick_indices.append(len(sweep_vals) - 1)
                else:
                    tick_indices = list(range(len(sweep_vals)))

                ax.set_xticks(tick_indices)
                ax.set_xticklabels([format_param_tick(sweep_vals[i]) for i in tick_indices], rotation=0)

                self.style_major_minor_grid(ax, minor_axis="both")

                self.despine(ax, top=True, right=True)

                if plotted_ds > 0:
                    self.add_centered_legend(
                        ax,
                        loc='upper center',
                        y=-0.24,
                        ncol=min(5, plotted_ds),
                        handlelength=1.5,
                        handletextpad=0.4,
                        columnspacing=1.0,
                    )

                fig.tight_layout()

                pdf_path = out_dir / f"sweep_{algo}_{param_name}_{metric}.pdf"
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close(fig)

                generated_files.extend([pdf_path])

        # ---------------------------------------------------------
        # Print and Save the Summary Tables
        # ---------------------------------------------------------
        if sweep_stats:
            stats_df = pd.DataFrame(sweep_stats)

            pivot_df = stats_df.pivot_table(
                index=["Algorithm", "Dataset", "Metric"],
                columns="Parameter",
                values="Value",
                aggfunc="mean"
            ).reset_index()

            detail_table = Table(
                title=f"Sweep Details: {param_name} Values",
                show_lines=True
            )
            detail_table.add_column("Algorithm", style="cyan", justify="left", no_wrap=True)
            detail_table.add_column("Dataset", style="magenta", justify="left", no_wrap=True)
            detail_table.add_column("Metric", style="green", justify="left", no_wrap=True)

            param_cols = [col for col in pivot_df.columns if col not in ["Algorithm", "Dataset", "Metric"]]
            for p in param_cols:
                detail_table.add_column(f"{param_name}={p:.4g}", justify="right", no_wrap=True)

            for _, row in pivot_df.iterrows():
                row_data = [
                    str(row["Algorithm"]),
                    str(row["Dataset"]),
                    str(row["Metric"])
                ]
                for p in param_cols:
                    val = row[p]
                    if pd.notna(val):
                        row_data.append(f"{val:.4g}")
                    else:
                        row_data.append("-")
                detail_table.add_row(*row_data)

            detail_txt_path = out_dir / f"sweep_{param_name}_detailed_stats.txt"
            with open(detail_txt_path, "w", encoding="utf-8") as f:
                calculated_width = 40 + len(param_cols) * 15
                file_console = Console(file=f, width=max(1000, calculated_width))
                file_console.print(detail_table)

            generated_files.append(detail_txt_path)

        return generated_files
