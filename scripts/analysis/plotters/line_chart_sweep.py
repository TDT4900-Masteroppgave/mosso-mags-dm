import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table

from scripts.analysis.plotters.base_plotter import Plotter, register


def format_param_tick(value) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


@register
class LineChartSweep(Plotter):
    plotter_id = "line_chart_sweep"
    description = "Parameter sweep visualization (1 plot per algorithm, lines = datasets)"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        options = options or {}

        self.set_chart_theme()
        generated_files = []
        console = Console()

        if "param_name" not in data.columns:
            console.print("[bold red][Warning] 'param_name' column missing.[/bold red]")
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

        # Container for our start-to-end statistics
        sweep_stats = []

        for algo in algorithms:
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            algo_data["param_value"] = pd.to_numeric(algo_data["param"], errors="coerce")
            if algo_data["param_value"].isna().any():
                console.print(
                    f"[bold yellow][Warning][/bold yellow] Non-numeric parameter values found for {algo}; "
                    "falling back to original sweep order."
                )
                ordered_params = list(dict.fromkeys(algo_data["param"].tolist()))
                order_map = {param: idx for idx, param in enumerate(ordered_params)}
                algo_data["param_value"] = algo_data["param"].map(order_map)

            # Group by the numeric parameter value
            summary_data = algo_data.groupby(['dataset', 'param_value'], as_index=False)[['time', 'ratio']].mean()

            # Extract unique parameter steps to force clean X-axis ticks
            sweep_vals = sorted(summary_data["param_value"].unique())
            x_positions = {value: idx for idx, value in enumerate(sweep_vals)}
            summary_data["param_position"] = summary_data["param_value"].map(x_positions)

            for metric, ylabel in metrics:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                plotted_ds = 0

                for ds in datasets:
                    ds_data = summary_data[summary_data["dataset"] == ds].sort_values(by="param_value")
                    if ds_data.empty: continue

                    # Collect all parameter values for the tables
                    for _, row in ds_data.iterrows():
                        sweep_stats.append({
                            "Algorithm": algo,
                            "Dataset": ds,
                            "Metric": metric,
                            "Parameter": row["param_value"],
                            "Value": row[metric]
                        })

                    # Derive a consistent style index based on dataset name
                    if ds in self.dataset_order:
                        style_idx = self.dataset_order.index(ds)
                    else:
                        style_idx = sum(ord(c) for c in ds)

                    color = colors[style_idx % len(colors)]
                    mk = markers[style_idx % len(markers)]

                    # Plot the line for this dataset
                    ax.plot(
                        ds_data["param_position"], ds_data[metric],
                        label=ds, marker=mk, markersize=5.5, color=color,
                        linestyle="-", linewidth=1.5, alpha=0.9, zorder=3
                    )

                    plotted_ds += 1

                ax.set_xlabel(f"parameter: {param_name}", fontsize=12, style='italic')
                ax.set_ylabel(ylabel, fontsize=12, style='italic')

                # Force X-ticks to exactly match the sweep values being tested
                ax.set_xticks(range(len(sweep_vals)))

                # Rotate labels if there are many values to prevent overlap
                rotation = 45 if len(sweep_vals) > 5 else 0
                ax.set_xticklabels([format_param_tick(v) for v in sweep_vals], rotation=rotation)

                # Consistent Grid Styling
                ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
                ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.4, alpha=0.3)

                sns.despine(ax=ax, top=True, right=True)

                # Consistent Legend Styling (Bottom row, compact)
                if plotted_ds > 0:
                    ax.legend(
                        title="",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.28 if rotation else -0.24),
                        ncol=min(5, plotted_ds),
                        handlelength=1.5,
                        handletextpad=0.4,
                        columnspacing=1.0,
                        frameon=False,
                        fontsize=9
                    )

                fig.tight_layout()

                out_path = out_dir / f"sweep_{algo}_{param_name}_{metric}.png"
                pdf_path = out_dir / f"sweep_{algo}_{param_name}_{metric}.pdf"
                fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close(fig)

                generated_files.extend([out_path, pdf_path])

        # ---------------------------------------------------------
        # Print and Save the Summary Tables
        # ---------------------------------------------------------
        if sweep_stats:
            stats_df = pd.DataFrame(sweep_stats)

            # Pivot table to make it easier to read (Parameters as columns)
            pivot_df = stats_df.pivot_table(
                index=["Algorithm", "Dataset", "Metric"],
                columns="Parameter",
                values="Value",
                aggfunc="mean"
            ).reset_index()

            # Detailed Dataset Breakdown
            detail_table = Table(
                title=f"Sweep Details: {param_name} Values",
                show_lines=True
            )
            # Added no_wrap=True to prevent any unwanted wrapping
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

            console.print(detail_table)

            # Save the rich table to a text file
            detail_txt_path = out_dir / f"sweep_{param_name}_detailed_stats.txt"
            with open(detail_txt_path, "w", encoding="utf-8") as f:
                # FIX: Set a large explicit width to prevent rich from truncating wide tables
                calculated_width = 40 + len(param_cols) * 15
                file_console = Console(file=f, width=max(1000, calculated_width))
                file_console.print(detail_table)

            generated_files.append(detail_txt_path)

            console.print(f"[green]✓[/green] Saved table text: [bold]{detail_txt_path.name}[/bold]")

        return generated_files