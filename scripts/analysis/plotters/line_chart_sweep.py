import pandas as pd
import matplotlib.pyplot as plt
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

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> \
            list[Path]:

        options = {
            "start_param": 1,
            "end_param": 100,
            "time_label": "time (seconds)"
        }

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

        # --- NEW: Extract custom start and end points from options ---
        custom_start = options.get("start_param", None)
        custom_end = options.get("end_param", None)

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

            max_val = algo_data["param_value"].max()
            if pd.notna(max_val) and max_val >= 5:
                desired_vals = [1.0] + list(range(5, int(max_val) + 5, 5))
                algo_data = algo_data[algo_data["param_value"].isin(desired_vals)]
            # ==========================================

            # Group by the numeric parameter value so values like 10 sort after 5, not after 1.
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

                    # ---------------------------------------------------------
                    # Calculate Start and End values for the tables
                    # ---------------------------------------------------------
                    if len(ds_data) > 1:
                        start_val, end_val = None, None

                        # --- NEW LOGIC: Use specific points if provided ---
                        if custom_start is not None and custom_end is not None:
                            start_row = ds_data[ds_data["param_value"] == custom_start]
                            end_row = ds_data[ds_data["param_value"] == custom_end]

                            # Only calculate if BOTH target parameters exist in this dataset
                            if not start_row.empty and not end_row.empty:
                                start_val = start_row.iloc[0][metric]
                                end_val = end_row.iloc[0][metric]
                        else:
                            # Default behavior: Just use the very first and last points in the data
                            start_val = ds_data.iloc[0][metric]
                            end_val = ds_data.iloc[-1][metric]

                        # Calculate diff if we found valid start and end points
                        if start_val is not None and end_val is not None and pd.notna(start_val) and start_val > 0:
                            pct_diff = ((end_val - start_val) / start_val) * 100
                            sweep_stats.append({
                                "Algorithm": algo,
                                "Dataset": ds,
                                "Metric": metric,
                                "Start": start_val,
                                "End": end_val,
                                "Diff": pct_diff
                            })
                    # ---------------------------------------------------------

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
                        label=ds, marker=mk, markersize=6, color=color,
                        linestyle="-", linewidth=1.5, alpha=0.9, zorder=3
                    )
                    plotted_ds += 1

                ax.set_xlabel(f"parameter: {param_name}", fontsize=14, style='italic')
                ax.set_ylabel(ylabel, fontsize=14, style='italic')

                # Force X-ticks to exactly match the sweep values being tested
                ax.set_xticks(range(len(sweep_vals)))
                ax.set_xticklabels([format_param_tick(v) for v in sweep_vals])

                # Consistent Grid Styling
                ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
                ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.4)

                # Consistent Legend Styling (Bottom row, compact)
                if plotted_ds > 0:
                    ax.legend(
                        title="",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.18),
                        ncol=min(5, plotted_ds),
                        handlelength=1.5,
                        handletextpad=0.4,
                        columnspacing=1.0,
                        frameon=False,
                        fontsize=9
                    )

                fig.tight_layout()

                out_path = out_dir / f"sweep_{algo}_{param_name}_{metric}.png"
                fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
                plt.close(fig)

                generated_files.append(out_path)

        # ---------------------------------------------------------
        # Print and Save the Summary Tables
        # ---------------------------------------------------------
        if sweep_stats:
            stats_df = pd.DataFrame(sweep_stats)
            # --- NEW: Dynamic Title string based on options ---
            title_range = f"({custom_start} vs {custom_end})" if custom_start and custom_end else "(First vs Last)"

            # 1. High-Level Average Table
            avg_table = Table(
                title=f"Average % Difference Across Datasets {title_range}",
                show_header=True,
                header_style="bold yellow"
            )
            avg_table.add_column("Algorithm", style="cyan")
            avg_table.add_column("Metric", style="green")
            avg_table.add_column("Avg % Difference", justify="right")

            avg_stats = stats_df.groupby(["Algorithm", "Metric"])["Diff"].mean().reset_index()
            for _, row in avg_stats.iterrows():
                diff_color = "red" if row["Diff"] > 0 else "green"
                avg_table.add_row(
                    row["Algorithm"],
                    row["Metric"],
                    f"[{diff_color}]{row['Diff']:+.2f}%[/{diff_color}]"
                )

            console.print(avg_table)
            # 2. Detailed Dataset Breakdown
            detail_table = Table(
                title=f"Sweep Details: Start vs End Values {title_range}",
                show_lines=True
            )
            detail_table.add_column("Algorithm", style="cyan", justify="left")
            detail_table.add_column("Dataset", style="magenta", justify="left")
            detail_table.add_column("Metric", style="green", justify="left")
            detail_table.add_column("Start Value", justify="right")
            detail_table.add_column("End Value", justify="right")
            detail_table.add_column("% Difference", justify="right")

            for stat in sweep_stats:
                diff_color = "red" if stat["Diff"] > 0 else "green"
                detail_table.add_row(
                    stat["Algorithm"],
                    stat["Dataset"],
                    stat["Metric"],
                    f"{stat['Start']:.4g}",
                    f"{stat['End']:.4g}",
                    f"[{diff_color}]{stat['Diff']:+.2f}%[/{diff_color}]"
                )

            console.print(detail_table)

            # 3. Save to CSV files
            detail_csv_path = out_dir / f"sweep_{param_name}_detailed_stats.csv"
            stats_df.to_csv(detail_csv_path, index=False)
            generated_files.append(detail_csv_path)

            avg_csv_path = out_dir / f"sweep_{param_name}_average_stats.csv"
            avg_stats.rename(columns={"Diff": "Avg % Difference"}).to_csv(avg_csv_path, index=False)
            generated_files.append(avg_csv_path)

            console.print(f"[green]✓[/green] Saved table data: [bold]{detail_csv_path.name}[/bold]")
            console.print(f"[green]✓[/green] Saved table data: [bold]{avg_csv_path.name}[/bold]")

        return generated_files