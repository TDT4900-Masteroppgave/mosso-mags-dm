import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class OptunaPlotter(Plotter):
    plotter_id = "pareto_front"
    description = "Optuna multi-objective visualizations for Bayesian tuning"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        self.set_chart_theme()

        # Handle context if aggregated (e.g., "average")
        if "dataset" not in data.columns:
            data = data.copy()
            data["dataset"] = context

        datasets = data["dataset"].unique()
        generated_files = []
        knee_points_data = [] # Store knee points for the summary table

        # Match paper's lowercase italic label styling
        time_label = options.get("time_label", "time (seconds)").lower()

        # Dynamically identify parameter columns (anything that isn't a core metric)
        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy()
            if ds_data.empty: continue

            fig, ax = plt.subplots(figsize=(6, 3.5))

            # Neutral, dark academic colors
            colors = ["#222222", "#4A4A4A", "#2C3E50", "#4A4E69", "#34495E", "#555555"]

            plotted_algos = 0
            plotted_knees = False

            for i, algo in enumerate(algos):
                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty: continue

                # Calculate Pareto Frontier
                algo_data = algo_data.sort_values(by=["ratio", "time"])
                pareto_front = []
                min_time = float('inf')

                for _, row in algo_data.iterrows():
                    if row["time"] < min_time:
                        pareto_front.append(row)
                        min_time = row["time"]

                if pareto_front:
                    pareto_df = pd.DataFrame(pareto_front).reset_index(drop=True)
                    color = colors[i % len(colors)]

                    ax.plot(
                        pareto_df["ratio"], pareto_df["time"],
                        label=algo,
                        color=color,
                        linestyle="-",
                        linewidth=2.0,
                        zorder=4
                    )
                    plotted_algos += 1

                    if len(pareto_df) > 1:
                        # Normalize Ratio (Linear Space)
                        r_min, r_max = pareto_df["ratio"].min(), pareto_df["ratio"].max()
                        r_range = r_max - r_min if r_max > r_min else 1.0
                        r_norm = (pareto_df["ratio"] - r_min) / r_range

                        # Normalize Time (Log Space)
                        log_t = np.log10(pareto_df["time"].clip(lower=1e-10))
                        t_min, t_max = log_t.min(), log_t.max()
                        t_range = t_max - t_min if t_max > t_min else 1.0
                        t_norm = (log_t - t_min) / t_range

                        # Find point closest to the ideal origin (0, 0)
                        dist_sq = r_norm**2 + t_norm**2
                        knee_idx = dist_sq.idxmin()
                        knee_point = pareto_df.loc[knee_idx]

                        knee_info = {
                            "Dataset": ds,
                            "Algorithm": algo,
                            "Relative Size": knee_point["ratio"],
                            "Time (s)": knee_point["time"]
                        }

                        for p in param_cols:
                            val = knee_point.get(p)
                            if pd.notna(val):
                                # Clean up formatting (e.g., 120.0 -> 120)
                                if isinstance(val, float) and val.is_integer():
                                    val = int(val)
                                knee_info[p] = val

                        knee_points_data.append(knee_info)

                        ax.plot(
                            knee_point["ratio"], knee_point["time"],
                            marker='o',
                            markersize=7,
                            markerfacecolor='white',
                            markeredgecolor=color,
                            markeredgewidth=1.8,
                            linestyle="none",
                            zorder=5,
                            label="Knee Point" if not plotted_knees else None
                        )
                        plotted_knees = True
                    # ------------------------------

            # Academic italicized labels
            plt.xlabel("relative size", fontsize=14, style='italic')
            plt.ylabel(time_label, fontsize=14, style='italic')

            # Uncluster the data
            ax.set_yscale("log")

            # Subtle grid
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.4)

            if plotted_algos > 0:
                plt.legend(
                    title="",
                    loc='upper right',
                    frameon=False
                )

            fig.tight_layout()

            out_path = out_dir / f"pareto_front_{ds}_{ts}.png"
            fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            generated_files.append(out_path)

        if knee_points_data:
            console = Console(record=True)
            knee_df = pd.DataFrame(knee_points_data)

            base_cols = ["Dataset", "Algorithm", "Relative Size", "Time (s)"]
            dynamic_cols = [c for c in knee_df.columns if c not in base_cols]
            all_cols = base_cols + dynamic_cols
            knee_df = knee_df[all_cols]

            csv_path = out_dir / f"pareto_knee_points_{ts}.csv"
            knee_df.to_csv(csv_path, index=False)
            generated_files.append(csv_path)

            # Print standard table
            table = Table(title="Pareto Optimal Knee Points", box=box.SIMPLE, show_header=True, header_style="bold yellow")
            for col in all_cols:
                if col == "Dataset": table.add_column(col, style="cyan")
                elif col == "Algorithm": table.add_column(col, style="green")
                elif col in base_cols: table.add_column(col, justify="right")
                else: table.add_column(str(col), justify="right", style="magenta")

            for _, row in knee_df.iterrows():
                row_data = []
                for col in all_cols:
                    val = row.get(col)
                    if pd.isna(val):
                        row_data.append("-")
                    elif col in ["Relative Size", "Time (s)"]:
                        row_data.append(f"{val:.4f}")
                    else:
                        row_data.append(str(val))
                table.add_row(*row_data)

            console.print("\n")
            console.print(table)

            if len(datasets) > 1:
                # Group by Algorithm, calculate mean for numeric columns
                avg_df = knee_df.drop(columns=["Dataset"]).groupby("Algorithm").mean(numeric_only=True).reset_index()

                # Save Average CSV
                avg_csv_path = out_dir / f"pareto_average_parameters_{ts}.csv"
                avg_df.to_csv(avg_csv_path, index=False)
                generated_files.append(avg_csv_path)

                # Print Average Table
                avg_table = Table(title="Average Optimal Parameters Across All Datasets", box=box.SIMPLE, show_header=True, header_style="bold green")
                avg_table.add_column("Algorithm", style="green")

                avg_cols = [c for c in avg_df.columns if c != "Algorithm"]
                for col in avg_cols:
                    if col in ["Relative Size", "Time (s)"]:
                        avg_table.add_column(f"Avg {col}", justify="right")
                    else:
                        avg_table.add_column(f"Avg {col}", justify="right", style="magenta")

                for _, row in avg_df.iterrows():
                    row_data = [str(row["Algorithm"])]
                    for col in avg_cols:
                        val = row[col]
                        if pd.isna(val):
                            row_data.append("-")
                        elif col in ["Relative Size", "Time (s)"]:
                            row_data.append(f"{val:.4f}")
                        else:
                            row_data.append(f"{val:.2f}")
                    avg_table.add_row(*row_data)

                console.print("\n")
                console.print(avg_table)

            txt_path = out_dir / f"pareto_summary_tables_{ts}.txt"
            console.save_text(str(txt_path))
            generated_files.append(txt_path)

        return generated_files