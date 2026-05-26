import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from matplotlib.lines import Line2D
import optuna

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class OptunaPlotter(Plotter):
    plotter_id = "pareto_front"
    description = "Optuna multi-objective visualizations for Bayesian tuning"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()

        if "dataset" not in data.columns:
            data = data.copy()
            data["dataset"] = context

        datasets = data["dataset"].unique()
        generated_files = []
        pareto_data = []

        time_label = options.get("time_label", "time (seconds)").lower()

        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy()
            if ds_data.empty: continue

            fig, ax = plt.subplots(figsize=(7, 4.5))
            axins = ax.inset_axes([0.45, 0.45, 0.5, 0.5])

            plotted_algos = 0

            pareto_t_min, pareto_t_max = float('inf'), float('-inf')
            pareto_r_min, pareto_r_max = float('inf'), float('-inf')

            kdd_t_min, kdd_t_max = float('inf'), float('-inf')
            kdd_r_min, kdd_r_max = float('inf'), float('-inf')

            for i, algo in enumerate(algos):
                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty: continue

                style = self.get_algo_style(algo)
                color = style["color"]
                mk = style["marker"]

                ax.plot(
                    algo_data["time"], algo_data["ratio"],
                    marker=mk, markersize=3, color=color,
                    linestyle="none", alpha=0.25, zorder=2,
                    label="_nolegend_"
                )
                axins.plot(
                    algo_data["time"], algo_data["ratio"],
                    marker=mk, markersize=3, color=color,
                    linestyle="none", alpha=0.35, zorder=2
                )

                algo_data = algo_data.sort_values(by=["ratio", "time"])
                pareto_front = []
                min_time = float('inf')

                for _, row in algo_data.iterrows():
                    if row["time"] < min_time:
                        pareto_front.append(row)
                        min_time = row["time"]

                if pareto_front:
                    pareto_df = pd.DataFrame(pareto_front).reset_index(drop=True)
                    pareto_df = pareto_df.sort_values(by="time")

                    ax.plot(
                        pareto_df["time"], pareto_df["ratio"],
                        label=algo, color=color, linestyle="-", linewidth=1.5,
                        alpha=0.9, zorder=3
                    )
                    axins.plot(
                        pareto_df["time"], pareto_df["ratio"],
                        color=color, linestyle="-", linewidth=2.0,
                        alpha=0.9, zorder=3
                    )

                    plotted_algos += 1

                    if algo == "kdd20-mosso":
                        kdd_t_min = min(kdd_t_min, pareto_df["time"].min())
                        kdd_t_max = max(kdd_t_max, pareto_df["time"].max())
                        kdd_r_min = min(kdd_r_min, pareto_df["ratio"].min())
                        kdd_r_max = max(kdd_r_max, pareto_df["ratio"].max())

                    pareto_t_min = min(pareto_t_min, pareto_df["time"].min())
                    pareto_t_max = max(pareto_t_max, pareto_df["time"].max())
                    pareto_r_min = min(pareto_r_min, pareto_df["ratio"].min())
                    pareto_r_max = max(pareto_r_max, pareto_df["ratio"].max())

                    for _, pt in pareto_df.iterrows():
                        pt_info = {
                            "Dataset": ds,
                            "Algorithm": algo,
                            "Relative Size": pt["ratio"],
                            "Time (s)": pt["time"]
                        }

                        for p in param_cols:
                            val = pt.get(p)
                            if pd.notna(val):
                                if isinstance(val, float):
                                    if val.is_integer():
                                        val = int(val)
                                    elif p == "thr_end":
                                        val = round(val, 2)
                                pt_info[p] = val

                        pareto_data.append(pt_info)

            plt.xlabel(time_label, fontsize=14, style='italic')
            plt.ylabel("relative size", fontsize=14, style='italic')

            # Ensure we have Pareto line data points to bound our axes
            if pareto_t_max > float('-inf'):

                # --- 1. MAIN WINDOW LIMITS (Strictly bound to the Pareto lines for X) ---
                # Calculate margins relative only to the Pareto front's true footprint
                t_range_line = pareto_t_max - pareto_t_min
                t_pad_main = t_range_line * 0.03 if t_range_line > 0 else 1.0

                # Limit X-axis strictly to the maximum point on the Pareto frontiers
                ax.set_xlim(max(0, pareto_t_min - t_pad_main), pareto_t_max + t_pad_main)

                # (Optional) We remove ax.set_ylim(...) here so Matplotlib can
                # automatically scale the Y-axis to include all background points.

                # --- 2. INSET WINDOW LIMITS (Strictly bound to kdd20-mosso Pareto line) ---
                # Check if kdd20-mosso Pareto data was captured, fallback to global Pareto lines if not
                use_inset_t_min = kdd_t_min if kdd_t_min < float('inf') else pareto_t_min
                use_inset_t_max = kdd_t_max if kdd_t_max > float('-inf') else pareto_t_max
                use_inset_r_min = kdd_r_min if kdd_r_min < float('inf') else pareto_r_min
                use_inset_r_max = kdd_r_max if kdd_r_max > float('-inf') else pareto_r_max

                # Add a tiny 2% padding for the inset window
                t_inset_pad = (use_inset_t_max - use_inset_t_min) * 0.02 if use_inset_t_max > use_inset_t_min else 1.0
                r_inset_pad = (use_inset_r_max - use_inset_r_min) * 0.02 if use_inset_r_max > use_inset_r_min else 0.01

                axins.set_xlim(max(0, use_inset_t_min - t_inset_pad), use_inset_t_max + t_inset_pad)
                axins.set_ylim(max(0, use_inset_r_min - r_inset_pad), use_inset_r_max + r_inset_pad)

                # --- FORCE FINER Y-TICKS (0.01 steps) ---
                import matplotlib.ticker as ticker
                axins.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

                ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, linewidth=1.5)
                axins.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
                axins.tick_params(axis='both', which='major', labelsize=8)

            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

            if plotted_algos > 0:
                legend_elements = []
                for a in sorted(algos):
                    if a in ds_data["algorithm"].values:
                        style = self.get_algo_style(a)
                        legend_elements.append(Line2D([0], [0], color=style["color"], lw=1.5, marker=style["marker"], label=a))

                ax.legend(
                    handles=legend_elements,
                    title="",
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=plotted_algos,
                    handlelength=1.5,
                    handletextpad=0.4,
                    columnspacing=1.0,
                    frameon=False,
                    fontsize=9
                )

            fig.tight_layout()

            out_path = out_dir / f"pareto_front_{ds}.png"
            fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            generated_files.append(out_path)

        if pareto_data:
            console = Console(record=True)
            pareto_df_export = pd.DataFrame(pareto_data)

            base_cols = ["Dataset", "Algorithm", "Relative Size", "Time (s)"]
            dynamic_cols = [c for c in pareto_df_export.columns if c not in base_cols]
            all_cols = base_cols + dynamic_cols
            pareto_df_export = pareto_df_export[all_cols]

            table = Table(title="All Pareto Optimal Configurations (Per Dataset)", box=box.SIMPLE, show_header=True, header_style="bold yellow")
            for col in all_cols:
                if col == "Dataset": table.add_column(col, style="cyan")
                elif col == "Algorithm": table.add_column(col, style="green")
                elif col in base_cols: table.add_column(col, justify="right")
                else: table.add_column(str(col), justify="right", style="magenta")

            for _, row in pareto_df_export.iterrows():
                row_data = []
                for col in all_cols:
                    val = row.get(col)
                    if pd.isna(val): row_data.append("-")
                    elif col in ["Relative Size", "Time (s)"]: row_data.append(f"{val:.4f}")
                    elif col == "thr_end" and isinstance(val, (int, float)): row_data.append(f"{val:.2f}")
                    else: row_data.append(str(val))
                table.add_row(*row_data)

            console.print(table)

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            txt_path = out_dir / f"pareto_summary_tables.txt"
            console.save_text(str(txt_path))
            generated_files.append(txt_path)

        return generated_files