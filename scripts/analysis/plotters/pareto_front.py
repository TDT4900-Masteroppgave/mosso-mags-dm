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

        # --- MANUAL DATA ADJUSTMENTS ---
        algo_adjustments = {
            "kdd20-mosso": {"time_mult": 1.0, "ratio_mult": 1.0},
            "sm": {"time_mult": 1.0, "ratio_mult": 1.0},
            "sm_thr": {"time_mult": 1.0, "ratio_mult": 1.0},
        }

        data = data.copy().reset_index(drop=True)
        for algo, adj in algo_adjustments.items():
            mask = data["algorithm"] == algo
            if mask.any():
                data.loc[mask, "time"] *= adj.get("time_mult", 1.0)
                data.loc[mask, "ratio"] *= adj.get("ratio_mult", 1.0)
        # -------------------------------

        # --- NORMALIZATION TO BASELINE (BOTH AXES) ---
        baseline_algo = "kdd20-mosso"
        normalized_x = False
        normalized_y = False

        for ds in datasets:
            ds_mask = data['dataset'] == ds
            baseline_mask = ds_mask & (data['algorithm'] == baseline_algo)

            # Find the anchor values (mean baseline performance, or max if baseline missing)
            if baseline_mask.any():
                baseline_time = data.loc[baseline_mask, 'time'].mean()
                baseline_ratio = data.loc[baseline_mask, 'ratio'].mean()
            else:
                baseline_time = data.loc[ds_mask, 'time'].max()
                baseline_ratio = data.loc[ds_mask, 'ratio'].max()

            # Normalize X (Time)
            if baseline_time and baseline_time > 0:
                data.loc[ds_mask, 'time'] = data.loc[ds_mask, 'time'] / baseline_time
                normalized_x = True

            # Normalize Y (Ratio)
            if baseline_ratio and baseline_ratio > 0:
                data.loc[ds_mask, 'ratio'] = data.loc[ds_mask, 'ratio'] / baseline_ratio
                normalized_y = True

        if normalized_x:
            time_label = "normalized time (vs baseline)"
            time_col_name = "Norm. Time"
        else:
            time_label = options.get("time_label", "time (seconds)").lower()
            time_col_name = "Time (s)"

        if normalized_y:
            ratio_label = "normalized size (vs baseline)"
            ratio_col_name = "Norm. Size"
        else:
            ratio_label = "relative size"
            ratio_col_name = "Relative Size"
        # ---------------------------------------------

        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        # ==========================================================
        # 1. GENERATE PER-DATASET ALGORITHM PLOTS
        # ==========================================================
        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy()
            if ds_data.empty: continue

            fig, ax = plt.subplots(figsize=(7, 4.5))
            plotted_algos = 0
            pareto_t_min, pareto_t_max = float('inf'), float('-inf')

            # Add reference lines for the 1.0 baseline intersection
            if normalized_x and normalized_y:
                ax.axhline(1.0, color="#CBD5E1", linestyle="--", linewidth=1.0, zorder=1)
                ax.axvline(1.0, color="#CBD5E1", linestyle="--", linewidth=1.0, zorder=1)

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

                    plotted_algos += 1
                    pareto_t_min = min(pareto_t_min, pareto_df["time"].min())
                    pareto_t_max = max(pareto_t_max, pareto_df["time"].max())

                    for _, pt in pareto_df.iterrows():
                        pt_info = {
                            "Dataset": ds,
                            "Algorithm": algo,
                            ratio_col_name: pt["ratio"],
                            time_col_name: pt["time"]
                        }
                        for p in param_cols:
                            val = pt.get(p)
                            if pd.notna(val):
                                if isinstance(val, float):
                                    if val.is_integer(): val = int(val)
                                    elif p == "thr_end": val = round(val, 2)
                                pt_info[p] = val
                        pareto_data.append(pt_info)

            import seaborn as sns
            ax.set_xlabel(time_label, fontsize=12, style='italic')
            ax.set_ylabel(ratio_label, fontsize=12, style='italic')

            if pareto_t_max > float('-inf'):
                t_range_line = pareto_t_max - pareto_t_min
                t_pad_main = t_range_line * 0.03 if t_range_line > 0 else 0.1
                ax.set_xlim(max(0, pareto_t_min - t_pad_main), pareto_t_max + t_pad_main)

            ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.4, alpha=0.3)
            sns.despine(ax=ax, top=True, right=True)

            if plotted_algos > 0:
                legend_elements = []
                for a in sorted(algos):
                    if a in ds_data["algorithm"].values:
                        style = self.get_algo_style(a)
                        legend_elements.append(Line2D([0], [0], color=style["color"], lw=1.5, marker=style["marker"], label=a))

                ax.legend(
                    handles=legend_elements, title="", loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=plotted_algos, handlelength=1.5, handletextpad=0.4, columnspacing=1.0, frameon=False, fontsize=9
                )

            fig.tight_layout()
            out_path = out_dir / f"pareto_front_{ds}.png"
            pdf_path = out_dir / f"pareto_front_{ds}.pdf"
            fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            generated_files.extend([out_path, pdf_path])

        # ==========================================================
        # 2. GLOBAL FRONTIER BY ALGORITHM (ALL DATASETS IN ONE PLOT)
        # ==========================================================
        global_pareto_data = []
        fig_g, ax_g = plt.subplots(figsize=(7, 4.5))
        plotted_global_algos = 0
        g_pareto_t_min, g_pareto_t_max = float('inf'), float('-inf')

        # Add crosshairs for the (1.0, 1.0) baseline center
        if normalized_x and normalized_y:
            ax_g.axhline(1.0, color="#CBD5E1", linestyle="--", linewidth=1.2, zorder=1)
            ax_g.axvline(1.0, color="#CBD5E1", linestyle="--", linewidth=1.2, zorder=1)
            # Add a distinct marker for the baseline center
            ax_g.plot([1.0], [1.0], marker="+", markersize=12, color="gray", zorder=4)

        for i, algo in enumerate(algos):
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            style = self.get_algo_style(algo)
            color = style["color"]
            mk = style["marker"]

            ax_g.plot(
                algo_data["time"], algo_data["ratio"],
                marker=mk, markersize=3, color=color,
                linestyle="none", alpha=0.25, zorder=2,
                label="_nolegend_"
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

                ax_g.plot(
                    pareto_df["time"], pareto_df["ratio"],
                    label=algo, color=color, linestyle="-", linewidth=1.5,
                    alpha=0.9, zorder=3
                )

                plotted_global_algos += 1
                g_pareto_t_min = min(g_pareto_t_min, pareto_df["time"].min())
                g_pareto_t_max = max(g_pareto_t_max, pareto_df["time"].max())

                for _, pt in pareto_df.iterrows():
                    pt_info = {
                        "Algorithm": algo,
                        "Dataset": pt.get("dataset", "Unknown"),
                        ratio_col_name: pt["ratio"],
                        time_col_name: pt["time"]
                    }
                    for p in param_cols:
                        val = pt.get(p)
                        if pd.notna(val):
                            if isinstance(val, float):
                                if val.is_integer(): val = int(val)
                                elif p == "thr_end": val = round(val, 2)
                            pt_info[p] = val
                    global_pareto_data.append(pt_info)

        ax_g.set_xlabel(time_label, fontsize=12, style='italic')
        ax_g.set_ylabel(ratio_label, fontsize=12, style='italic')

        if g_pareto_t_max > float('-inf'):
            t_range_line = g_pareto_t_max - g_pareto_t_min
            t_pad_main = t_range_line * 0.03 if t_range_line > 0 else 0.1
            ax_g.set_xlim(max(0, g_pareto_t_min - t_pad_main), g_pareto_t_max + t_pad_main)

        ax_g.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_g.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.4, alpha=0.3)
        import seaborn as sns
        sns.despine(ax=ax_g, top=True, right=True)

        if plotted_global_algos > 0:
            legend_elements = []
            for a in sorted(algos):
                if a in data["algorithm"].values:
                    style = self.get_algo_style(a)
                    legend_elements.append(Line2D([0], [0], color=style["color"], lw=1.5, marker=style["marker"], label=a))

            ax_g.legend(
                handles=legend_elements, title="", loc='upper center', bbox_to_anchor=(0.5, -0.15),
                ncol=plotted_global_algos, handlelength=1.5, handletextpad=0.4, columnspacing=1.0, frameon=False, fontsize=9
            )

        plt.title("Global Pareto Frontier by Algorithm", fontsize=12, pad=10, weight="bold")
        fig_g.tight_layout()
        global_png = out_dir / "pareto_front_global.png"
        global_pdf = out_dir / "pareto_front_global.pdf"
        fig_g.savefig(global_png, format="png", dpi=300, bbox_inches="tight")
        fig_g.savefig(global_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig_g)

        generated_files.extend([global_png, global_pdf])

        # ==========================================================
        # 3. STATISTICAL TABLES EXPORT
        # ==========================================================
        console = Console(record=True)

        # 3.1 Per Dataset Table
        if pareto_data:
            pareto_df_export = pd.DataFrame(pareto_data)

            base_cols = ["Dataset", "Algorithm", ratio_col_name, time_col_name]
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
                    elif col in [ratio_col_name, time_col_name]: row_data.append(f"{val:.4f}")
                    elif col == "thr_end" and isinstance(val, (int, float)): row_data.append(f"{val:.2f}")
                    else: row_data.append(str(val))
                table.add_row(*row_data)

            console.print(table)

        # 3.2 Global Table (Per Algorithm)
        if global_pareto_data:
            g_df_export = pd.DataFrame(global_pareto_data)
            base_cols_g = ["Algorithm", "Dataset", ratio_col_name, time_col_name]
            dynamic_cols_g = [c for c in g_df_export.columns if c not in base_cols_g]
            all_cols_g = base_cols_g + dynamic_cols_g
            g_df_export = g_df_export[all_cols_g]

            g_table = Table(title="Global Pareto Optimal Configurations (Per Algorithm)", box=box.SIMPLE, show_header=True, header_style="bold yellow")
            for col in all_cols_g:
                if col == "Algorithm": g_table.add_column(col, style="green")
                elif col == "Dataset": g_table.add_column(col, style="cyan")
                elif col in base_cols_g: g_table.add_column(col, justify="right")
                else: g_table.add_column(str(col), justify="right", style="magenta")

            for _, row in g_df_export.iterrows():
                row_data = []
                for col in all_cols_g:
                    val = row.get(col)
                    if pd.isna(val): row_data.append("-")
                    elif col in [ratio_col_name, time_col_name]: row_data.append(f"{val:.4f}")
                    elif col == "thr_end" and isinstance(val, (int, float)): row_data.append(f"{val:.2f}")
                    else: row_data.append(str(val))
                g_table.add_row(*row_data)

            console.print("\n")
            console.print(g_table)

        # Output Text File
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        txt_path = out_dir / f"pareto_summary_tables.txt"
        console.save_text(str(txt_path))
        generated_files.append(txt_path)

        return generated_files