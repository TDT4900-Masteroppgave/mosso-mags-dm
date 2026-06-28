import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from matplotlib.lines import Line2D
import optuna

from scripts.analysis.plotters.base_plotter import FigureProfile, Plotter, register
from matplotlib.ticker import MaxNLocator


@register
class OptunaPlotter(Plotter):
    plotter_id = "pareto_front"
    description = "Optuna multi-objective visualizations for Bayesian tuning"
    DATASET_FIGURE_PROFILE = FigureProfile(
        figsize=(10.8, 6.075),
        font_size=16.0,
        label_size=19.0,
        tick_size=16.0,
        legend_size=15.0,
        line_width=3.0,
        marker_size=7.0,
        title_size=19.0,
        extra_rc={
            "axes.linewidth": 1.6,
            "xtick.major.size": 7.0,
            "ytick.major.size": 7.0,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
        },
    )
    GLOBAL_FIGURE_PROFILE = FigureProfile(
        figsize=(10.8, 6.075),
        font_size=16.0,
        label_size=19.0,
        tick_size=16.0,
        legend_size=15.0,
        line_width=3.0,
        marker_size=7.0,
        title_size=19.0,
        extra_rc={
            "axes.linewidth": 1.6,
            "xtick.major.size": 7.0,
            "ytick.major.size": 7.0,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
        },
    )

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        dataset_profile = self.use_figure_profile(
            self.DATASET_FIGURE_PROFILE,
            options,
            profile_name="pareto_dataset",
        )
        global_profile = self.get_figure_profile(
            self.GLOBAL_FIGURE_PROFILE,
            options,
            profile_name="pareto_global",
        )

        if "dataset" not in data.columns:
            data = data.copy()
            data["dataset"] = context

        datasets = data["dataset"].unique()
        generated_files = []
        pareto_data = []

        algo_adjustments = {
            "MoSSo": {"time_mult": 1.0, "ratio_mult": 1.0},
            "sm": {"time_mult": 1.0, "ratio_mult": 1.0},
            "sm_thr": {"time_mult": 1.0, "ratio_mult": 1.0},
        }

        data = data.copy().reset_index(drop=True)
        for algo, adj in algo_adjustments.items():
            mask = data["algorithm"] == algo
            if mask.any():
                data.loc[mask, "time"] *= adj.get("time_mult", 1.0)
                data.loc[mask, "ratio"] *= adj.get("ratio_mult", 1.0)

        time_label = options.get("pareto_time_label", "Execution Time (s)")
        time_col_name = "Time (s)"
        ratio_label = options.get("pareto_ratio_label", "Relative Size")
        ratio_col_name = "Relative Size"
        trial_alpha = options.get("pareto_trial_alpha", 0.16)
        trial_marker_size = options.get("pareto_trial_marker_size", 18.0)

        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        # ==========================================================
        # 1. GENERATE PER-DATASET ALGORITHM PLOTS
        # ==========================================================
        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy()
            if ds_data.empty: continue

            fig, ax = self.create_figure(dataset_profile)
            plotted_algos = 0
            pareto_t_min, pareto_t_max = float('inf'), float('-inf')
            pareto_r_min, pareto_r_max = float('inf'), float('-inf')

            for i, algo in enumerate(algos):
                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty: continue

                style = self.get_algo_style(algo)
                color = style["color"]
                mk = style["marker"]

                pareto_r_min = min(pareto_r_min, algo_data["ratio"].min())
                pareto_r_max = max(pareto_r_max, algo_data["ratio"].max())

                ax.scatter(
                    algo_data["time"],
                    algo_data["ratio"],
                    marker=mk,
                    s=trial_marker_size,
                    color=color,
                    alpha=trial_alpha,
                    linewidths=0,
                    zorder=2,
                    label="_nolegend_",
                    rasterized=True,
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
                        label=algo, color=color, linestyle="-",
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

            ax.set_xlabel(time_label, style='italic', labelpad=10)
            ax.set_ylabel(ratio_label, style='italic', labelpad=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", which="major", pad=7)
            ax.set_axisbelow(True)

            if pareto_t_max > float('-inf'):
                t_range_line = pareto_t_max - pareto_t_min
                t_pad_main = t_range_line * 0.03 if t_range_line > 0 else 0.1
                ax.set_xlim(max(0, pareto_t_min - t_pad_main), pareto_t_max + t_pad_main)
            if pareto_r_max > float('-inf'):
                r_range_line = pareto_r_max - pareto_r_min
                r_pad_main = r_range_line * 0.08 if r_range_line > 0 else 0.01
                ax.set_ylim(max(0, pareto_r_min - r_pad_main), pareto_r_max + r_pad_main)

            self.style_major_minor_grid(ax, minor_axis="x")
            self.despine(ax, top=True, right=True)

            if plotted_algos > 0:
                legend_elements = []
                for a in sorted(algos):
                    if a in ds_data["algorithm"].values:
                        style = self.get_algo_style(a)
                        legend_elements.append(Line2D([0], [0], color=style["color"], lw=1.5, marker=style["marker"], label=a))

                self.add_centered_legend(
                    ax,
                    handles=legend_elements, loc='upper center', y=-0.18,
                    ncol=min(plotted_algos, 5), handlelength=1.8, handletextpad=0.5, columnspacing=1.2
                )

            fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
            pdf_path = out_dir / f"pareto_front_{ds}.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            generated_files.extend([pdf_path])

        # ==========================================================
        # 2. GLOBAL FRONTIER BY ALGORITHM (ALL DATASETS IN ONE PLOT)
        # ==========================================================
        global_pareto_data = []
        fig_g, ax_g = self.create_figure(global_profile)
        plotted_global_algos = 0
        g_pareto_t_min, g_pareto_t_max = float('inf'), float('-inf')
        g_pareto_r_min, g_pareto_r_max = float('inf'), float('-inf')

        for i, algo in enumerate(algos):
            algo_data = data[data["algorithm"] == algo].copy()
            if algo_data.empty: continue

            style = self.get_algo_style(algo)
            color = style["color"]
            mk = style["marker"]

            g_pareto_r_min = min(g_pareto_r_min, algo_data["ratio"].min())
            g_pareto_r_max = max(g_pareto_r_max, algo_data["ratio"].max())

            ax_g.scatter(
                algo_data["time"],
                algo_data["ratio"],
                marker=mk,
                s=trial_marker_size,
                color=color,
                alpha=trial_alpha,
                linewidths=0,
                zorder=2,
                label="_nolegend_",
                rasterized=True,
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
                    label=algo, color=color, linestyle="-",
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

        ax_g.set_xlabel(time_label, style='italic', labelpad=10)
        ax_g.set_ylabel(ratio_label, style='italic', labelpad=12)
        ax_g.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_g.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_g.tick_params(axis="both", which="major", pad=7)
        ax_g.set_axisbelow(True)

        if g_pareto_t_max > float('-inf'):
            t_range_line = g_pareto_t_max - g_pareto_t_min
            t_pad_main = t_range_line * 0.03 if t_range_line > 0 else 0.1
            ax_g.set_xlim(max(0, g_pareto_t_min - t_pad_main), g_pareto_t_max + t_pad_main)
        if g_pareto_r_max > float('-inf'):
            r_range_line = g_pareto_r_max - g_pareto_r_min
            r_pad_main = r_range_line * 0.08 if r_range_line > 0 else 0.01
            ax_g.set_ylim(max(0, g_pareto_r_min - r_pad_main), g_pareto_r_max + r_pad_main)

        self.style_major_minor_grid(ax_g, minor_axis="x")
        self.despine(ax_g, top=True, right=True)

        if plotted_global_algos > 0:
            legend_elements = []
            for a in sorted(algos):
                if a in data["algorithm"].values:
                    style = self.get_algo_style(a)
                    legend_elements.append(Line2D([0], [0], color=style["color"], lw=1.5, marker=style["marker"], label=a))

            self.add_centered_legend(
                ax_g,
                handles=legend_elements, loc='upper center', y=-0.18,
                ncol=min(plotted_global_algos, 5), handlelength=1.8, handletextpad=0.5, columnspacing=1.2
            )

        ax_g.set_title("Global Pareto Frontier by Algorithm", pad=10, weight="bold")
        fig_g.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
        global_pdf = out_dir / "pareto_front_global.pdf"
        fig_g.savefig(global_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig_g)

        generated_files.extend([global_pdf])

        # ==========================================================
        # 3. STATISTICAL TABLES EXPORT
        # ==========================================================
        txt_path = out_dir / f"pareto_summary_tables.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            file_console = Console(file=f, width=1000)

            if pareto_data:
                pareto_df_export = pd.DataFrame(pareto_data)

                base_cols = ["Dataset", "Algorithm", ratio_col_name, time_col_name]
                dynamic_cols = [c for c in pareto_df_export.columns if c not in base_cols]
                all_cols = base_cols + dynamic_cols
                pareto_df_export = pareto_df_export[all_cols]

                table = Table(title="All Pareto Optimal Configurations (Per Dataset)", box=box.SIMPLE, show_header=True, header_style="bold yellow")
                for col in all_cols:
                    if col == "Dataset": table.add_column(col, style="cyan")
                    elif col == "Algorithm": table.add_column(col, style="green", no_wrap=True)
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

                file_console.print(table)

            if global_pareto_data:
                g_df_export = pd.DataFrame(global_pareto_data)
                base_cols_g = ["Algorithm", "Dataset", ratio_col_name, time_col_name]
                dynamic_cols_g = [c for c in g_df_export.columns if c not in base_cols_g]
                all_cols_g = base_cols_g + dynamic_cols_g
                g_df_export = g_df_export[all_cols_g]

                g_table = Table(title="Global Pareto Optimal Configurations (Per Algorithm)", box=box.SIMPLE, show_header=True, header_style="bold yellow")
                for col in all_cols_g:
                    if col == "Algorithm": g_table.add_column(col, style="green", no_wrap=True)
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

                if pareto_data:
                    file_console.print()
                file_console.print(g_table)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        generated_files.append(txt_path)

        return generated_files
