import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gmean

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class OptunaTradeoffPlotter(Plotter):
    plotter_id = "pareto_tradeoff"
    description = "Extracts tradeoff points and generates raw/comparison tables (per dataset and global average) vs MoSSo"

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

        raw_tradeoff_data = []
        comparison_data = []

        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros', 'dist_to_ideal'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        # Define split categories for Table 2
        comp_categories_1 = ["Best vs Best", "Fastest vs Fastest", "Balanced vs Balanced"]
        comp_categories_2 = ["Default vs Balanced"] + [f"Default vs {i}% Loss" for i in range(5, -1, -1)]
        comp_categories = comp_categories_1 + comp_categories_2

        # ==========================================================
        # EXTRACTION & PLOTTING LOOP
        # ==========================================================
        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy().reset_index(drop=True)
            if ds_data.empty: continue

            # --- 1. Extract MoSSo Baseline Points ---
            mosso_names = {"kdd20-mosso", "MoSSo"}
            mosso_data = ds_data[ds_data["algorithm"].isin(mosso_names)].copy()

            if mosso_data.empty:
                print(f"[Warning] MoSSo not found in dataset '{ds}'. Skipping.")
                continue

            # Identify MoSSo Default (c=120, e=3)
            if "c" not in mosso_data.columns or "e" not in mosso_data.columns:
                print(f"[Warning] MoSSo default columns c/e missing in '{ds}'. Skipping.")
                continue

            c_vals = pd.to_numeric(mosso_data["c"], errors="coerce")
            e_vals = pd.to_numeric(mosso_data["e"], errors="coerce")
            default_mask = (c_vals == 120) & (e_vals == 3)

            default_runs = mosso_data[default_mask]
            if default_runs.empty:
                print(f"[Warning] MoSSo default (c=120, e=3) missing in '{ds}'. Skipping.")
                continue

            mosso_default_pt = default_runs.sort_values(by=["ratio", "time"]).iloc[0]
            def_t, def_r = mosso_default_pt["time"], mosso_default_pt["ratio"]

            # Normalize distance to Ideal (0,0) based on default params
            mosso_data["dist_to_ideal"] = np.sqrt((mosso_data["time"] / def_t)**2 + (mosso_data["ratio"] / def_r)**2)

            mosso_points = {
                "Default": mosso_default_pt,
                "Best Compression": mosso_data.loc[mosso_data["ratio"].idxmin()],
                "Fastest Time": mosso_data.loc[mosso_data["time"].idxmin()],
                "Balanced": mosso_data.loc[mosso_data["dist_to_ideal"].idxmin()]
            }

            mosso_label = str(mosso_data["algorithm"].iloc[0])
            for profile, pt in mosso_points.items():
                raw_tradeoff_data.append(
                    self._format_raw_row(ds, mosso_label, profile, pt, param_cols)
                )

            # --- Plot Setup ---
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.axhline(def_r, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)
            ax.axvline(def_t, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)

            mosso_pareto = self._pareto_front(mosso_data)
            if not mosso_pareto.empty:
                ax.plot(mosso_pareto["time"], mosso_pareto["ratio"], color="gray", linestyle="-",
                        linewidth=1.8, alpha=0.85, zorder=3)

            ax.plot(mosso_data["time"], mosso_data["ratio"], marker=".", color="gray", linestyle="none", alpha=0.15, zorder=2)
            ax.plot(def_t, def_r, marker="*", markersize=14, color="gold", markeredgecolor="black", zorder=10, label="MoSSo Default")

            # --- 2. Extract Other Algorithms ---
            for algo in algos:
                if algo in mosso_names: continue

                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty: continue

                color = self.get_algo_style(algo)["color"]

                algo_pareto = self._pareto_front(algo_data)
                if not algo_pareto.empty:
                    ax.plot(algo_pareto["time"], algo_pareto["ratio"], color=color, linestyle="-",
                            linewidth=1.8, alpha=0.85, zorder=3)

                ax.plot(algo_data["time"], algo_data["ratio"], marker=".", color=color, linestyle="none", alpha=0.15, zorder=2)

                algo_data["dist_to_ideal"] = np.sqrt((algo_data["time"] / def_t)**2 + (algo_data["ratio"] / def_r)**2)

                algo_points = {
                    "Best Compression": algo_data.loc[algo_data["ratio"].idxmin()],
                    "Fastest Time": algo_data.loc[algo_data["time"].idxmin()],
                    "Balanced": algo_data.loc[algo_data["dist_to_ideal"].idxmin()]
                }

                # Extract Compression Loss Profiles (5% down to 0%)
                for loss_pct in range(5, -1, -1):
                    target_ratio = def_r * (1.0 + (loss_pct / 100.0))
                    valid_pts = algo_data[algo_data["ratio"] <= target_ratio]
                    if not valid_pts.empty:
                        algo_points[f"{loss_pct}% Loss"] = valid_pts.loc[valid_pts["time"].idxmin()]

                # Save raw points and plot
                # Expanded markers to accommodate new profiles
                markers = {
                    "Best Compression": "v", "Fastest Time": "<", "Balanced": "D",
                    "5% Loss": "p", "4% Loss": "h", "3% Loss": "H",
                    "2% Loss": "8", "1% Loss": "s", "0% Loss": "o"
                }

                for profile, pt in algo_points.items():
                    raw_tradeoff_data.append(self._format_raw_row(ds, algo, profile, pt, param_cols))
                    if profile in markers:
                        ax.plot(pt["time"], pt["ratio"], marker=markers[profile], markersize=8,
                                color=color, markeredgecolor="black", markeredgewidth=0.8, alpha=0.9, zorder=5)

                # --- 3. Build Comparison Row ---
                comp_row = {"Dataset": ds, "Algorithm": algo}

                def add_comp_metrics(col_name, base_pt, target_pt):
                    if target_pt is None:
                        comp_row[f"{col_name}_speed"] = np.nan
                        comp_row[f"{col_name}_size"] = np.nan
                    else:
                        comp_row[f"{col_name}_speed"] = base_pt["time"] / target_pt["time"]
                        comp_row[f"{col_name}_size"] = target_pt["ratio"] / base_pt["ratio"]

                # Group 1 Comparisons
                add_comp_metrics("Best vs Best", mosso_points.get("Best Compression"), algo_points.get("Best Compression"))
                add_comp_metrics("Fastest vs Fastest", mosso_points.get("Fastest Time"), algo_points.get("Fastest Time"))
                add_comp_metrics("Balanced vs Balanced", mosso_points.get("Balanced"), algo_points.get("Balanced"))

                # Group 2 Comparisons
                add_comp_metrics("Default vs Balanced", mosso_points.get("Default"), algo_points.get("Balanced"))
                for i in range(5, -1, -1):
                    add_comp_metrics(f"Default vs {i}% Loss", mosso_points.get("Default"), algo_points.get(f"{i}% Loss"))

                comparison_data.append(comp_row)

            # --- Finalize Plot ---
            ax.set_xlabel(options.get("time_label", "Time (seconds)"), fontsize=12, style='italic')
            ax.set_ylabel("Relative Size (Compression Ratio)", fontsize=12, style='italic')
            ax.set_title(f"Pareto Tradeoff Extractions - {ds}", pad=15, weight="bold")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            sns.despine(ax=ax)

            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='-', linewidth=1.8, label='Pareto frontier'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=12, label='MoSSo Default'),
            ]

            # Simplified legend for clean plot mapping
            for prof, mk in [("Best Comp.", "v"), ("Fastest", "<"), ("Balanced", "D"), ("0% Loss", "o"), ("5% Loss", "p")]:
                legend_elements.append(Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=prof))

            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)

            fig.tight_layout()
            svg_path = out_dir / f"tradeoffs_{ds}.svg"
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
            plt.close(fig)
            generated_files.append(svg_path)

        # ==========================================================
        # RICH TABLES EXPORT
        # ==========================================================
        console = Console(record=True)

        def format_comp_str(speed, size):
            if pd.isna(speed) or pd.isna(size):
                return "-"
            return f"{speed:.2f}x speed, {size:.3f}x size"

        def numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
            cols: list[str] = []
            for col in candidates:
                if col not in df.columns:
                    continue
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().any():
                    df[col] = converted
                    cols.append(col)
            return cols

        def safe_gmean(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            values = values[values > 0]
            return float(gmean(values)) if not values.empty else np.nan

        # --- Table 1: RAW VALUES (Per Dataset & Global) ---
        if raw_tradeoff_data:
            df_raw = pd.DataFrame(raw_tradeoff_data)
            base_cols = ["Algorithm", "Profile", "Ratio", "Time (s)"]
            all_cols = ["Dataset"] + base_cols + [c for c in df_raw.columns if c not in ["Dataset"] + base_cols]

            # 1a. Per Dataset (Kept intact as it tracks everything raw)
            t1a = Table(title="Table 1a: Raw Operational Points (Per Dataset)", box=box.SIMPLE, header_style="bold yellow")
            for col in all_cols:
                t1a.add_column(col, justify="right" if col in ["Ratio", "Time (s)"] else "left")
            for _, row in df_raw.iterrows():
                row_vals = [f"{row.get(c):.5f}" if c in ["Ratio", "Time (s)"] else str(row.get(c, "-")) for c in all_cols]
                t1a.add_row(*row_vals)
            console.print(t1a)
            console.print("\n")

            # 1b. Global Averages (Split into two)
            candidate_avg_cols = ["Ratio", "Time (s)"] + [c for c in param_cols if c in df_raw.columns]
            cols_to_average = numeric_columns(df_raw, candidate_avg_cols)
            df_raw_global = df_raw.groupby(["Algorithm", "Profile"], observed=False)[cols_to_average].mean().reset_index()
            header_cols = ["Algorithm", "Profile"] + cols_to_average

            # Group 1 Profiles
            g1_profiles = ["Best Compression", "Fastest Time", "Balanced"]
            df_g1 = df_raw_global[df_raw_global["Profile"].isin(g1_profiles)]

            t1b_1 = Table(title="Table 1b-1: GLOBAL AVERAGES - Core Profiles", box=box.SIMPLE, header_style="bold yellow")
            for col in header_cols:
                t1b_1.add_column(col, justify="right" if col in ["Ratio", "Time (s)"] else "left")
            for _, row in df_g1.iterrows():
                row_vals = []
                for c in header_cols:
                    val = row.get(c)
                    if c in ["Ratio", "Time (s)"]: row_vals.append(f"{val:.5f}" if pd.notna(val) else "-")
                    else: row_vals.append(f"{val:.2f}" if pd.notna(val) and isinstance(val, (int, float)) else str(val))
                t1b_1.add_row(*row_vals)
            console.print(t1b_1)
            console.print("\n")

            # Group 2 Profiles
            g2_profiles = ["Default"] + [f"{i}% Loss" for i in range(5, -1, -1)]
            df_g2 = df_raw_global[df_raw_global["Profile"].isin(g2_profiles)]

            t1b_2 = Table(title="Table 1b-2: GLOBAL AVERAGES - Default & Compression Loss", box=box.SIMPLE, header_style="bold yellow")
            for col in header_cols:
                t1b_2.add_column(col, justify="right" if col in ["Ratio", "Time (s)"] else "left")
            for _, row in df_g2.iterrows():
                row_vals = []
                for c in header_cols:
                    val = row.get(c)
                    if c in ["Ratio", "Time (s)"]: row_vals.append(f"{val:.5f}" if pd.notna(val) else "-")
                    else: row_vals.append(f"{val:.2f}" if pd.notna(val) and isinstance(val, (int, float)) else str(val))
                t1b_2.add_row(*row_vals)
            console.print(t1b_2)
            console.print("\n")

        # --- Table 2: COMPARISONS (Per Dataset & Global) ---
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)

            # Helper to generate table 2 variants
            def build_comp_table(title, comp_list, data_df, is_global=False):
                t = Table(title=title, box=box.SIMPLE, header_style="bold green")
                t.add_column("Algorithm" if is_global else "Dataset", justify="left")
                if not is_global: t.add_column("Algorithm", justify="left")

                for cat in comp_list: t.add_column(cat, justify="left")

                for _, row in data_df.iterrows():
                    row_vals = [str(row["Algorithm"])] if is_global else [str(row["Dataset"]), str(row["Algorithm"])]
                    for cat in comp_list:
                        row_vals.append(format_comp_str(row.get(f"{cat}_speed"), row.get(f"{cat}_size")))
                    t.add_row(*row_vals)
                return t

            # 2a. Per Dataset (Split)
            t2a_1 = build_comp_table("Table 2a-1: Relative Comparisons (Core)", comp_categories_1, df_comp, is_global=False)
            t2a_2 = build_comp_table("Table 2a-2: Relative Comparisons (Loss Tradeoffs)", comp_categories_2, df_comp, is_global=False)

            console.print(t2a_1)
            console.print("\n")
            console.print(t2a_2)
            console.print("\n")

            # 2b. Global Averages (Split)
            numeric_comp_cols = [c for c in df_comp.columns if "_speed" in c or "_size" in c]
            df_comp_global = df_comp.groupby("Algorithm", observed=False)[numeric_comp_cols].agg(safe_gmean).reset_index()

            t2b_1 = build_comp_table("Table 2b-1: GLOBAL AVERAGES - Relative Comparisons (Core)", comp_categories_1, df_comp_global, is_global=True)
            t2b_2 = build_comp_table("Table 2b-2: GLOBAL AVERAGES - Relative Comparisons (Loss Tradeoffs)", comp_categories_2, df_comp_global, is_global=True)

            console.print(t2b_1)
            console.print("\n")
            console.print(t2b_2)

        # Save Text Output
        txt_path = out_dir / "tradeoff_summary_tables.txt"
        console.save_text(str(txt_path))
        generated_files.append(txt_path)

        return generated_files

    @staticmethod
    def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        required = {"time", "ratio"}
        if df.empty or not required.issubset(df.columns):
            return pd.DataFrame(columns=["time", "ratio"])

        points = (
            df.loc[:, ["time", "ratio"]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(["time", "ratio"], ascending=[True, True])
        )
        if points.empty:
            return points

        points = points.groupby("time", as_index=False, sort=True)["ratio"].min()

        best_ratio = np.inf
        keep = []
        for ratio in points["ratio"]:
            is_frontier_point = ratio < best_ratio
            keep.append(is_frontier_point)
            if is_frontier_point:
                best_ratio = ratio

        return points.loc[keep].reset_index(drop=True)

    @staticmethod
    def _format_raw_row(ds: str, algo: str, profile: str, pt: pd.Series, param_cols: list) -> dict:
        row = {"Dataset": ds, "Algorithm": algo, "Profile": profile, "Ratio": pt["ratio"], "Time (s)": pt["time"]}
        for p in param_cols:
            val = pt.get(p)
            if pd.notna(val):
                row[p] = val
        return row