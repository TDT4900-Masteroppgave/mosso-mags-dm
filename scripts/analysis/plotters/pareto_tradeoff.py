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

        comp_categories = [
            "Best vs Best", "Fastest vs Fastest", "Balanced vs Balanced",
            "MoSSo Default vs Balanced", "MoSSo Default vs Matches Comp", "MoSSo Default vs Matches Time"
        ]

        # ==========================================================
        # EXTRACTION & PLOTTING LOOP
        # ==========================================================
        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy().reset_index(drop=True)
            if ds_data.empty: continue

            # --- 1. Extract MoSSo Baseline Points ---
            mosso_data = ds_data[ds_data["algorithm"] == "kdd20-mosso"].copy()
            if mosso_data.empty:
                print(f"[Warning] MoSSo not found in dataset '{ds}'. Skipping.")
                continue

            # Identify MoSSo Default (c=120, e=3)
            default_mask = mosso_data["c"] == 120 if "c" in mosso_data.columns else pd.Series(True, index=mosso_data.index)
            if "e" in mosso_data.columns:
                default_mask = default_mask & (mosso_data["e"] == 3)

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
                "Balanced 50/50": mosso_data.loc[mosso_data["dist_to_ideal"].idxmin()]
            }

            for profile, pt in mosso_points.items():
                raw_tradeoff_data.append(self._format_raw_row(ds, "kdd20-mosso", profile, pt, param_cols))

            # --- Plot Setup ---
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.axhline(def_r, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)
            ax.axvline(def_t, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)

            ax.plot(mosso_data["time"], mosso_data["ratio"], marker=".", color="gray", linestyle="none", alpha=0.15, zorder=2)
            ax.plot(def_t, def_r, marker="*", markersize=14, color="gold", markeredgecolor="black", zorder=10, label="MoSSo Default")

            # --- 2. Extract Other Algorithms ---
            for algo in algos:
                if algo == "kdd20-mosso": continue

                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty: continue

                color = self.get_algo_style(algo)["color"]
                ax.plot(algo_data["time"], algo_data["ratio"], marker=".", color=color, linestyle="none", alpha=0.15, zorder=2)

                algo_data["dist_to_ideal"] = np.sqrt((algo_data["time"] / def_t)**2 + (algo_data["ratio"] / def_r)**2)

                algo_points = {
                    "Best Compression": algo_data.loc[algo_data["ratio"].idxmin()],
                    "Fastest Time": algo_data.loc[algo_data["time"].idxmin()],
                    "Balanced 50/50": algo_data.loc[algo_data["dist_to_ideal"].idxmin()]
                }

                same_comp = algo_data[algo_data["ratio"] <= def_r]
                if not same_comp.empty:
                    algo_points["Matches MoSSo Compression"] = same_comp.loc[same_comp["time"].idxmin()]

                same_time = algo_data[algo_data["time"] <= def_t]
                if not same_time.empty:
                    algo_points["Matches MoSSo Time"] = same_time.loc[same_time["ratio"].idxmin()]

                # Save raw points and plot
                markers = {"Best Compression": "v", "Fastest Time": "<", "Balanced 50/50": "D",
                           "Matches MoSSo Compression": "s", "Matches MoSSo Time": "o"}

                for profile, pt in algo_points.items():
                    raw_tradeoff_data.append(self._format_raw_row(ds, algo, profile, pt, param_cols))
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

                add_comp_metrics("Best vs Best", mosso_points["Best Compression"], algo_points.get("Best Compression"))
                add_comp_metrics("Fastest vs Fastest", mosso_points["Fastest Time"], algo_points.get("Fastest Time"))
                add_comp_metrics("Balanced vs Balanced", mosso_points["Balanced 50/50"], algo_points.get("Balanced 50/50"))
                add_comp_metrics("MoSSo Default vs Balanced", mosso_points["Default"], algo_points.get("Balanced 50/50"))
                add_comp_metrics("MoSSo Default vs Matches Comp", mosso_points["Default"], algo_points.get("Matches MoSSo Compression"))
                add_comp_metrics("MoSSo Default vs Matches Time", mosso_points["Default"], algo_points.get("Matches MoSSo Time"))

                comparison_data.append(comp_row)

            # --- Finalize Plot ---
            ax.set_xlabel(options.get("time_label", "Time (seconds)"), fontsize=12, style='italic')
            ax.set_ylabel("Relative Size (Compression Ratio)", fontsize=12, style='italic')
            ax.set_title(f"Pareto Tradeoff Extractions - {ds}", pad=15, weight="bold")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            sns.despine(ax=ax)

            legend_elements = [Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=12, label='MoSSo Default')]
            for prof, mk in [("Best Comp.", "v"), ("Fastest", "<"), ("Balanced", "D"), ("Matches Comp.", "s"), ("Matches Time", "o")]:
                legend_elements.append(Line2D([0], [0], marker=mk, color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=prof))

            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)

            fig.tight_layout()
            out_path = out_dir / f"tradeoffs_{ds}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated_files.append(out_path)

        # ==========================================================
        # RICH TABLES EXPORT (NO CSV)
        # ==========================================================
        console = Console(record=True)

        # Helper to format comparison strings
        def format_comp_str(speed, size):
            if pd.isna(speed) or pd.isna(size):
                return "-"
            return f"{speed:.2f}x speed, {size:.3f}x size"

        def numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
            """Return only columns that can safely be averaged.

            Bayesian result tables may contain string-valued metadata columns
            such as `param`/`param_name`. Those are useful in the per-dataset
            table, but pandas cannot compute a global mean over them.
            """
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
            """Geometric mean over positive finite values, or NaN if absent."""
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            values = values[values > 0]
            return float(gmean(values)) if not values.empty else np.nan

        # --- Table 1: RAW VALUES (Per Dataset & Global) ---
        if raw_tradeoff_data:
            df_raw = pd.DataFrame(raw_tradeoff_data)
            base_cols = ["Algorithm", "Profile", "Ratio", "Time (s)"]
            all_cols = ["Dataset"] + base_cols + [c for c in df_raw.columns if c not in ["Dataset"] + base_cols]

            # 1a. Per Dataset
            t1a = Table(title="Table 1a: Raw Operational Points (Per Dataset)", box=box.SIMPLE, header_style="bold yellow")
            for col in all_cols:
                t1a.add_column(col, justify="right" if col in ["Ratio", "Time (s)"] else "left")
            for _, row in df_raw.iterrows():
                row_vals = [f"{row.get(c):.5f}" if c in ["Ratio", "Time (s)"] else str(row.get(c, "-")) for c in all_cols]
                t1a.add_row(*row_vals)

            console.print(t1a)

            # 1b. Global Averages
            candidate_avg_cols = ["Ratio", "Time (s)"] + [c for c in param_cols if c in df_raw.columns]
            cols_to_average = numeric_columns(df_raw, candidate_avg_cols)

            # Group by Algorithm and Profile, selecting only columns that are numeric.
            # Non-numeric metadata remains in Table 1a, but is skipped here.
            df_raw_global = df_raw.groupby(["Algorithm", "Profile"], observed=False)[cols_to_average].mean().reset_index()

            t1b = Table(title="Table 1b: GLOBAL AVERAGES - Raw Operational Points", box=box.SIMPLE, header_style="bold yellow")

            # 3. Define the correct header order
            header_cols = ["Algorithm", "Profile"] + cols_to_average

            for col in header_cols:
                t1b.add_column(col, justify="right" if col in ["Ratio", "Time (s)"] else "left")

            for _, row in df_raw_global.iterrows():
                row_vals = []
                for c in header_cols:
                    val = row.get(c)
                    if c in ["Ratio", "Time (s)"]:
                        row_vals.append(f"{val:.5f}" if pd.notna(val) else "-")
                    else:
                        # For hyperparams, display as float or int
                        row_vals.append(f"{val:.2f}" if pd.notna(val) and isinstance(val, (int, float)) else str(val))
                t1b.add_row(*row_vals)

            console.print(t1b)

        # --- Table 2: COMPARISONS (Per Dataset & Global) ---
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            comp_cols_display = ["Dataset", "Algorithm"] + comp_categories

            # 2a. Per Dataset
            t2a = Table(title="Table 2a: Relative Comparisons (Per Dataset)", box=box.SIMPLE, header_style="bold green")
            for col in comp_cols_display:
                t2a.add_column(col, justify="left")

            for _, row in df_comp.iterrows():
                row_vals = [row["Dataset"], row["Algorithm"]]
                for cat in comp_categories:
                    row_vals.append(format_comp_str(row.get(f"{cat}_speed"), row.get(f"{cat}_size")))
                t2a.add_row(*row_vals)

            # 2b. Global Averages
            numeric_comp_cols = [c for c in df_comp.columns if "_speed" in c or "_size" in c]
            df_comp_global = df_comp.groupby("Algorithm", observed=False)[numeric_comp_cols].agg(safe_gmean).reset_index()

            t2b = Table(title="Table 2b: GLOBAL AVERAGES - Relative Comparisons", box=box.SIMPLE, header_style="bold green")
            t2b.add_column("Algorithm", justify="left")
            for cat in comp_categories:
                t2b.add_column(cat, justify="left")

            for _, row in df_comp_global.iterrows():
                # Build list of row values based on column order
                row_vals = [str(row["Algorithm"])]
                for cat in comp_categories:
                    row_vals.append(format_comp_str(row.get(f"{cat}_speed"), row.get(f"{cat}_size")))
                t2b.add_row(*row_vals)

            console.print(t2a)
            console.print("\n")
            console.print(t2b)

        # Save Text Output
        txt_path = out_dir / "tradeoff_summary_tables.txt"
        console.save_text(str(txt_path))
        generated_files.append(txt_path)

        return generated_files

    @staticmethod
    def _format_raw_row(ds: str, algo: str, profile: str, pt: pd.Series, param_cols: list) -> dict:
        row = {"Dataset": ds, "Algorithm": algo, "Profile": profile, "Ratio": pt["ratio"], "Time (s)": pt["time"]}
        for p in param_cols:
            val = pt.get(p)
            if pd.notna(val):
                row[p] = val
        return row