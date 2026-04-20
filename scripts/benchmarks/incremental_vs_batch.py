"""
Fair comparison of incremental (streaming) vs batch graph summarization algorithms.

Methodology based on Ko et al., "Incremental Lossless Graph Summarization" (KDD 2020):

  Figure 5 (Compression quality over time):
  - Splits edge streams into checkpoints (e.g., 20%, 40%, 60%, 80%, 100%)
  - At each checkpoint, runs ALL algorithms on the partial graph (or parses logs for streaming algos)
  - Compares compression ratio evolution

  Figure 4 (Speed - update cost):
  - Incremental update cost: time to process ONLY the new edges since last checkpoint
    (derived as time(cp) - time(prev_cp), since MoSSo streams edges sequentially)
  - Batch re-run cost: time to run from scratch on the full snapshot at each checkpoint
  - This shows MoSSo's real advantage: near-constant update cost vs growing re-run cost
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from scripts.benchmark import Benchmark
from scripts.config import ALGORITHMS, PARAM_CONFIG


class IncrementalVsBatchBenchmark(Benchmark):

    DEFAULT_CHECKPOINTS = [0.2, 0.4, 0.6, 0.8, 1.0]
    _EDGE_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)")
    _MOSSO_LOG_PATTERN = re.compile(r"(\d+)\s*:\s*Elapsed time\s*:\s*([\d.]+)\s*:\s*ratio\s*:\s*([\d.]+)")

    def __init__(self):
        super().__init__("incremental_vs_batch")

    def add_custom_args(self, parser):
        parser.add_argument(
            "--checkpoints", nargs="+", type=float,
            default=self.DEFAULT_CHECKPOINTS,
            help="Edge stream fractions to evaluate at (default: 0.2 0.4 0.6 0.8 1.0)",
        )

    def _create_partial_dataset(self, dataset_path: str, fraction: float, total_edges: int):
        """Write a file containing the first *fraction* of edges."""
        target_edges = int(total_edges * fraction)

        base_dir = os.path.dirname(dataset_path)
        basename = os.path.basename(dataset_path)
        partial_dir = os.path.join(base_dir, "partial")
        os.makedirs(partial_dir, exist_ok=True)

        partial_path = os.path.join(partial_dir, f"p{int(fraction * 100)}_{basename}")

        # Cache: reuse if already created in an earlier run
        if os.path.exists(partial_path):
            return partial_path

        edges_written = 0
        seen_edges = set()
        with open(dataset_path, "r", encoding="utf-8") as f_in, \
                open(partial_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.startswith(('#', '%')): continue

                match = self._EDGE_PATTERN.search(line)
                if match:
                    u, v = int(match.group(1)), int(match.group(2))
                    if u == v: continue  # Remove self-loops

                    # ignore the direction of edge
                    edge = tuple(sorted((u, v)))
                    # remove duplicate edges
                    if edge in seen_edges: continue
                    seen_edges.add(edge)

                    f_out.write(line)
                    edges_written += 1

        self.logger.debug(
            f"\t[*] Partial dataset ({fraction:.0%}): {edges_written:,} edges -> {partial_path}"
        )
        return partial_path

    def _parse_mosso_stdout(self, stdout: str):
        """Parses the interval logs from MoSSo stdout to extract time and ratio."""
        parsed = {}
        for line in stdout.splitlines():
            match = self._MOSSO_LOG_PATTERN.search(line)
            if match:
                edges = int(match.group(1))
                cum_time = float(match.group(2))
                ratio = float(match.group(3))
                parsed[edges] = {"cum_time": cum_time, "ratio": ratio}
        return parsed

    def process(self, dataset_path: str, ds: dict, dataset_name: str):
        meta = ds.get("meta", dataset_name)
        total_edges = meta["edges"]
        self.logger.info(f"\t[*] Total edges in stream: {total_edges:,}")

        checkpoints = sorted(self.args.checkpoints)

        # Initialize dictionary to hold row data for each checkpoint
        checkpoint_rows = {
            cp: {
                "Dataset": dataset_name,
                "Checkpoint": cp,
                "Edges": int(total_edges * cp),
            }
            for cp in checkpoints
        }

        for algo_name, algo_config in self.active_algos.items():
            resolved_params = {}
            params = algo_config.get("params", {})
            for p_key in PARAM_CONFIG:
                resolved_params[p_key] = params.get(p_key, getattr(self.args, p_key))

            is_batch = self._algo_type(algo_name) == "mags"

            if is_batch:
                # MAGS (Batch Algorithm): Needs to be run from scratch on partial datasets
                self.logger.info(f"\t[*] Running Batch Algorithm: {algo_name}")
                for cp in checkpoints:
                    n_edges = int(total_edges * cp)

                    if cp >= 1.0:
                        partial_path = dataset_path
                    else:
                        partial_path = self._create_partial_dataset(dataset_path, cp, total_edges)

                    t_avg, r_avg, _, _ = self.execute_runner(
                        algo_name=algo_name,
                        algo_config=algo_config,
                        dataset_path=partial_path,
                        dataset_name=f"{dataset_name}_p{int(cp * 100)}",
                        resolved_params=resolved_params,
                    )

                    checkpoint_rows[cp][f"Time_{algo_name}"] = t_avg
                    checkpoint_rows[cp][f"Ratio_{algo_name}"] = r_avg

                    if t_avg is not None:
                        self.logger.info(
                            f"\t=> {algo_name:<12} [CP {cp:.0%}] Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}"
                        )
            else:
                # MoSSo (Incremental Algorithm): Run once on 100% dataset, read log files
                self.logger.info(f"\t[*] Running Incremental Algorithm: {algo_name}")

                # execute_runner returns times/ratios, not the raw stdout strings
                t_avg, r_avg, _, _ = self.execute_runner(
                    algo_name=algo_name,
                    algo_config=algo_config,
                    dataset_path=dataset_path,
                    dataset_name=f"{dataset_name}_p100",
                    resolved_params=resolved_params,
                )

                if t_avg is None:
                    continue  # Execution failed

                # Find log files on disk
                base_name = f"{algo_name}_{dataset_name}_p100_{self.timestamp}"
                log_files = list(self.session_dir.rglob(f"{base_name}_run*.log"))

                if not log_files:
                    self.logger.warning(f"\t[!] Could not find log files for {base_name} to parse checkpoints.")
                    continue

                # Read all runs to average them out
                stdouts = []
                for log_file in log_files:
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            stdouts.append(f.read())
                    except Exception as e:
                        self.logger.error(f"\t[!] Failed to read log file {log_file}: {e}")

                # Aggregate parsed data across all runs to calculate averages
                aggregated_data = {}
                for out in stdouts:
                    run_data = self._parse_mosso_stdout(out)
                    for edges, metrics in run_data.items():
                        if edges not in aggregated_data:
                            aggregated_data[edges] = {"cum_time": [], "ratio": []}
                        aggregated_data[edges]["cum_time"].append(metrics["cum_time"])
                        aggregated_data[edges]["ratio"].append(metrics["ratio"])

                final_parsed = {}
                for edges, metrics in aggregated_data.items():
                    final_parsed[edges] = {
                        "cum_time": sum(metrics["cum_time"]) / len(metrics["cum_time"]),
                        "ratio": sum(metrics["ratio"]) / len(metrics["ratio"]),
                    }

                # Ensure the 100% mark contains the exact final time/ratio
                if total_edges not in final_parsed:
                    final_parsed[total_edges] = {"cum_time": t_avg, "ratio": r_avg}

                # Map extracted checkpoint data to rows
                for cp in checkpoints:
                    n_edges = int(total_edges * cp)

                    if final_parsed:
                        # Find the interval marker closest to this checkpoint's edge count
                        closest_edge = min(final_parsed.keys(), key=lambda k: abs(k - n_edges))

                        c_time = final_parsed[closest_edge]["cum_time"]
                        c_ratio = final_parsed[closest_edge]["ratio"]

                        checkpoint_rows[cp][f"Time_{algo_name}"] = c_time
                        checkpoint_rows[cp][f"Ratio_{algo_name}"] = c_ratio
                        self.logger.info(
                            f"\t=> {algo_name:<12} [CP {cp:.0%} mapped -> {closest_edge} edges] Time: {c_time:.3f}s | Ratio: {c_ratio:.5f}"
                        )
                    else:
                        checkpoint_rows[cp][f"Time_{algo_name}"] = None
                        checkpoint_rows[cp][f"Ratio_{algo_name}"] = None

        # Append fully populated rows to results sequentially
        for cp in checkpoints:
            self.results.append(checkpoint_rows[cp])

    def _detect_algos(self, df):
        return [c.replace("Time_", "") for c in df.columns if c.startswith("Time_")]

    def _format_display(self, df, algos):
        """Return a human-readable copy of the results dataframe."""
        disp = df.copy()
        disp["Checkpoint"] = disp["Checkpoint"].apply(lambda x: f"{x:.0%}")
        disp["Edges"] = disp["Edges"].apply(lambda x: f"{x:,}")
        for algo in algos:
            for prefix, fmt in [("Time_", "{:.3f}s"), ("Ratio_", "{:.5f}")]:
                col = f"{prefix}{algo}"
                if col in disp.columns:
                    disp[col] = disp[col].apply(
                        lambda x, f=fmt: f.format(x) if pd.notna(x) else "N/A"
                    )
        return disp

    def print_table(self):
        df = pd.DataFrame(self.results)
        algos = self._detect_algos(df)
        disp = self._format_display(df, algos)

        table_str = tabulate(disp, headers="keys", tablefmt="grid", showindex=False)
        for line in table_str.split("\n"):
            self.logger.info(line)

        self._print_summary(df, algos)

    def _algo_type(self, algo):
        return ALGORITHMS.get(algo, {}).get("type", "mosso")

    def _print_summary(self, df, algos):
        """Log a brief comparison summary at the 100% checkpoint."""
        full = df[df["Checkpoint"] >= 0.99]
        if full.empty:
            return

        self.logger.info("\n" + "=" * 10 + " SUMMARY AT 100% " + "=" * 10)

        for dataset in full["Dataset"].unique():
            row = full[full["Dataset"] == dataset].iloc[0]
            n_edges = row.get("Edges", 0)
            self.logger.info(f"\n  [{dataset}] ({int(n_edges):,} edges)")

            for algo in algos:
                t = row.get(f"Time_{algo}")
                r = row.get(f"Ratio_{algo}")
                kind = "batch" if self._algo_type(algo) == "mags" else "incremental"
                if pd.notna(t) and pd.notna(r):
                    per_edge_us = (t / n_edges * 1e6) if n_edges > 0 else 0
                    self.logger.info(
                        f"    {algo:<12} ({kind:<11}) "
                        f"Time: {t:.3f}s | "
                        f"Per-edge: {per_edge_us:.1f}\u00b5s | "
                        f"Ratio: {r:.5f}"
                    )

            # Pairwise speed / quality comparisons
            for i, a1 in enumerate(algos):
                for a2 in algos[i + 1:]:
                    t1, t2 = row.get(f"Time_{a1}"), row.get(f"Time_{a2}")
                    r1, r2 = row.get(f"Ratio_{a1}"), row.get(f"Ratio_{a2}")
                    if not all(pd.notna(v) for v in [t1, t2, r1, r2]):
                        continue

                    if t1 > 0 and t2 > 0:
                        faster = a1 if t1 < t2 else a2
                        speedup = max(t1, t2) / min(t1, t2)
                        self.logger.info(f"    -> {faster} is {speedup:.1f}x faster (total time)")

                    better = a1 if r1 < r2 else a2
                    diff = abs(r1 - r2)
                    rel = diff / max(r1, r2) * 100
                    self.logger.info(
                        f"    -> {better} has {diff:.5f} better compression ({rel:.1f}% relative)"
                    )

        # Update cost analysis
        self._print_update_cost(df, algos)

    def _print_update_cost(self, df, algos):
        """
        Compare the cost of keeping summaries up to date at every checkpoint.

        Incremental algorithms process edges as a stream, so when the graph
        changes they only need to process the NEW edges.  Batch algorithms
        must re-run from scratch on the entire current snapshot.

        - Incremental update cost at checkpoint k:
              time(k) - time(k-1)   (only the new edges since last checkpoint)
        - Batch update cost at checkpoint k:
              time(k)               (full re-run on snapshot)

        Total cost to stay current at ALL checkpoints:
        - Incremental: time(last_checkpoint)  (stream covers all prior states)
        - Batch: sum of time(k) for all k     (must re-run each time)
        """
        datasets = df["Dataset"].unique()
        checkpoints = sorted(df["Checkpoint"].unique())
        if len(checkpoints) < 2:
            return

        self.logger.info("\n" + "=" * 10 + " UPDATE COST ANALYSIS " + "=" * 10)
        self.logger.info(
            "  Incremental: cost = new edges only (time[k] - time[k-1])"
        )
        self.logger.info(
            "  Batch: cost = full re-run from scratch (time[k])\n"
        )

        for dataset in datasets:
            ds = df[df["Dataset"] == dataset].sort_values("Checkpoint")
            self.logger.info(f"  [{dataset}]")

            # Per-checkpoint update cost table
            update_rows = []
            for algo in algos:
                t_col = f"Time_{algo}"
                if t_col not in ds.columns:
                    continue
                is_batch = self._algo_type(algo) == "mags"
                prev_time = 0.0
                for _, r in ds.iterrows():
                    t = r.get(t_col)
                    cp = r["Checkpoint"]
                    edges = r["Edges"]
                    if pd.isna(t):
                        continue
                    if is_batch:
                        update_cost = t
                    else:
                        update_cost = t - prev_time
                        prev_time = t
                    update_rows.append({
                        "Algorithm": algo,
                        "Type": "batch" if is_batch else "incremental",
                        "Checkpoint": f"{cp:.0%}",
                        "Edges": f"{int(edges):,}",
                        "Update Cost": f"{update_cost:.3f}s",
                    })

            if update_rows:
                tbl = tabulate(
                    pd.DataFrame(update_rows),
                    headers="keys", tablefmt="simple", showindex=False,
                )
                for line in tbl.split("\n"):
                    self.logger.info(f"    {line}")

            # Total cost summary
            self.logger.info("")
            self.logger.info(f"    Total cost to maintain summary at ALL {len(checkpoints)} checkpoints:")
            for algo in algos:
                t_col = f"Time_{algo}"
                if t_col not in ds.columns:
                    continue
                is_batch = self._algo_type(algo) == "mags"
                valid = ds[ds[t_col].notna()]
                if valid.empty:
                    continue

                if is_batch:
                    cost = valid[t_col].sum()
                    label = "sum of re-runs"
                else:
                    cost = valid[t_col].max()
                    label = "single stream-through"

                self.logger.info(f"      {algo:<12} {cost:.3f}s ({label})")
            self.logger.info("")

    def finalize(self):
        csv_file = os.path.join(self.session_dir, "results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)

        algos = self._detect_algos(df)

        # Save pretty table
        disp = self._format_display(df, algos)
        table_str = tabulate(disp, headers="keys", tablefmt="grid", showindex=False)
        with open(os.path.join(self.session_dir, "table_results.txt"), "w") as f:
            f.write(table_str)

        # Plots
        self._plot_compression_evolution(df, algos)
        self._plot_speed_bar_chart(df, algos)

    def _algo_style(self, algo, color_idx, cmap):
        """Return visual properties depending on algorithm type."""
        color = cmap(color_idx)
        if self._algo_type(algo) == "mags":
            return dict(
                color=color, marker="s", linestyle="--", label=f"{algo} (batch)",
                is_batch=True,
            )
        return dict(
            color=color, marker="o", linestyle="-", label=f"{algo} (incremental)",
            is_batch=False,
        )

    def _plot_per_dataset(self, df, algos, y_prefix, ylabel, title, filename,
                          log_scale=False):
        datasets = df["Dataset"].unique()
        n = len(datasets)
        cols = min(n, 3)
        rows = max(1, (n + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
        cmap = plt.get_cmap("tab10")

        for idx, dataset in enumerate(datasets):
            ax = axes[idx // cols][idx % cols]
            ds = df[df["Dataset"] == dataset].sort_values("Checkpoint")

            for i, algo in enumerate(algos):
                col = f"{y_prefix}{algo}"
                if col not in ds.columns or ds[col].isna().all():
                    continue

                style = self._algo_style(algo, i, cmap)
                valid = ds[ds[col].notna()]

                if style.pop("is_batch"):
                    ax.scatter(
                        valid["Checkpoint"], valid[col],
                        color=style["color"], s=100, marker=style["marker"],
                        zorder=5, edgecolors="black", linewidths=0.5,
                        label=style["label"],
                    )
                else:
                    ax.plot(
                        valid["Checkpoint"], valid[col],
                        color=style["color"], linewidth=2,
                        marker=style["marker"], markersize=6,
                        linestyle=style["linestyle"], label=style["label"],
                    )

            ax.set_title(dataset, fontsize=13, fontweight="bold")
            ax.set_xlabel("Ratio of Processed Changes", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(0, 1.05)
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=9, loc="best")

        for idx in range(n, rows * cols):
            fig.delaxes(axes[idx // cols][idx % cols])

        plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(self.session_dir, filename)
        plt.savefig(path, format="pdf", bbox_inches="tight")
        plt.close()

    def _plot_compression_evolution(self, df, algos):
        """Compression ratio at each checkpoint (Figure 5 style from MoSSo paper)."""
        self._plot_per_dataset(
            df, algos,
            y_prefix="Ratio_",
            ylabel="Compression Ratio",
            title="Compression Ratio Evolution (Lower is Better)",
            filename="compression_ratio_evolution.pdf",
        )

    def _plot_speed_bar_chart(self, df, algos):
        """
        Speed comparison bar chart (MoSSo paper, Figure 4).

        The MoSSo paper compares algorithms by answering:
        "If the graph receives one new edge, how long to get an updated summary?"

        - Incremental (MoSSo): per-change time = total_time / n_edges
        - Batch (MAGS): total time (must re-run from scratch)

        Both are plotted in microseconds on a shared log-scale Y-axis.
        Speedup annotations show how many times faster the best incremental
        algorithm is compared to the best batch algorithm per dataset.
        """
        full = df[df["Checkpoint"] >= 0.99]
        if full.empty:
            return

        datasets = full["Dataset"].unique()
        if len(datasets) == 0:
            return

        # Hatches to distinguish algorithm types visually (like the paper)
        hatches = ["", "//", "\\\\", "xx", "..", "++", "oo"]
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2.5), 6))

        x = np.arange(len(datasets))
        n_algos = len(algos)
        width = 0.8 / n_algos

        algo_values = {}  # algo -> list of values per dataset (in microseconds)

        for i, algo in enumerate(algos):
            is_batch = self._algo_type(algo) == "mags"
            values = []
            for dataset in datasets:
                ds_row = full[full["Dataset"] == dataset]
                t = ds_row[f"Time_{algo}"].values[0] if f"Time_{algo}" in ds_row.columns and len(ds_row) > 0 else np.nan
                n_edges = ds_row["Edges"].values[0] if len(ds_row) > 0 else 0

                if pd.isna(t):
                    values.append(0)
                elif is_batch:
                    values.append(t * 1e6)  # total seconds -> microseconds
                else:
                    values.append((t / n_edges * 1e6) if n_edges > 0 else 0)  # per-change us

            algo_values[algo] = (values, is_batch)

            offset = (i - n_algos / 2 + 0.5) * width
            label = f"{algo} (total time)" if is_batch else f"{algo} (per-change)"
            hatch = hatches[i % len(hatches)]

            ax.bar(
                x + offset, values, width,
                label=label, color=cmap(i), hatch=hatch,
                edgecolor="black", linewidth=0.5,
                )

        # Add speedup annotations above each dataset group
        for j, dataset in enumerate(datasets):
            inc_vals = []
            batch_vals = []
            for algo, (vals, is_batch) in algo_values.items():
                v = vals[j]
                if v > 0:
                    if is_batch:
                        batch_vals.append(v)
                    else:
                        inc_vals.append(v)

            if inc_vals and batch_vals:
                fastest_inc = min(inc_vals)
                fastest_batch = min(batch_vals)
                speedup = fastest_batch / fastest_inc

                # Position annotation above the tallest bar in this group
                all_vals = [v for v in inc_vals + batch_vals if v > 0]
                max_val = max(all_vals)
                ax.text(
                    j, max_val * 2.0, f"{speedup:,.0f}x",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="black",
                       )

        ax.set_yscale("log")
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Execution Time (microseconds, log scale)", fontsize=12)
        ax.set_title(
            "Speed: Per-Change Time (incremental) vs Total Time (batch)",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        path = os.path.join(self.session_dir, "speed_comparison.pdf")
        plt.savefig(path, format="pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    IncrementalVsBatchBenchmark().run()