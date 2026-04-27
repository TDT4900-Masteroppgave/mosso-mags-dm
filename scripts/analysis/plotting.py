"""All plotting functions for benchmark results: style, comparison, parameter, Pareto, BO, significance, IvB."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.analysis.stats import approx_hv2d


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']
ALGO_CMAP = 'tab10'


def algo_colors(algos: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap(ALGO_CMAP)
    return {a: cmap(i % 10) for i, a in enumerate(algos)}


def _stars(p) -> str:
    if p is None or (isinstance(p, float) and p != p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def algo_style(algo: str, algo_type: str, color_idx: int, cmap) -> dict:
    """Visual style dict for incremental vs batch algorithms."""
    color = cmap(color_idx)
    if algo_type == "mags":
        return dict(color=color, marker="s", linestyle="--",
                    label=f"{algo} (batch)", is_batch=True)
    return dict(color=color, marker="o", linestyle="-",
                label=f"{algo} (incremental)", is_batch=False)


# ---------------------------------------------------------------------------
# Compare plots
# ---------------------------------------------------------------------------

def plot_results(csv_file: str, plot_file: str, logger) -> None:
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]
    ratio_cols = [f"Ratio_{s}" for s in strategies if f"Ratio_{s}" in df.columns]
    time_cols = [f"Time_{s}" for s in strategies if f"Time_{s}" in df.columns]

    if not ratio_cols and not time_cols:
        logger.warning("[!] No valid columns found in dataframe. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(strategies)))

    if ratio_cols:
        df.plot(x="Dataset", y=ratio_cols, kind="bar", ax=axes[0], color=colors[: len(ratio_cols)])
        axes[0].set_title("Compression Ratio (Lower is Better)")
        axes[0].tick_params(axis="x", rotation=45 if len(df) > 1 else 0)
        axes[0].legend([c.replace("Ratio_", "") for c in ratio_cols])

    if time_cols:
        df.plot(x="Dataset", y=time_cols, kind="bar", ax=axes[1], color=colors[: len(time_cols)])
        axes[1].set_title("Execution Time (Seconds)")
        axes[1].tick_params(axis="x", rotation=45 if len(df) > 1 else 0)
        axes[1].legend([c.replace("Time_", "") for c in time_cols])

    plt.tight_layout()
    plt.savefig(plot_file)
    logger.debug(f"Saved bar plot to {plot_file}")
    plt.close()


def plot_runs_variance(
    dataset_name: str, all_times_dict: dict, all_ratios_dict: dict, runs_dir
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(all_times_dict)))

    for idx, (strat, times) in enumerate(all_times_dict.items()):
        if not times:
            continue
        runs_x = list(range(1, len(times) + 1))
        marker = MARKERS[idx % len(MARKERS)]
        axes[0].plot(runs_x, all_ratios_dict[strat], marker=marker, color=colors[idx], label=strat)
        axes[1].plot(runs_x, times, marker=marker, color=colors[idx], label=strat)

    axes[0].set_title("Compression Ratio Variance")
    axes[0].legend()
    axes[1].set_title("Execution Time Variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, f"{dataset_name}.pdf"))
    plt.close()


def plot_scalability(
    df: pd.DataFrame, strategies: list[str], plot_file: str, logger
) -> None:
    """Log-log scatter of time (and memory if available) vs. |E| with fitted slope."""
    if "Edges" not in df.columns:
        logger.debug("[!] No 'Edges' column; skipping scalability plot.")
        return

    plot_df = df[df["Edges"].notna() & (df["Edges"] > 0)].copy()
    if plot_df.empty:
        return

    has_memory = any(f"Memory_avg_{s}" in df.columns for s in strategies)
    n_rows = 2 if has_memory else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    colors = algo_colors(strategies)

    for panel, (metric_prefix, ylabel) in enumerate([
        ("Time_", "Execution Time (s)"),
        ("Memory_avg_", "Peak Memory (MB)"),
    ]):
        if panel >= n_rows:
            break
        ax = axes[panel]
        for idx, strat in enumerate(strategies):
            col = f"{metric_prefix}{strat}"
            if col not in plot_df.columns:
                continue
            sub = plot_df[["Edges", col]].dropna()
            if sub.empty or len(sub) < 2:
                continue
            edges_arr = sub["Edges"].values.astype(float)
            vals_arr = sub[col].values.astype(float)
            ax.scatter(
                edges_arr, vals_arr,
                color=colors[strat], marker=MARKERS[idx % len(MARKERS)],
                s=60, zorder=5, label=strat,
            )
            try:
                log_e = np.log10(edges_arr)
                log_v = np.log10(vals_arr)
                slope, intercept = np.polyfit(log_e, log_v, 1)
                e_fit = np.logspace(log_e.min(), log_e.max(), 50)
                v_fit = 10 ** (slope * np.log10(e_fit) + intercept)
                ax.plot(e_fit, v_fit, color=colors[strat], linestyle="--", linewidth=1.5,
                        label=f"{strat} slope={slope:.2f}")
            except Exception:
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Edges |E| (log scale)", fontsize=12)
        ax.set_ylabel(ylabel + " (log scale)", fontsize=12)
        ax.set_title(f"Scalability: {ylabel} vs. Graph Size", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, which="both")
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    logger.debug(f"Saved scalability plot to {plot_file}")
    plt.close()


# ---------------------------------------------------------------------------
# Parameter plots
# ---------------------------------------------------------------------------

def plot_parameter_analysis(csv_file: str, param_name: str, plot_file: str) -> None:
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]
    avg_df = df.groupby(param_name).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(strategies)))

    for idx, strat in enumerate(strategies):
        marker = MARKERS[idx % len(MARKERS)]
        color = colors[idx]
        if f"Ratio_{strat}" in avg_df.columns and not avg_df[f"Ratio_{strat}"].isnull().all():
            axes[0].plot(avg_df[param_name], avg_df[f"Ratio_{strat}"], marker=marker,
                         linestyle="-", color=color, linewidth=2.5, markersize=8, label=strat)
        if f"Time_{strat}" in avg_df.columns and not avg_df[f"Time_{strat}"].isnull().all():
            axes[1].plot(avg_df[param_name], avg_df[f"Time_{strat}"], marker=marker,
                         linestyle="-", color=color, linewidth=2.5, markersize=8, label=strat)

    for ax, title, ylabel in [
        (axes[0], f"Average Compression Ratio vs {param_name.upper()}", "Compression Ratio (Lower is Better)"),
        (axes[1], f"Average Execution Time vs {param_name.upper()}", "Execution Time in Seconds (Lower is Better)"),
    ]:
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(f"Parameter: {param_name.upper()}", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(avg_df[param_name])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_heatmap(
    df: pd.DataFrame, param_x: str, param_y: str, plot_file: str,
    title: str = "Parameter Heatmap",
) -> None:
    """2D heatmap of avg time and ratio for two-parameter slices from LHS data."""
    if not all(c in df.columns for c in [param_x, param_y, "Time", "Ratio"]):
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, label in [(axes[0], "Time", "Avg Time (s)"), (axes[1], "Ratio", "Avg Ratio")]:
        n_bins = min(8, df[param_x].nunique(), df[param_y].nunique())
        if n_bins < 2:
            ax.set_visible(False)
            continue
        try:
            tmp = df.copy()
            tmp[f"{param_x}_bin"] = pd.qcut(tmp[param_x], n_bins, duplicates="drop")
            tmp[f"{param_y}_bin"] = pd.qcut(tmp[param_y], n_bins, duplicates="drop")
        except Exception:
            ax.set_visible(False)
            continue

        pivot = tmp.pivot_table(
            values=metric, index=f"{param_y}_bin", columns=f"{param_x}_bin", aggfunc="mean"
        )
        if pivot.empty:
            ax.set_visible(False)
            continue

        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([str(r) for r in pivot.index], fontsize=7)
        ax.set_xlabel(param_x.upper(), fontsize=11)
        ax.set_ylabel(param_y.upper(), fontsize=11)
        ax.set_title(f"{title}: {label}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def get_pareto_front_2d(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    sorted_df = df.sort_values(by=[x_col, y_col])
    pareto_indices = []
    min_y = float("inf")
    for index, row in sorted_df.iterrows():
        if row[y_col] < min_y:
            pareto_indices.append(index)
            min_y = row[y_col]
    return df.loc[pareto_indices].copy()


def plot_pareto_front(csv_file: str, plot_file: str, logger) -> None:
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    datasets = df["Dataset"].unique()
    n_datasets = len(datasets)
    cols = 2 if n_datasets > 1 else 1
    rows = (n_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes_flat = [axes] if n_datasets == 1 else axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        ds_df = df[df["Dataset"] == dataset]
        algos = ds_df["Algorithm"].unique()
        colors = algo_colors(list(algos))

        for algo in algos:
            algo_df = ds_df[ds_df["Algorithm"] == algo]
            ax.scatter(
                algo_df["Time"], algo_df["Ratio"],
                color=colors[algo], alpha=0.3, s=30, edgecolors="none", label=f"{algo} (All)",
            )

        for algo in algos:
            algo_df = ds_df[ds_df["Algorithm"] == algo]
            pareto_df = get_pareto_front_2d(algo_df, "Time", "Ratio")
            if not pareto_df.empty:
                pareto_df = pareto_df.sort_values("Time")
                ax.plot(
                    pareto_df["Time"], pareto_df["Ratio"],
                    color=colors[algo], linestyle="-", linewidth=2, alpha=0.9,
                    label=f"{algo} (Front)",
                )
                ax.scatter(
                    pareto_df["Time"], pareto_df["Ratio"],
                    color=colors[algo], edgecolor="black", zorder=5, s=150, marker="*",
                    label=f"{algo} (Optimal)",
                )

        ax.set_title(f"Optimization Landscape: {dataset}", fontsize=13, fontweight="bold")
        x_label = (
            "Normalized Time Score ↓"
            if dataset == "GLOBAL_NORMALIZED_AVERAGE"
            else "Execution Time (Seconds) ↓"
        )
        y_label = (
            "Normalized Ratio Score ↓"
            if dataset == "GLOBAL_NORMALIZED_AVERAGE"
            else "Compression Ratio ↓"
        )
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            loc="upper right", fontsize=9,
        )

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Bayesian optimization convergence
# ---------------------------------------------------------------------------

def plot_bo_convergence(db_path: str, plot_file: str, logger) -> None:
    """2D hypervolume indicator over Optuna trial number per (algo, dataset)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("[!] optuna not installed; skipping BO convergence plot.")
        return

    storage = f"sqlite:///{db_path}"
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
    except Exception as e:
        logger.warning(f"[!] Could not read Optuna DB: {e}")
        return

    if not summaries:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    for i, summary in enumerate(summaries):
        try:
            study = optuna.load_study(study_name=summary.study_name, storage=storage)
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values]
            if len(trials) < 2:
                continue
            hvs = []
            pareto_times: list[float] = []
            pareto_ratios: list[float] = []
            for t in trials:
                pareto_times.append(t.values[0])
                pareto_ratios.append(t.values[1])
                hv = approx_hv2d(list(zip(pareto_times, pareto_ratios)))
                hvs.append(hv)
            ax.plot(range(1, len(hvs) + 1), hvs, color=cmap(i % 10),
                    linewidth=1.5, label=summary.study_name)
        except Exception:
            continue

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Approx. Hypervolume Indicator", fontsize=12)
    ax.set_title("Bayesian Optimization Convergence (Hypervolume over Trials)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Significance plots
# ---------------------------------------------------------------------------

def _grouped_boxplot(
    ax,
    raw_df: pd.DataFrame,
    metric: str,
    baseline_algo: str,
    stars_by_strategy: dict,
    colors: dict,
) -> None:
    datasets = sorted(raw_df["Dataset"].unique())
    algos = [baseline_algo] + [a for a in sorted(raw_df["Algorithm"].unique()) if a != baseline_algo]
    n_algo = len(algos)
    width = 0.8 / max(n_algo, 1)
    positions_per_dataset = np.arange(len(datasets))

    for i, algo in enumerate(algos):
        offsets = positions_per_dataset + (i - (n_algo - 1) / 2.0) * width
        data, valid_pos = [], []
        for ds, pos in zip(datasets, offsets):
            vals = raw_df[(raw_df["Algorithm"] == algo) & (raw_df["Dataset"] == ds)][metric].dropna().tolist()
            if vals:
                data.append(vals)
                valid_pos.append(pos)
        if not data:
            continue
        bp = ax.boxplot(
            data, positions=valid_pos, widths=width * 0.85,
            patch_artist=True, manage_ticks=False, showfliers=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(colors[algo])
            box.set_alpha(0.7 if algo != baseline_algo else 0.95)
            box.set_edgecolor("black")
        for median in bp["medians"]:
            median.set_color("black")

        star = stars_by_strategy.get(algo, "")
        if star:
            ymax = raw_df[raw_df["Algorithm"] == algo][metric].max()
            for pos in valid_pos:
                ax.text(pos, ymax * 1.02, star, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(positions_per_dataset)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[a], edgecolor="black",
                       alpha=0.95 if a == baseline_algo else 0.7)
        for a in algos
    ]
    labels = [f"{a} (baseline)" if a == baseline_algo else a for a in algos]
    ax.legend(handles, labels, loc="best", fontsize=8)


def plot_significance_boxplots(
    raw_csv: str, summary_csv: str, output_pdf: str, baseline_algo: str
) -> None:
    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)
    if raw_df.empty:
        return

    stars_time = {row["Strategy"]: _stars(row.get("p_time (Holm)")) for _, row in summary_df.iterrows()}
    stars_ratio = {row["Strategy"]: _stars(row.get("p_ratio (Holm)")) for _, row in summary_df.iterrows()}
    algos = [baseline_algo] + [a for a in sorted(raw_df["Algorithm"].unique()) if a != baseline_algo]
    colors = algo_colors(algos)

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, 1.2 * len(raw_df["Dataset"].unique()) * len(algos)), 12)
    )
    _grouped_boxplot(axes[0], raw_df, "Time", baseline_algo, stars_time, colors)
    axes[0].set_title(f"Execution Time per Rep (vs. {baseline_algo})", fontsize=13, fontweight="bold")
    _grouped_boxplot(axes[1], raw_df, "Ratio", baseline_algo, stars_ratio, colors)
    axes[1].set_title(f"Compression Ratio per Rep (vs. {baseline_algo})", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Significance stars (* p<0.05, ** p<0.01, *** p<0.001) are Holm-adjusted "
        "cross-dataset Wilcoxon results — they describe the strategy as a whole, "
        "not individual datasets.",
        fontsize=9, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()


def plot_significance_violins(
    raw_csv: str, summary_csv: str, output_pdf: str, baseline_algo: str
) -> None:
    """Violin + strip overlay — reveals multi-modality hidden by boxplots."""
    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)
    if raw_df.empty:
        return

    stars_time = {row["Strategy"]: _stars(row.get("p_time (Holm)")) for _, row in summary_df.iterrows()}
    stars_ratio = {row["Strategy"]: _stars(row.get("p_ratio (Holm)")) for _, row in summary_df.iterrows()}
    algos = [baseline_algo] + [a for a in sorted(raw_df["Algorithm"].unique()) if a != baseline_algo]
    colors = algo_colors(algos)
    datasets = sorted(raw_df["Dataset"].unique())
    n_algo = len(algos)
    width = 0.8 / max(n_algo, 1)
    positions_per_dataset = np.arange(len(datasets))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.2 * len(datasets) * n_algo), 12))

    for ax, metric, stars_by_strategy in [
        (axes[0], "Time", stars_time),
        (axes[1], "Ratio", stars_ratio),
    ]:
        for i, algo in enumerate(algos):
            offsets = positions_per_dataset + (i - (n_algo - 1) / 2.0) * width
            for ds, pos in zip(datasets, offsets):
                vals = raw_df[(raw_df["Algorithm"] == algo) & (raw_df["Dataset"] == ds)][metric].dropna().tolist()
                if len(vals) < 2:
                    continue
                vp = ax.violinplot(
                    [vals], positions=[pos], widths=width * 0.85,
                    showmedians=True, showextrema=False,
                )
                for body in vp["bodies"]:
                    body.set_facecolor(colors[algo])
                    body.set_alpha(0.6 if algo != baseline_algo else 0.85)
                    body.set_edgecolor("black")
                    body.set_linewidth(0.8)
                vp["cmedians"].set_color("black")
                jitter = np.random.default_rng(0).uniform(-width * 0.2, width * 0.2, len(vals))
                ax.scatter([pos + j for j in jitter], vals,
                           color=colors[algo], alpha=0.5, s=10, zorder=3)

            star = stars_by_strategy.get(algo, "")
            if star:
                ymax = raw_df[raw_df["Algorithm"] == algo][metric].max()
                for ds, pos in zip(datasets, offsets):
                    ax.text(pos, ymax * 1.02, star, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(positions_per_dataset)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[a], edgecolor="black",
                           alpha=0.85 if a == baseline_algo else 0.6)
            for a in algos
        ]
        labels = [f"{a} (baseline)" if a == baseline_algo else a for a in algos]
        ax.legend(handles, labels, loc="best", fontsize=8)

    axes[0].set_title(f"Execution Time Distribution (vs. {baseline_algo})", fontsize=13, fontweight="bold")
    axes[1].set_title(f"Compression Ratio Distribution (vs. {baseline_algo})", fontsize=13, fontweight="bold")
    fig.suptitle(
        "Violin plots show distribution shape; stars are Holm-adjusted Wilcoxon significance.",
        fontsize=9, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Incremental vs batch update cost
# ---------------------------------------------------------------------------

def plot_update_cost(
    df: pd.DataFrame, algos: list[str], algo_types: dict[str, str], plot_file: str
) -> None:
    """Per-checkpoint update cost: incremental = delta time, batch = full re-run time."""
    datasets = df["Dataset"].unique()
    checkpoints = sorted(df["Checkpoint"].unique())
    if len(checkpoints) < 2:
        return

    n = len(datasets)
    cols = min(n, 3)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // cols][idx % cols]
        ds = df[df["Dataset"] == dataset].sort_values("Checkpoint")

        for i, algo in enumerate(algos):
            t_col = f"Time_{algo}"
            if t_col not in ds.columns:
                continue
            is_batch = algo_types.get(algo) == "mags"
            costs, cps = [], []
            prev_time = 0.0
            for _, r in ds.iterrows():
                t = r.get(t_col)
                if pd.isna(t):
                    continue
                cost = t if is_batch else (t - prev_time)
                if not is_batch:
                    prev_time = t
                costs.append(cost)
                cps.append(r["Checkpoint"])

            if costs:
                label = f"{algo} ({'batch' if is_batch else 'incremental'})"
                ax.plot(cps, costs, marker="o", linestyle="--" if is_batch else "-",
                        color=cmap(i), linewidth=2, markersize=6, label=label)

                if not is_batch and len(costs) >= 2:
                    for cp, c in zip(cps, costs):
                        batch_ref = ds[ds["Checkpoint"] == cp]
                        for ba in algos:
                            if algo_types.get(ba) == "mags" and f"Time_{ba}" in batch_ref.columns:
                                b_t = batch_ref[f"Time_{ba}"].values
                                if len(b_t) > 0 and pd.notna(b_t[0]) and c < b_t[0]:
                                    ax.axvline(cp, color=cmap(i), linestyle=":", alpha=0.5)
                                    break

        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xlabel("Checkpoint (fraction of edges)", fontsize=11)
        ax.set_ylabel("Update Cost (s)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

    for idx in range(n, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.suptitle("Update Cost: incremental (delta) vs. batch (full re-run)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    plt.close()
