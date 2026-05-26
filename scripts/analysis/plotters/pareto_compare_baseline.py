import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import optuna
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.analysis.plotters.base_plotter import Plotter, register

# Enable recording to save the console output to a text file
console = Console(record=True, highlight=False)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={_format_value(value)}" for key, value in params.items())


def _format_endpoints(endpoints: list[dict[str, Any]] | None) -> str:
    if not endpoints:
        return "-"
    formatted = []
    for endpoint in endpoints:
        params = endpoint.get("params", {})
        formatted.append(
            f"trial={endpoint.get('trial', '-')} "
            f"({float(endpoint['avg_normalized_time']):.4g}, {float(endpoint['avg_compression_ratio']):.4g}) "
            f"{_format_params(params)}"
        )
    return " | ".join(formatted)


def _normalise(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return (value - lower) / (upper - lower)


def _knee_trial(front: list[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    min_time = min(float(t.values[0]) for t in front)
    max_time = max(float(t.values[0]) for t in front)
    min_ratio = min(float(t.values[1]) for t in front)
    max_ratio = max(float(t.values[1]) for t in front)

    def score(trial: optuna.trial.FrozenTrial) -> float:
        time_norm = _normalise(float(trial.values[0]), min_time, max_time)
        ratio_norm = _normalise(float(trial.values[1]), min_ratio, max_ratio)
        return math.sqrt(time_norm * time_norm + ratio_norm * ratio_norm)

    return min(front, key=score)

def _knee_row(front_df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    min_x = front_df[x_col].min()
    max_x = front_df[x_col].max()
    min_y = front_df[y_col].min()
    max_y = front_df[y_col].max()

    def score(row):
        x_norm = _normalise(row[x_col], min_x, max_x)
        y_norm = _normalise(row[y_col], min_y, max_y)
        return math.sqrt(x_norm * x_norm + y_norm * y_norm)

    idx = front_df.apply(score, axis=1).idxmin()
    return front_df.loc[idx]


def _trial_record(
        algorithm: str,
        trial: optuna.trial.FrozenTrial,
        point_type: str,
) -> dict[str, Any]:
    params = dict(trial.params)
    return {
        "algorithm": algorithm,
        "trial": trial.number,
        "point_type": point_type,
        "avg_normalized_time": float(trial.values[0]),
        "avg_compression_ratio": float(trial.values[1]),
        "params": json.dumps(params),
        "params_display": _format_params(params),
        "status": "observed",
        "source": "optuna_study.db",
        "interpolation_endpoints": "[]",
    }


def _point_record(
        algorithm: str,
        point_type: str,
        avg_normalized_time: float,
        avg_compression_ratio: float,
        params: dict[str, Any] | None = None,
        trial: int | str = "",
        status: str = "observed",
        source: str = "",
        interpolation_endpoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = params or {}
    params_display = _format_params(params)
    if not params and interpolation_endpoints:
        params_display = _format_endpoints(interpolation_endpoints)
    return {
        "algorithm": algorithm,
        "trial": trial,
        "point_type": point_type,
        "avg_normalized_time": avg_normalized_time,
        "avg_compression_ratio": avg_compression_ratio,
        "params": json.dumps(params) if params else "",
        "params_display": params_display,
        "status": status,
        "source": source,
        "interpolation_endpoints": json.dumps(interpolation_endpoints or []),
    }


def _endpoint(row: pd.Series) -> dict[str, Any]:
    params_str = row.get("params", "")
    params = json.loads(params_str) if isinstance(params_str, str) and params_str else {}
    trial = row.get("trial", "-")
    if hasattr(trial, "item"):
        trial = trial.item()
    return {
        "trial": trial,
        "avg_normalized_time": float(row.get("avg_normalized_time", row.get("time", 0))),
        "avg_compression_ratio": float(row.get("avg_compression_ratio", row.get("ratio", 0))),
        "params": params,
    }


def _dedupe_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    for _, group in df.dropna(subset=[x_col, y_col]).sort_values(x_col).groupby(x_col, sort=True):
        rows.append(group.loc[group[y_col].idxmin()])
    return pd.DataFrame(rows).reset_index(drop=True)


def _interpolate_front(
        front_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        target: float,
) -> tuple[float, str, list[dict[str, Any]]]:
    ordered = _dedupe_xy(front_df, x_col, y_col)
    if ordered.empty:
        raise ValueError("Cannot interpolate on an empty Pareto front.")

    xs = ordered[x_col].to_numpy(dtype=float)
    ys = ordered[y_col].to_numpy(dtype=float)

    exact_matches = [i for i, value in enumerate(xs) if math.isclose(value, target)]
    if exact_matches:
        idx = exact_matches[0]
        return float(ys[idx]), "exact", [_endpoint(ordered.iloc[idx])]

    if target < float(xs.min()):
        return float(ys[0]), "target_below_front_range", [_endpoint(ordered.iloc[0])]

    if target > float(xs.max()):
        return float(ys[-1]), "target_above_front_range", [_endpoint(ordered.iloc[-1])]

    upper_idx = int(xs.searchsorted(target))
    lower_idx = max(0, upper_idx - 1)
    interpolated = float(np.interp(target, xs, ys))
    return interpolated, "interpolated", [
        _endpoint(ordered.iloc[lower_idx]),
        _endpoint(ordered.iloc[upper_idx]),
    ]


def _read_dataset_edges(db_path: Path) -> dict[str, int]:
    results_db = db_path.with_name("results.db")
    if not results_db.exists():
        return {}

    conn = sqlite3.connect(str(results_db))
    try:
        row = conn.execute("SELECT datasets FROM metadata LIMIT 1").fetchone()
    except sqlite3.DatabaseError:
        return {}
    finally:
        conn.close()

    if not row or not row[0]:
        return {}

    try:
        datasets = json.loads(row[0])
    except json.JSONDecodeError:
        return {}

    edges = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        name = dataset.get("short_name") or dataset.get("filename")
        edge_count = dataset.get("edges")
        if name and edge_count:
            edges[str(name)] = int(edge_count)
    return edges


def _matches_params(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, expected in params.items():
        if key not in df.columns:
            return pd.Series(False, index=df.index)
        if isinstance(expected, (int, float)):
            mask &= pd.to_numeric(df[key], errors="coerce").sub(float(expected)).abs() < 1e-9
        else:
            mask &= df[key].astype(str) == str(expected)
    return mask


def _baseline_default_record(
        data: pd.DataFrame,
        baseline: str,
        baseline_params: dict[str, Any],
        dataset_edges: dict[str, int],
) -> dict[str, Any] | None:
    required = {"dataset", "algorithm", "time", "ratio"}
    if not required.issubset(data.columns):
        return None

    sub = data[data["algorithm"] == baseline].copy()
    if sub.empty:
        return None

    sub = sub[_matches_params(sub, baseline_params)]
    if sub.empty:
        return None

    sub["time"] = pd.to_numeric(sub["time"], errors="coerce")
    sub["ratio"] = pd.to_numeric(sub["ratio"], errors="coerce")
    sub = sub.dropna(subset=["time", "ratio"])
    if sub.empty:
        return None

    per_dataset = sub.groupby("dataset", as_index=False)[["time", "ratio"]].mean()
    per_dataset["edges"] = per_dataset["dataset"].astype(str).map(dataset_edges)
    per_dataset = per_dataset.dropna(subset=["edges"])
    per_dataset = per_dataset[per_dataset["edges"] > 0]
    if per_dataset.empty:
        return None

    avg_normalized_time = (per_dataset["time"] / per_dataset["edges"]).mean()
    avg_compression_ratio = per_dataset["ratio"].mean()
    return _point_record(
        baseline,
        "baseline_default",
        float(avg_normalized_time),
        float(avg_compression_ratio),
        params=baseline_params,
        trial="default",
        status="baseline",
        source="results.db",
    )


def _baseline_default_record_single_dataset(
        ds_data: pd.DataFrame,
        baseline: str,
        baseline_params: dict[str, Any],
) -> dict[str, Any] | None:
    required = {"algorithm", "time", "ratio"}
    if not required.issubset(ds_data.columns):
        return None

    sub = ds_data[ds_data["algorithm"] == baseline].copy()
    if sub.empty:
        return None

    sub = sub[_matches_params(sub, baseline_params)]
    if sub.empty:
        return None

    sub["time"] = pd.to_numeric(sub["time"], errors="coerce")
    sub["ratio"] = pd.to_numeric(sub["ratio"], errors="coerce")
    sub = sub.dropna(subset=["time", "ratio"])
    if sub.empty:
        return None

    avg_time = sub["time"].mean()
    avg_ratio = sub["ratio"].mean()

    return _point_record(
        baseline,
        "baseline_default",
        float(avg_time),
        float(avg_ratio),
        params=baseline_params,
        trial="default",
        status="baseline",
        source="results.db",
    )


def _baseline_default_from_study(
        storage_url: str,
        baseline: str,
        baseline_params: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        study = optuna.load_study(study_name=baseline, storage=storage_url)
    except Exception:
        return None

    matches = []
    for trial in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        if trial.values is None or len(trial.values) < 2:
            continue

        ok = True
        for key, expected in baseline_params.items():
            actual = trial.params.get(key)
            if actual is None:
                ok = False
                break
            if isinstance(expected, (int, float)):
                ok = math.isclose(float(actual), float(expected))
            else:
                ok = str(actual) == str(expected)
            if not ok:
                break

        if ok:
            matches.append(trial)

    if not matches:
        return None

    avg_time = sum(float(trial.values[0]) for trial in matches) / len(matches)
    avg_ratio = sum(float(trial.values[1]) for trial in matches) / len(matches)
    trial_label = matches[0].number if len(matches) == 1 else ",".join(str(trial.number) for trial in matches)
    return _point_record(
        baseline,
        "baseline_default",
        avg_time,
        avg_ratio,
        params=baseline_params,
        trial=trial_label,
        status="baseline",
        source="optuna_study.db",
    )


def _plot_global_front(
        algorithm: str,
        completed: list[optuna.trial.FrozenTrial],
        front_df: pd.DataFrame,
        point_df: pd.DataFrame,
        out_dir: Path,
        style: dict,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    completed_df = pd.DataFrame(
        {
            "avg_normalized_time": [float(t.values[0]) for t in completed],
            "avg_compression_ratio": [float(t.values[1]) for t in completed],
        }
    )

    ax.scatter(
        completed_df["avg_normalized_time"],
        completed_df["avg_compression_ratio"],
        color=style["color"],
        marker=style["marker"],
        alpha=0.18,
        s=22,
        label="completed trials",
        zorder=1,
    )

    ordered_front = front_df.sort_values("avg_normalized_time")
    ax.plot(
        ordered_front["avg_normalized_time"],
        ordered_front["avg_compression_ratio"],
        color=style["color"],
        linewidth=1.8,
        label="Pareto front",
        zorder=2,
    )

    point_styles = {
        "baseline_default": ("*", "#000000", 130),
        "fastest_time": ("P", "#D55E00", 100),
        "best_compression": ("X", "#009E73", 100),
        "knee_50_50": ("D", "#CC79A7", 95),
        "same_compression_as_baseline": ("^", "#882255", 100),
        "same_time_as_baseline": ("s", "#56B4E9", 90),
    }
    for _, row in point_df.iterrows():
        marker, color, size = point_styles.get(row["point_type"], ("o", "#000000", 80))
        label = str(row["point_type"])
        ax.scatter(
            row["avg_normalized_time"],
            row["avg_compression_ratio"],
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            s=size,
            label=label,
            zorder=4,
        )

    ax.set_title(f"Optuna Pareto front: {algorithm}")
    ax.set_xlabel("average normalized time")
    ax.set_ylabel("average compression ratio")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / f"pareto_{algorithm}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_combined_fronts(
        front_df: pd.DataFrame,
        point_df: pd.DataFrame,
        out_dir: Path,
        plotter: Plotter,
        baseline_algo: str,
        title: str,
        filename: str,
        x_label: str = "time",
        y_label: str = "compression ratio"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Plot baseline if it exists
    baseline = point_df[point_df["point_type"] == "baseline_default"]
    if not baseline.empty:
        row = baseline.iloc[0]
        ax.scatter(
            row["avg_normalized_time"],
            row["avg_compression_ratio"],
            marker="*",
            color="#000000",
            edgecolor="black",
            linewidth=0.6,
            s=130,
            label="baseline_default",
            zorder=4,
        )

    # Plot all algorithm Pareto fronts
    if not front_df.empty:
        for algo, group in front_df.groupby("algorithm"):
            ordered_front = group.sort_values("avg_normalized_time")
            style = plotter.get_algo_style(algo)

            ax.plot(
                ordered_front["avg_normalized_time"],
                ordered_front["avg_compression_ratio"],
                color=style["color"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.8,
                label=f"{algo}",
                zorder=2,
            )

    point_styles = {
        "fastest_time": ("P", "#D55E00", 100),
        "best_compression": ("X", "#009E73", 100),
        "knee_50_50": ("D", "#CC79A7", 95),
        "same_compression_as_baseline": ("^", "#882255", 100),
        "same_time_as_baseline": ("s", "#56B4E9", 90),
    }

    plotted_labels = set()

    for _, row in point_df.iterrows():
        pt_type = row["point_type"]
        algo = row["algorithm"]

        if pt_type == "baseline_default":
            continue

        if algo == baseline_algo:
            continue

        if pt_type in point_styles:
            marker, color, size = point_styles[pt_type]
            label = pt_type

            label_to_use = label if label not in plotted_labels else "_nolegend_"
            plotted_labels.add(label)

            ax.scatter(
                row["avg_normalized_time"],
                row["avg_compression_ratio"],
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                s=size,
                label=label_to_use,
                zorder=4,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    # --- ZOOM LOGIC ---
    mosso_front = front_df[front_df["algorithm"] == "kdd20-mosso"]
    if not mosso_front.empty:
        t_min = mosso_front["avg_normalized_time"].min()
        t_max = mosso_front["avg_normalized_time"].max()
        r_min = mosso_front["avg_compression_ratio"].min()
        r_max = mosso_front["avg_compression_ratio"].max()

        t_pad = (t_max - t_min) * 0.05
        r_pad = (r_max - r_min) * 0.05

        ax.set_xlim(max(0, t_min - t_pad), t_max + t_pad)
        ax.set_ylim(max(0, r_min - r_pad), r_max + r_pad)

    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


@register
class ParetoComparePlotter(Plotter):
    plotter_id = "pareto_compare_baseline"
    description = "Compare Optuna Pareto fronts with a baseline algorithm"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()
        generated_files = []

        db_path = None
        for parent in out_dir.parents:
            potential_db = parent / "optuna_study.db"
            if potential_db.exists():
                db_path = potential_db
                break

        if not db_path:
            console.print("[bold red]Error: Could not find optuna_study.db in parent directories.[/bold red]")
            return generated_files

        storage_url = f"sqlite:///{db_path}"
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        baseline_algo = options.get("baseline_algo", "kdd20-mosso")
        baseline_params = options.get("baseline_params", {})
        dataset_edges = _read_dataset_edges(db_path)

        if "dataset" not in data.columns:
            data = data.copy()
            data["dataset"] = context

        datasets = data["dataset"].unique()
        exclude_cols = {'dataset', 'algorithm', 'time', 'ratio', 'change_ratio', 'trial', 'power_of_2', 'edges_evaluated', 'time_micros'}
        param_cols = [c for c in data.columns if c not in exclude_cols]

        # ---------------------------------------------------------
        # PER-DATASET PARETO COMPARISON
        # ---------------------------------------------------------
        for ds in datasets:
            ds_data = data[data["dataset"] == ds].copy()
            if ds_data.empty:
                continue

            ds_front_records = []
            ds_point_records = []

            # 1. Baseline for this specific dataset
            baseline_point = _baseline_default_record_single_dataset(ds_data, baseline_algo, baseline_params)
            if baseline_point:
                ds_point_records.append(baseline_point)

            # 2. Extract Pareto Fronts from raw data for this dataset
            for algo in algos:
                algo_data = ds_data[ds_data["algorithm"] == algo].copy()
                if algo_data.empty:
                    continue

                algo_data = algo_data.sort_values(by=["ratio", "time"])
                pareto_front = []
                min_time = float('inf')

                for _, row in algo_data.iterrows():
                    if row["time"] < min_time:
                        pareto_front.append(row)
                        min_time = row["time"]

                if not pareto_front:
                    continue

                pareto_df = pd.DataFrame(pareto_front).reset_index(drop=True)

                # Format to match global structure for the plotter functions
                # Format to match global structure for the plotter functions
                algo_front_formatted = []
                for _, row in pareto_df.iterrows():
                    params = {p: row[p] for p in param_cols if pd.notna(row.get(p))}
                    record = _point_record(
                        algorithm=algo, # <--- Fixed parameter name
                        point_type="pareto_front",
                        avg_normalized_time=float(row["time"]),
                        avg_compression_ratio=float(row["ratio"]),
                        params=params,
                        trial=row.get("trial", "-")
                    )
                    algo_front_formatted.append(record)
                    ds_front_records.append(record)

                algo_front_df = pd.DataFrame(algo_front_formatted)

                # Characteristic points for this dataset
                fastest_idx = algo_front_df["avg_normalized_time"].idxmin()
                best_comp_idx = algo_front_df["avg_compression_ratio"].idxmin()

                ds_point_records.append(_point_record(algo, "fastest_time", algo_front_df.loc[fastest_idx, "avg_normalized_time"], algo_front_df.loc[fastest_idx, "avg_compression_ratio"], json.loads(algo_front_df.loc[fastest_idx, "params"]), algo_front_df.loc[fastest_idx, "trial"]))
                ds_point_records.append(_point_record(algo, "best_compression", algo_front_df.loc[best_comp_idx, "avg_normalized_time"], algo_front_df.loc[best_comp_idx, "avg_compression_ratio"], json.loads(algo_front_df.loc[best_comp_idx, "params"]), algo_front_df.loc[best_comp_idx, "trial"]))

                knee_row = _knee_row(algo_front_df, "avg_normalized_time", "avg_compression_ratio")
                ds_point_records.append(_point_record(algo, "knee_50_50", knee_row["avg_normalized_time"], knee_row["avg_compression_ratio"], json.loads(knee_row["params"]), knee_row["trial"]))

                if baseline_point:
                    target_time = float(baseline_point["avg_normalized_time"])
                    target_ratio = float(baseline_point["avg_compression_ratio"])

                    try:
                        interp_time, status, endpoints = _interpolate_front(algo_front_df, x_col="avg_compression_ratio", y_col="avg_normalized_time", target=target_ratio)
                        ds_point_records.append(_point_record(algo, "same_compression_as_baseline", interp_time, target_ratio, status=status, interpolation_endpoints=endpoints))
                    except Exception:
                        pass

                    try:
                        interp_ratio, status, endpoints = _interpolate_front(algo_front_df, x_col="avg_normalized_time", y_col="avg_compression_ratio", target=target_time)
                        ds_point_records.append(_point_record(algo, "same_time_as_baseline", target_time, interp_ratio, status=status, interpolation_endpoints=endpoints))
                    except Exception:
                        pass

            ds_front_df = pd.DataFrame(ds_front_records)
            ds_point_df = pd.DataFrame(ds_point_records)

            if not ds_point_df.empty:
                console.print(f"\n[bold cyan]=== Pareto Characteristic Points ({ds}) ===[/bold cyan]")
                table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
                table.add_column("Algorithm", style="green")
                table.add_column("Point Type", style="cyan")
                table.add_column("Trial", justify="right")
                table.add_column("Time (s)", justify="right")
                table.add_column("Ratio", justify="right")
                table.add_column("Parameters", justify="left", style="magenta")

                summary_df = ds_point_df[ds_point_df["point_type"] != "pareto_front"].sort_values(by=["algorithm", "point_type"])
                for _, row in summary_df.iterrows():
                    table.add_row(
                        str(row["algorithm"]), str(row["point_type"]), str(row["trial"]),
                        f"{row['avg_normalized_time']:.4g}", f"{row['avg_compression_ratio']:.4g}", str(row["params_display"])
                    )
                console.print(table)

                # Plot for this dataset
                try:
                    ds_compare_path = _plot_combined_fronts(
                        front_df=ds_front_df, point_df=ds_point_df, out_dir=out_dir, plotter=self,
                        baseline_algo=baseline_algo, title=f"Compare Pareto Fronts: {ds}",
                        filename=f"pareto_compare_{ds}.png", x_label="time (seconds)", y_label="relative size"
                    )
                    generated_files.append(ds_compare_path)
                except Exception as e:
                    console.print(f"[red]Failed plotting dataset fronts for {ds}: {e}[/red]")


        # ---------------------------------------------------------
        # GLOBAL PARETO COMPARISON
        # ---------------------------------------------------------
        front_records = []
        point_records = []

        baseline_point = _baseline_default_record(data, baseline_algo, baseline_params, dataset_edges)
        if not baseline_point:
            baseline_point = _baseline_default_from_study(storage_url, baseline_algo, baseline_params)

        if baseline_point:
            point_records.append(baseline_point)

        for algo in algos:
            try:
                study = optuna.load_study(study_name=algo, storage=storage_url)
            except Exception:
                continue

            pareto_trials = study.best_trials
            completed_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

            if not pareto_trials:
                continue

            algo_front = []
            for t in pareto_trials:
                if not t.values or len(t.values) < 2:
                    continue
                record = _trial_record(algo, t, "pareto_front")
                algo_front.append(record)
                front_records.append(record)

            if not algo_front:
                continue

            algo_front_df = pd.DataFrame(algo_front)

            fastest = min(pareto_trials, key=lambda t: float(t.values[0]))
            point_records.append(_trial_record(algo, fastest, "fastest_time"))

            best_comp = min(pareto_trials, key=lambda t: float(t.values[1]))
            point_records.append(_trial_record(algo, best_comp, "best_compression"))

            knee = _knee_trial(pareto_trials)
            point_records.append(_trial_record(algo, knee, "knee_50_50"))

            if baseline_point:
                target_time = float(baseline_point["avg_normalized_time"])
                target_ratio = float(baseline_point["avg_compression_ratio"])

                try:
                    interp_time, status, endpoints = _interpolate_front(algo_front_df, x_col="avg_compression_ratio", y_col="avg_normalized_time", target=target_ratio)
                    point_records.append(_point_record(algo, "same_compression_as_baseline", interp_time, target_ratio, status=status, interpolation_endpoints=endpoints))
                except Exception:
                    pass

                try:
                    interp_ratio, status, endpoints = _interpolate_front(algo_front_df, x_col="avg_normalized_time", y_col="avg_compression_ratio", target=target_time)
                    point_records.append(_point_record(algo, "same_time_as_baseline", target_time, interp_ratio, status=status, interpolation_endpoints=endpoints))
                except Exception:
                    pass

            algo_point_df = pd.DataFrame([r for r in point_records if r["algorithm"] == algo or r["point_type"] == "baseline_default"])
            style = self.get_algo_style(algo)

            try:
                algo_path = _plot_global_front(algo, completed_trials, algo_front_df, algo_point_df, out_dir, style)
                generated_files.append(algo_path)
            except Exception:
                pass

        front_df = pd.DataFrame(front_records)
        point_df = pd.DataFrame(point_records)

        if not point_df.empty:
            console.print("\n[bold cyan]=== Pareto Characteristic Points (Global) ===[/bold cyan]")
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
            table.add_column("Algorithm", style="green")
            table.add_column("Point Type", style="cyan")
            table.add_column("Trial", justify="right")
            table.add_column("Norm. Time", justify="right")
            table.add_column("Comp. Ratio", justify="right")
            table.add_column("Parameters", justify="left", style="magenta")

            summary_df = point_df[point_df["point_type"] != "pareto_front"].sort_values(by=["algorithm", "point_type"])

            for _, row in summary_df.iterrows():
                table.add_row(
                    str(row["algorithm"]), str(row["point_type"]), str(row["trial"]),
                    f"{row['avg_normalized_time']:.4g}", f"{row['avg_compression_ratio']:.4g}", str(row["params_display"])
                )
            console.print(table)

            txt_path = out_dir / "pareto_compare_summary.txt"
            console.save_text(str(txt_path))
            generated_files.append(txt_path)

        try:
            compare_path = _plot_combined_fronts(
                front_df=front_df, point_df=point_df, out_dir=out_dir, plotter=self,
                baseline_algo=baseline_algo, title="Compare Optuna Pareto Fronts (Global)",
                filename="pareto_compare_global.png"
            )
            generated_files.append(compare_path)
        except Exception:
            pass

        return generated_files