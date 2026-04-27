"""
Analysis CLI for benchmark sessions.

Usage:  ./run.sh analyze
"""
import json
import logging
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from scripts.db import read_results, get_metadata

console = Console(highlight=False)

_BENCHMARK_DIR = Path("output/benchmarks")

_PLOTS_BY_TYPE: dict[str, list[tuple[str, str]]] = {
    "compare":              [("bar",        "Bar chart — time & ratio per dataset"),
                             ("scalability","Log-log scalability vs |E|"),
                             ("variance",   "Per-run variance (requires runs > 1)")],
    "reps":                 [("significance", "Significance test: Wilcoxon/Cliff's/Holm vs baseline"),
                             ("iso-ratio",   "Iso-ratio speedup table (needs BO session)"),
                             ("boxplots",    "Significance boxplots with stars"),
                             ("violins",     "Violin plots with jitter overlay")],
    "bayesian":             [("pareto",      "Pareto-front scatter (time vs ratio)"),
                             ("convergence", "Hypervolume convergence over trials"),
                             ("reps",        "Run N JVM reps at knee-point configs → new reps session")],
    "sweep":                [("sweep",      "Parameter sensitivity line plots")],
    "incremental_vs_batch": [("update-cost","Update cost per checkpoint")],
}

def _scan_sessions(benchmark_dir: Path) -> list[dict]:
    """Return list of session info dicts sorted newest-first."""
    sessions = []
    for type_dir in sorted(benchmark_dir.iterdir()):
        if not type_dir.is_dir() or type_dir.name == "versions":
            continue
        for run_dir in sorted(type_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            has_db = (run_dir / "results.db").exists()
            info = _read_session_info(run_dir)
            sessions.append({
                "path":       run_dir,
                "type":       info.get("benchmark_type", type_dir.name),
                "timestamp":  run_dir.name.replace("run_", ""),
                "algorithms": info.get("algorithms", []),
                "n_datasets": info.get("n_datasets", "?"),
                "has_db":     has_db,
            })
    return sessions


def _read_session_info(run_dir: Path) -> dict:
    info = {}
    # Prefer metadata.json (richer)
    meta_json = run_dir / "metadata.json"
    if meta_json.exists():
        try:
            data = json.loads(meta_json.read_text())
            info["benchmark_type"] = data.get("benchmark_type", "")
            cli = data.get("cli_args", {})
            algos = cli.get("algorithm") or list(data.get("algorithms", {}).keys())
            info["algorithms"] = algos if isinstance(algos, list) else [algos]
            # Count datasets from DB if possible
            db_meta = get_metadata(run_dir)
            if db_meta:
                cli_db = json.loads(db_meta.get("cli_args", "{}"))
                ds_arg = cli_db.get("dataset")
                info["n_datasets"] = len(ds_arg) if ds_arg else "all"
            else:
                info["n_datasets"] = cli.get("group", "?")
            return info
        except Exception:
            pass
    # Fall back to results.db metadata
    db_meta = get_metadata(run_dir)
    if db_meta:
        info["benchmark_type"] = db_meta.get("benchmark_type", "")
        cli = json.loads(db_meta.get("cli_args", "{}"))
        algos = cli.get("algorithm") or []
        info["algorithms"] = algos if isinstance(algos, list) else [algos]
        info["n_datasets"] = cli.get("group", "?")
    return info


def _session_label(s: dict) -> str:
    ts = s["timestamp"]
    date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
    algos = ", ".join(s["algorithms"]) if s["algorithms"] else "?"
    db_flag = "" if s["has_db"] else "  (no DB)"
    return f"{s['type']:<22} {date}   {algos}{db_flag}"


def _pick_sessions(db_sessions: list[dict], questionary, Choice) -> list[Path] | None:
    """Prompt user to pick sessions. Returns list of Paths, or None on Ctrl-C."""
    while True:
        choices = [Choice(title=_session_label(s), value=s["path"]) for s in db_sessions]
        result = questionary.checkbox(
            "Select session(s) to analyse:",
            choices=choices,
            instruction="(space=toggle, enter=confirm)",
        ).ask()
        if result is None:
            return None
        if result:
            return result
        console.print("[yellow]Select at least one session (press space to toggle).[/yellow]")


def _interactive(benchmark_dir: Path) -> None:
    try:
        import questionary
        from questionary import Choice
    except ImportError:
        console.print("[red]questionary not installed.[/red] Run: pip install questionary")
        sys.exit(1)

    console.print()
    console.rule("[bold]MOSSO Analysis[/bold]")
    console.print(f"[dim]Scanning {benchmark_dir} ...[/dim]")

    all_sessions = _scan_sessions(benchmark_dir)
    db_sessions = [s for s in all_sessions if s["has_db"]]

    if not all_sessions:
        console.print("[red]No benchmark sessions found.[/red]")
        sys.exit(1)

    if not db_sessions:
        console.print("[yellow]Sessions found but none have results.db yet.[/yellow]")
        console.print("Re-run benchmarks to generate the DB, then use analyze.")
        sys.exit(1)

    _show_session_summary(all_sessions)

    # ── Session selection ───────────────────────────────────────────────────
    selected_sessions = _pick_sessions(db_sessions, questionary, Choice)
    if selected_sessions is None:
        console.print("\n[dim]Cancelled.[/dim]")
        return

    # ── Main analysis loop ─────────────────────────────────────────────────
    df_cache: dict[tuple, object] = {}

    while True:
        # Determine available plots for the chosen sessions
        types_in_use = {s["type"] for s in db_sessions if s["path"] in selected_sessions}
        available_plots: list[tuple[str, str]] = []
        seen = set()
        for t in types_in_use:
            for key, desc in _PLOTS_BY_TYPE.get(t, []):
                if key not in seen:
                    available_plots.append((key, desc))
                    seen.add(key)

        if not available_plots:
            console.print("[yellow]No plots available for the selected session types.[/yellow]")
            break

        plot_choices = [Choice(title=f"{k:<14} {desc}", value=k) for k, desc in available_plots]
        chosen_plots = questionary.checkbox(
            "Which plots to generate?",
            choices=plot_choices,
            instruction="(space to toggle, enter to confirm)",
        ).ask()

        if chosen_plots is None:
            console.print("\n[dim]Cancelled.[/dim]")
            return
        if not chosen_plots:
            console.print("[yellow]Select at least one plot.[/yellow]")
            continue

        # ── Optional filters ───────────────────────────────────────────────
        df_key = tuple(sorted(str(p) for p in selected_sessions))
        if df_key not in df_cache:
            df_cache[df_key] = read_results(selected_sessions)
        full_df = df_cache[df_key]

        all_algos = sorted(full_df["algorithm"].unique())
        all_datasets = sorted(full_df["dataset"].unique())

        # Algorithm filter
        algo_choices = [Choice(title=a, value=a, checked=True) for a in all_algos]
        chosen_algos = questionary.checkbox(
            "Algorithms to include:",
            choices=algo_choices,
        ).ask()
        if chosen_algos is None:
            return
        chosen_algos = chosen_algos or all_algos

        # Dataset filter
        ds_choices = [Choice(title=d, value=d, checked=True) for d in all_datasets]
        chosen_datasets = questionary.checkbox(
            "Datasets to include:",
            choices=ds_choices,
        ).ask()
        if chosen_datasets is None:
            return
        chosen_datasets = chosen_datasets or all_datasets

        # Baseline (only for reps / compare)
        baseline = None
        if any(t in types_in_use for t in ("reps", "compare")):
            b_choices = [Choice(title="(none)", value=None)] + [
                Choice(title=a, value=a) for a in chosen_algos
            ]
            baseline = questionary.select("Baseline algorithm:", choices=b_choices).ask()
            if baseline is None:
                return

        # Output dir
        default_out = str(selected_sessions[0] / "analysis")
        out_str = questionary.text(
            "Output directory:", default=default_out
        ).ask()
        if out_str is None:
            return
        output_dir = Path(out_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Run reps (bayesian sessions only) ─────────────────────────────
        if "reps" in chosen_plots:
            _run_reps_interactive(selected_sessions, chosen_algos, chosen_datasets,
                                  questionary, Choice)

        analysis_plots = [p for p in chosen_plots if p != "reps"]
        if not analysis_plots:
            pass  # only reps was selected; skip generate section below
        else:
            # ── Generate ─────────────────────────────────────────────────
            console.print()
            console.rule("[bold]Generating[/bold]")

            import argparse as _ap
            fake_args = _ap.Namespace(
                algorithms=chosen_algos,
                datasets=chosen_datasets,
                baseline=baseline,
                output=str(output_dir),
            )

            df = read_results(selected_sessions, algorithms=chosen_algos, datasets=chosen_datasets)
            logger = _make_logger()

            from scripts.analysis import plotting as P
            for plot_type in analysis_plots:
                try:
                    _dispatch_plot(plot_type, df, selected_sessions, output_dir, fake_args, P, logger)
                except Exception as e:
                    console.print(f"  [red]✗[/red]  {plot_type}: {e}")
                    traceback.print_exc()

        # ── What next? ────────────────────────────────────────────────────
        console.print()
        action = questionary.select(
            "What next?",
            choices=[
                Choice("Generate more plots from these sessions",  "more"),
                Choice("Change sessions",                          "sessions"),
                Choice("Exit",                                     "exit"),
            ],
        ).ask()

        if action is None or action == "exit":
            console.print("\n[dim]Done.[/dim]")
            return
        elif action == "sessions":
            result = _pick_sessions(db_sessions, questionary, Choice)
            if result is None:
                return
            selected_sessions = result
        # "more" → loop continues with same sessions


def _show_session_summary(sessions: list[dict]) -> None:
    by_type: dict[str, int] = {}
    for s in sessions:
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Benchmark type")
    t.add_column("Sessions", justify="right")
    t.add_column("Has DB", justify="center")
    for btype, count in sorted(by_type.items()):
        db_count = sum(1 for s in sessions if s["type"] == btype and s["has_db"])
        t.add_row(btype, str(count), f"{db_count}/{count}")
    console.print(t)



def _run_reps_interactive(
    session_dirs: list[Path],
    chosen_algos: list[str],
    chosen_datasets: list[str],
    questionary,
    Choice,
) -> None:
    from datetime import datetime
    import pandas as pd
    from scripts.config import ALGORITHMS, DATASETS, EXPERIMENT_DIR, PARAM_CONFIG
    from scripts.datasets import download_dataset
    from scripts.runners.base_runner import get_runner
    import scripts.db as db

    reps_str = questionary.text("Repetitions per (algo, dataset):", default="15").ask()
    if reps_str is None:
        return
    try:
        n_reps = int(reps_str)
    except ValueError:
        console.print("[red]Invalid number.[/red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reps_session_dir = Path(EXPERIMENT_DIR) / "reps" / f"run_{timestamp}"
    reps_session_dir.mkdir(parents=True, exist_ok=True)

    logger = _make_logger()

    db_conn = db.init_db(reps_session_dir)
    db.write_metadata(
        db_conn, "reps",
        {"bo_session": [str(s) for s in session_dirs]},
        reps_session_dir,
    )

    all_ds_configs = {d["short_name"]: d for group in DATASETS.values() for d in group}

    results = []
    knee_log = []

    console.print()
    console.rule("[bold]Running Reps[/bold]")

    for dataset_name in chosen_datasets:
        ds_config = all_ds_configs.get(dataset_name)
        if ds_config is None:
            console.print(f"  [yellow]![/yellow]  {dataset_name}: not in DATASETS config, skipping")
            continue

        dataset_path = download_dataset(ds_config["url"], ds_config["filename"], logger)
        if not dataset_path:
            console.print(f"  [yellow]![/yellow]  {dataset_name}: download failed, skipping")
            continue

        for algo_name in chosen_algos:
            study = _load_bo_study(algo_name, dataset_name, session_dirs)
            if study is None:
                console.print(f"  [yellow]![/yellow]  No BO study for {algo_name}/{dataset_name}, skipping")
                continue

            knee = _knee_from_study(study)
            if knee is None:
                console.print(f"  [yellow]![/yellow]  No knee for {algo_name}/{dataset_name}, skipping")
                continue

            row, k_time, k_ratio = knee
            params = {c.replace("params_", ""): row[c] for c in row.index if c.startswith("params_")}
            for k, v in list(params.items()):
                if isinstance(v, float) and v.is_integer():
                    params[k] = int(v)

            console.print(
                f"  [cyan]{algo_name}[/cyan] / {dataset_name}: "
                f"time={k_time:.3f} ratio={k_ratio:.5f} params={params}"
            )

            algo_config = ALGORITHMS.get(algo_name, {})
            runner = get_runner(algo_name, logger, str(reps_session_dir))
            if not runner.binary_exists():
                console.print(f"  [yellow]![/yellow]  Binary missing for {algo_name}, skipping")
                continue

            template = algo_config.get("template", [])
            merged = {**params}
            for p_key in template:
                if p_key not in merged:
                    merged[p_key] = PARAM_CONFIG.get(p_key, {}).get("default", "")

            _, _, times, ratios = runner.run_multiple(
                dataset_path=dataset_path,
                base_output_name=f"{algo_name}_{dataset_name}_{timestamp}",
                n_runs=n_reps,
                parameters=merged,
                template=template,
            )

            knee_log.append({
                "Dataset": dataset_name, "Algorithm": algo_name,
                "Knee_Time": k_time, "Knee_Ratio": k_ratio,
                **{f"param_{k}": v for k, v in params.items()},
            })

            for i, (t, r) in enumerate(zip(times or [], ratios or []), 1):
                results.append({
                    "Dataset": dataset_name, "Algorithm": algo_name,
                    "Rep": i, "Time": t, "Ratio": r,
                })
                db.write_result(
                    db_conn, algorithm=algo_name, dataset=dataset_name,
                    time=t, ratio=r, rep=i, params=params,
                )

    if knee_log:
        pd.DataFrame(knee_log).to_csv(reps_session_dir / "knee_points.csv", index=False)

    db_conn.close()

    console.print(f"\n  [green]✓[/green]  Reps session: {reps_session_dir}")
    console.print(f"  [dim]{len(results)} results written[/dim]")

def _dispatch_plot(plot_type: str, df, session_dirs: list[Path], output_dir: Path, args, P, logger) -> None:
    if plot_type == "bar":
        _plot_bar(df, output_dir, args, P, logger)
    elif plot_type == "scalability":
        _plot_scalability(df, output_dir, args, P, logger)
    elif plot_type == "variance":
        _plot_variance(df, output_dir, args, P, logger)
    elif plot_type == "boxplots":
        _plot_boxplots(df, output_dir, args, P, logger)
    elif plot_type == "violins":
        _plot_violins(df, output_dir, args, P, logger)
    elif plot_type == "pareto":
        _plot_pareto(df, output_dir, args, P, logger)
    elif plot_type == "convergence":
        _plot_convergence(session_dirs, output_dir, P, logger)
    elif plot_type == "heatmap":
        _plot_heatmap(df, output_dir, P, logger)
    elif plot_type == "sweep":
        _plot_sweep(df, output_dir, args, P, logger)
    elif plot_type == "update-cost":
        _plot_update_cost(df, output_dir, P, logger)
    elif plot_type in ("summary", "significance"):
        _emit_significance_summary(df, session_dirs, output_dir, args)
    elif plot_type == "iso-ratio":
        _emit_iso_ratio(df, session_dirs, output_dir, args)
    else:
        console.print(f"  [yellow]?[/yellow]  Unknown plot type: {plot_type}")

def _ok(path: str) -> None:
    console.print(f"  [green]✓[/green]  {path}")


def _wide_compare(df):
    import pandas as pd
    rows: dict = {}
    for _, r in df.iterrows():
        ds, algo = r["dataset"], r["algorithm"]
        if ds not in rows:
            rows[ds] = {"Dataset": ds}
        rows[ds][f"Time_{algo}"]  = r["time"]
        rows[ds][f"Ratio_{algo}"] = r["ratio"]
    return pd.DataFrame(list(rows.values()))


def _plot_bar(df, output_dir, args, P, logger) -> None:
    import tempfile
    wide = _wide_compare(df)
    drop = [c for c in wide.columns if any(p in c for p in ["_std_", "_ci_", "Memory_", "Edges"])]
    wide = wide.drop(columns=drop, errors="ignore")
    out = str(output_dir / "compare_plot.pdf")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        wide.to_csv(f.name, index=False)
    P.plot_results(f.name, out, logger)
    _ok(out)


def _plot_scalability(df, output_dir, args, P, logger) -> None:
    from scripts.config import DATASETS
    wide = _wide_compare(df)
    all_ds = {d["short_name"]: d.get("meta", {}).get("edges")
              for group in DATASETS.values() for d in group}
    wide["Edges"] = wide["Dataset"].map(all_ds)
    algos = [c.replace("Time_", "") for c in wide.columns if c.startswith("Time_")]
    out = str(output_dir / "scalability_plot.pdf")
    P.plot_scalability(wide, algos, out, logger)
    _ok(out)


def _plot_variance(df, output_dir, args, P, logger) -> None:
    times_dict, ratios_dict = {}, {}
    for algo, group in df.groupby("algorithm"):
        times_dict[algo]  = group["time"].dropna().tolist()
        ratios_dict[algo] = group["ratio"].dropna().tolist()
    out = str(output_dir / "runs_variance_plot.pdf")
    P.plot_runs_variance(out, times_dict, ratios_dict, output_dir)
    _ok(out)


def _plot_boxplots(df, output_dir, args, P, logger) -> None:
    import tempfile
    raw = _to_sig_raw(df)
    baseline = _resolve_baseline(args)
    alpha, seed = _resolve_alpha_seed(args)
    summary = _build_significance_summary(raw, baseline, alpha=alpha, seed=seed)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as rf:
        raw.to_csv(rf.name, index=False)
        raw_path = rf.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as sf:
        summary.to_csv(sf.name, index=False)
        sum_path = sf.name
    out = str(output_dir / "significance_boxplots.pdf")
    P.plot_significance_boxplots(raw_path, sum_path, out, baseline)
    _ok(out)


def _plot_violins(df, output_dir, args, P, logger) -> None:
    import tempfile
    raw = _to_sig_raw(df)
    baseline = _resolve_baseline(args)
    alpha, seed = _resolve_alpha_seed(args)
    summary = _build_significance_summary(raw, baseline, alpha=alpha, seed=seed)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as rf:
        raw.to_csv(rf.name, index=False)
        raw_path = rf.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as sf:
        summary.to_csv(sf.name, index=False)
        sum_path = sf.name
    out = str(output_dir / "significance_violins.pdf")
    P.plot_significance_violins(raw_path, sum_path, out, baseline)
    _ok(out)


def _to_sig_raw(df):
    return df.rename(columns={"algorithm": "Algorithm", "dataset": "Dataset",
                              "rep": "Rep", "time": "Time", "ratio": "Ratio"})


def _resolve_baseline(args) -> str | None:
    return getattr(args, "baseline", None)


def _resolve_alpha_seed(args) -> tuple[float, int]:
    alpha = getattr(args, "alpha", None) or 0.05
    seed = getattr(args, "seed", None) or 42
    return float(alpha), int(seed)


def _build_significance_summary(
    raw,
    baseline: str | None,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Wilcoxon / Cliff's delta / bootstrap CI / Holm-Bonferroni summary.

    Expects `raw` with columns: Algorithm, Dataset, Rep, Time, Ratio.
    Returns empty DataFrame if baseline is missing or not in data.
    """
    import numpy as np
    import pandas as pd
    from scripts.analysis.stats import (
        get_confidence_interval, cliffs_delta, holm_bonferroni, paired_wilcoxon,
    )

    if raw is None or raw.empty or not baseline:
        return pd.DataFrame()
    if baseline not in raw["Algorithm"].unique():
        return pd.DataFrame()

    non_baseline = sorted(a for a in raw["Algorithm"].unique() if a != baseline)

    med = raw.groupby(["Algorithm", "Dataset"])[["Time", "Ratio"]].median().reset_index()
    baseline_time_med = dict(zip(med[med["Algorithm"] == baseline]["Dataset"],
                                 med[med["Algorithm"] == baseline]["Time"]))
    baseline_ratio_med = dict(zip(med[med["Algorithm"] == baseline]["Dataset"],
                                  med[med["Algorithm"] == baseline]["Ratio"]))
    baseline_time_pool = raw[raw["Algorithm"] == baseline]["Time"].tolist()
    baseline_ratio_pool = raw[raw["Algorithm"] == baseline]["Ratio"].tolist()

    raw_p_time, raw_p_ratio = {}, {}
    rows = []
    for algo in non_baseline:
        sub = med[med["Algorithm"] == algo]
        s_time = dict(zip(sub["Dataset"], sub["Time"]))
        s_ratio = dict(zip(sub["Dataset"], sub["Ratio"]))

        _, p_time = paired_wilcoxon(s_time, baseline_time_med)
        _, p_ratio = paired_wilcoxon(s_ratio, baseline_ratio_med)
        raw_p_time[algo], raw_p_ratio[algo] = p_time, p_ratio

        algo_time_pool = raw[raw["Algorithm"] == algo]["Time"].tolist()
        algo_ratio_pool = raw[raw["Algorithm"] == algo]["Ratio"].tolist()
        d_time, mag_time = cliffs_delta(algo_time_pool, baseline_time_pool)
        d_ratio, mag_ratio = cliffs_delta(algo_ratio_pool, baseline_ratio_pool)

        shared = sorted(set(s_time) & set(baseline_time_med))
        if shared:
            t_ratio_arr = np.array([s_time[k] / baseline_time_med[k] for k in shared])
            r_ratio_arr = np.array([s_ratio[k] / baseline_ratio_med[k] for k in shared])
            t_rel = float(np.median(t_ratio_arr))
            r_rel = float(np.median(r_ratio_arr))
        else:
            t_rel = r_rel = float("nan")

        t_ci_lo, t_ci_hi = get_confidence_interval(algo_time_pool, seed=seed)
        r_ci_lo, r_ci_hi = get_confidence_interval(algo_ratio_pool, seed=seed)

        rows.append({
            "Strategy": algo,
            "Datasets": len(shared),
            "Median Time / Baseline": t_rel,
            "Time CI (lo)": t_ci_lo,
            "Time CI (hi)": t_ci_hi,
            "p_time (raw)": p_time,
            "Cliff's d_time": d_time,
            "d_time mag": mag_time,
            "Median Ratio / Baseline": r_rel,
            "Ratio CI (lo)": r_ci_lo,
            "Ratio CI (hi)": r_ci_hi,
            "p_ratio (raw)": p_ratio,
            "Cliff's d_ratio": d_ratio,
            "d_ratio mag": mag_ratio,
        })

    adj_time = holm_bonferroni(raw_p_time, alpha=alpha)
    adj_ratio = holm_bonferroni(raw_p_ratio, alpha=alpha)
    for row in rows:
        algo = row["Strategy"]
        ap_t, sig_t = adj_time.get(algo, (float("nan"), False))
        ap_r, sig_r = adj_ratio.get(algo, (float("nan"), False))
        row["p_time (Holm)"] = ap_t
        row["p_ratio (Holm)"] = ap_r
        row["Verdict_Time"] = _verdict(row["Median Time / Baseline"], sig_t)
        row["Verdict_Ratio"] = _verdict(row["Median Ratio / Baseline"], sig_r)

    col_order = [
        "Strategy", "Datasets",
        "Median Time / Baseline", "Time CI (lo)", "Time CI (hi)",
        "p_time (raw)", "p_time (Holm)", "Cliff's d_time", "d_time mag", "Verdict_Time",
        "Median Ratio / Baseline", "Ratio CI (lo)", "Ratio CI (hi)",
        "p_ratio (raw)", "p_ratio (Holm)", "Cliff's d_ratio", "d_ratio mag", "Verdict_Ratio",
    ]
    return pd.DataFrame(rows)[col_order] if rows else pd.DataFrame()


def _verdict(rel_median: float, significant: bool) -> str:
    import numpy as np
    if np.isnan(rel_median):
        return "n/a"
    if not significant:
        return "n.s."
    return "better*" if rel_median < 1.0 else "worse*"


def _bo_session_dirs(session_dirs: list[Path]) -> list[Path]:
    """Read --bo-session paths from each significance session's metadata.json."""
    out: list[Path] = []
    for sdir in session_dirs:
        meta_json = sdir / "metadata.json"
        if not meta_json.exists():
            continue
        try:
            data = json.loads(meta_json.read_text())
            bo = data.get("cli_args", {}).get("bo_session") or []
            if isinstance(bo, str):
                bo = [bo]
            out.extend(Path(p) for p in bo)
        except Exception:
            continue
    # Deduplicate, preserve order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _load_bo_study(algo: str, dataset: str, bo_dirs: list[Path]):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"{algo}_{dataset}"
    for d in bo_dirs:
        db_path = d / "optuna_study.db"
        if not db_path.exists():
            continue
        try:
            return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        except KeyError:
            continue
        except Exception:
            continue
    return None


def _knee_from_study(study):
    """(params, knee_time, knee_ratio) or None."""
    from scripts.analysis.plotting import get_pareto_front_2d
    from scripts.analysis.stats import knee_point
    df = study.trials_dataframe()
    if df.empty or "values_0" not in df.columns or "values_1" not in df.columns:
        return None
    df = df[df["state"] == "COMPLETE"].dropna(subset=["values_0", "values_1"])
    if df.empty:
        return None
    df = df.rename(columns={"values_0": "Time", "values_1": "Ratio"})
    pareto = get_pareto_front_2d(df, "Time", "Ratio").sort_values("Time").reset_index(drop=True)
    if pareto.empty:
        return None
    idx = knee_point(pareto["Time"].tolist(), pareto["Ratio"].tolist())
    row = pareto.iloc[idx]
    return row, float(row["Time"]), float(row["Ratio"])


def _build_iso_ratio_table(
    df,
    session_dirs: list[Path],
    baseline: str | None,
    bo_dirs: list[Path] | None = None,
):
    """Interpolate each algo's Pareto front to baseline's knee ratio. Needs BO DBs.

    `bo_dirs`: explicit BO session dirs; if None, read from session metadata.json.
    """
    import pandas as pd
    from scripts.analysis.plotting import get_pareto_front_2d
    from scripts.analysis.stats import interp_time_at_ratio

    if not baseline or df is None or df.empty:
        return pd.DataFrame()

    if bo_dirs is None:
        bo_dirs = _bo_session_dirs(session_dirs)
    if not bo_dirs:
        console.print("  [yellow]![/yellow]  iso-ratio: no BO sessions found in session metadata")
        return pd.DataFrame()

    algos = sorted(df["algorithm"].unique())
    datasets = sorted(df["dataset"].unique())
    rows = []

    for dataset_name in datasets:
        b_study = _load_bo_study(baseline, dataset_name, bo_dirs)
        if b_study is None:
            continue
        b_knee = _knee_from_study(b_study)
        if b_knee is None:
            continue
        _, b_time, b_ratio = b_knee

        rows.append({
            "Dataset": dataset_name,
            "Algorithm": baseline,
            "Baseline_Knee_Ratio": b_ratio,
            "Baseline_Knee_Time_s": b_time,
            "Iso_Time_s": b_time,
            "Speedup_vs_Baseline": 1.0,
            "Pct_Slower": 0.0,
            "Note": "baseline",
        })

        for algo_name in algos:
            if algo_name == baseline:
                continue
            study = _load_bo_study(algo_name, dataset_name, bo_dirs)
            if study is None:
                continue
            sdf = study.trials_dataframe()
            if sdf.empty or "values_0" not in sdf.columns:
                continue
            sdf = sdf[sdf["state"] == "COMPLETE"].dropna(subset=["values_0", "values_1"])
            if sdf.empty:
                continue
            sdf = sdf.rename(columns={"values_0": "Time", "values_1": "Ratio"})
            pareto = get_pareto_front_2d(sdf, "Time", "Ratio")
            if pareto.empty:
                continue
            iso_time = interp_time_at_ratio(pareto, b_ratio)
            if iso_time is None:
                continue
            speedup = b_time / iso_time
            pct_slower = (iso_time / b_time - 1.0) * 100.0
            direction = "faster" if speedup > 1.0 else "slower"
            rows.append({
                "Dataset": dataset_name,
                "Algorithm": algo_name,
                "Baseline_Knee_Ratio": b_ratio,
                "Baseline_Knee_Time_s": b_time,
                "Iso_Time_s": iso_time,
                "Speedup_vs_Baseline": speedup,
                "Pct_Slower": pct_slower,
                "Note": f"{direction} by {abs(pct_slower):.1f}%",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _emit_significance_summary(df, session_dirs: list[Path], output_dir: Path, args) -> None:
    from tabulate import tabulate
    baseline = _resolve_baseline(args)
    if not baseline:
        console.print("  [yellow]![/yellow]  summary: --baseline required")
        return
    alpha, seed = _resolve_alpha_seed(args)
    raw = _to_sig_raw(df)
    summary = _build_significance_summary(raw, baseline, alpha=alpha, seed=seed)
    if summary.empty:
        console.print("  [yellow]![/yellow]  summary: empty (baseline absent from data?)")
        return
    csv_path = output_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)
    _ok(str(csv_path))

    txt_path = output_dir / "table_results.txt"
    header = f"\n--- SIGNIFICANCE vs {baseline} (alpha={alpha}) ---\n"
    body = tabulate(summary, headers="keys", tablefmt="grid",
                    showindex=False, floatfmt=".4f") + "\n"
    mode = "a" if txt_path.exists() else "w"
    with open(txt_path, mode) as f:
        f.write(header + body)
    _ok(str(txt_path))


def _emit_iso_ratio(df, session_dirs: list[Path], output_dir: Path, args) -> None:
    from tabulate import tabulate
    baseline = _resolve_baseline(args)
    if not baseline:
        console.print("  [yellow]![/yellow]  iso-ratio: --baseline required")
        return
    iso_df = _build_iso_ratio_table(df, session_dirs, baseline)
    if iso_df.empty:
        console.print("  [yellow]![/yellow]  iso-ratio: no rows produced")
        return
    csv_path = output_dir / "iso_ratio.csv"
    iso_df.to_csv(csv_path, index=False)
    _ok(str(csv_path))

    txt_path = output_dir / "table_results.txt"
    header = f"\n--- ISO-RATIO: time to match {baseline} knee-point compression ---\n"
    body = tabulate(iso_df, headers="keys", tablefmt="grid",
                    showindex=False, floatfmt=".4f") + "\n"
    mode = "a" if txt_path.exists() else "w"
    with open(txt_path, mode) as f:
        f.write(header + body)
    _ok(str(txt_path))


def _plot_pareto(df, output_dir, args, P, logger) -> None:
    import tempfile
    bo = df.rename(columns={"algorithm": "Algorithm", "dataset": "Dataset",
                             "time": "Time", "ratio": "Ratio"})
    for c in [c for c in df.columns if c.startswith("param_")]:
        bo[c.replace("param_", "")] = df[c]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        bo.to_csv(f.name, index=False)
    out = str(output_dir / "pareto_front.pdf")
    P.plot_pareto_front(f.name, out, logger)
    _ok(out)


def _plot_convergence(session_dirs, output_dir, P, logger) -> None:
    for sdir in session_dirs:
        db_path = sdir / "optuna_study.db"
        if db_path.exists():
            out = str(output_dir / "bo_convergence.pdf")
            P.plot_bo_convergence(str(db_path), out, logger)
            _ok(out)
            return
    console.print("  [yellow]![/yellow]  convergence: no optuna_study.db found")


def _plot_heatmap(df, output_dir, P, logger) -> None:
    param_cols = [c.replace("param_", "") for c in df.columns if c.startswith("param_")]
    if len(param_cols) < 2:
        console.print("  [yellow]![/yellow]  heatmap: need ≥ 2 param columns")
        return
    for algo, group in df.groupby("algorithm"):
        g = group.rename(columns={"time": "Time", "ratio": "Ratio",
                                   "dataset": "Dataset", "algorithm": "Algorithm"})
        for c in param_cols:
            g[c] = df.loc[group.index, f"param_{c}"]
        out = str(output_dir / f"heatmap_{algo}.pdf")
        P.plot_heatmap(g, param_cols[0], param_cols[1], out, title=f"Parameter Heatmap: {algo}")
        _ok(out)


def _plot_sweep(df, output_dir, args, P, logger) -> None:
    import tempfile
    import pandas as pd
    param_cols = [c for c in df.columns if c.startswith("param_")]
    if not param_cols:
        console.print("  [yellow]![/yellow]  sweep: no param columns found")
        return
    param_name = param_cols[0].replace("param_", "")
    wide: dict = {}
    for _, r in df.iterrows():
        val  = r.get(f"param_{param_name}")
        key  = (r["dataset"], val)
        if key not in wide:
            wide[key] = {"Dataset": r["dataset"], param_name: val}
        wide[key][f"Time_{r['algorithm']}"]  = r["time"]
        wide[key][f"Ratio_{r['algorithm']}"] = r["ratio"]
    wide_df = pd.DataFrame(list(wide.values()))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        wide_df.to_csv(f.name, index=False)
    out = str(output_dir / "parameter_plot.pdf")
    P.plot_parameter_analysis(f.name, param_name, out)
    _ok(out)


def _plot_update_cost(df, output_dir, p, logger) -> None:
    import pandas as pd
    from scripts.config import ALGORITHMS
    algos = df["algorithm"].unique().tolist()
    wide: dict = {}
    for _, r in df.iterrows():
        key = (r["dataset"], r["checkpoint"])
        if key not in wide:
            wide[key] = {"Dataset": r["dataset"], "Checkpoint": r["checkpoint"]}
        wide[key][f"Time_{r['algorithm']}"]  = r["time"]
        wide[key][f"Ratio_{r['algorithm']}"] = r["ratio"]
    wide_df = pd.DataFrame(list(wide.values()))
    algo_types = {a: ALGORITHMS.get(a, {}).get("type", "mosso") for a in algos}
    out = str(output_dir / "update_cost_plot.pdf")
    p.plot_update_cost(wide_df, algos, algo_types, out)
    _ok(out)

def _make_logger():
    logger = logging.getLogger("analyze")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger

def main() -> None:
    _interactive(_BENCHMARK_DIR)


if __name__ == "__main__":
    main()