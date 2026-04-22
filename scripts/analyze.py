"""
Analysis CLI for benchmark sessions.

Interactive mode (no --session):  ./run.sh --type analyze
Non-interactive (scripted):        ./run.sh --type analyze --session <path> --plot bar
"""
import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import box

from scripts.db import read_results, get_metadata

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Plot catalogue
# ---------------------------------------------------------------------------

_PLOTS_BY_TYPE: dict[str, list[tuple[str, str]]] = {
    "compare":              [("bar",        "Bar chart — time & ratio per dataset"),
                             ("scalability","Log-log scalability vs |E|"),
                             ("variance",   "Per-run variance (requires runs > 1)")],
    "significance":         [("boxplots",   "Significance boxplots with stars"),
                             ("violins",    "Violin plots with jitter overlay")],
    "bayesian":             [("pareto",     "Pareto-front scatter (time vs ratio)"),
                             ("convergence","Hypervolume convergence over trials")],
    "lhs":                  [("pareto",     "Pareto-front scatter (time vs ratio)"),
                             ("heatmap",    "2-D parameter interaction heatmaps")],
    "sweep":                [("sweep",      "Parameter sensitivity line plots")],
    "incremental_vs_batch": [("update-cost","Update cost per checkpoint")],
}

_BENCHMARK_DIR = Path("output/benchmarks")


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

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

        # Baseline (only for significance / compare)
        baseline = None
        if any(t in types_in_use for t in ("significance", "compare")):
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

        # ── Generate ───────────────────────────────────────────────────────
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

        from scripts import plotting as P
        for plot_type in chosen_plots:
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


# ---------------------------------------------------------------------------
# Non-interactive (scripted) path
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze",
        description=(
            "Analyse benchmark sessions and generate plots.\n"
            "Run with no --session to enter interactive mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session", dest="sessions", action="append", metavar="PATH",
        help="Session directory (repeat for multiple). Omit to enter interactive mode.",
    )
    all_plot_keys = sorted({k for plots in _PLOTS_BY_TYPE.values() for k, _ in plots})
    parser.add_argument(
        "--plot", dest="plots", nargs="+", default=["all"], metavar="TYPE",
        help=f"Plot type(s). 'all' = all applicable. Choices: {', '.join(all_plot_keys)}",
    )
    parser.add_argument("--algorithm", dest="algorithms", nargs="+", metavar="ALGO")
    parser.add_argument("--dataset",   dest="datasets",   nargs="+", metavar="DS")
    parser.add_argument("--baseline",  metavar="ALGO")
    parser.add_argument("--output",    metavar="DIR")
    return parser.parse_args()


def _resolve_plots(requested: list[str], session_dirs: list[Path]) -> list[str]:
    if requested == ["all"]:
        types = set()
        for sdir in session_dirs:
            meta = get_metadata(sdir)
            types.add(meta.get("benchmark_type", ""))
        keys = []
        seen = set()
        for t in types:
            for k, _ in _PLOTS_BY_TYPE.get(t, []):
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        return keys
    return requested


def run_analyze(args: argparse.Namespace) -> None:
    session_dirs = [Path(s) for s in args.sessions]
    for sdir in session_dirs:
        if not sdir.exists():
            console.print(f"[red]Session not found:[/red] {sdir}")
            sys.exit(1)
        if not (sdir / "results.db").exists():
            console.print(f"[red]No results.db in[/red] {sdir}. Run the benchmark first.")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else session_dirs[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = _resolve_plots(args.plots, session_dirs)
    if not plots:
        console.print("[red]No plots resolved.[/red] Check --plot or benchmark type.")
        sys.exit(1)

    df = read_results(session_dirs, algorithms=args.algorithms, datasets=args.datasets)
    if df.empty:
        console.print("[red]No results found[/red] in the specified session(s).")
        sys.exit(1)

    console.print(f"Loaded [bold]{len(df)}[/bold] rows from {len(session_dirs)} session(s).")
    console.print(f"Algorithms : {sorted(df['algorithm'].unique())}")
    console.print(f"Datasets   : {sorted(df['dataset'].unique())}")
    console.rule("[bold]Generating[/bold]")

    from scripts import plotting as P
    logger = _make_logger()
    for plot_type in plots:
        try:
            _dispatch_plot(plot_type, df, session_dirs, output_dir, args, P, logger)
        except Exception as e:
            console.print(f"  [red]✗[/red]  {plot_type}: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

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
    else:
        console.print(f"  [yellow]?[/yellow]  Unknown plot type: {plot_type}")


# ---------------------------------------------------------------------------
# Per-plot helpers
# ---------------------------------------------------------------------------

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
        if r.get("memory_mb") is not None:
            rows[ds][f"Memory_avg_{algo}"] = r["memory_mb"]
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
    raw = df.rename(columns={"algorithm": "Algorithm", "dataset": "Dataset",
                              "rep": "Rep", "time": "Time", "ratio": "Ratio"})
    baseline = getattr(args, "baseline", None)
    summary = _significance_summary(raw, baseline)
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
    raw = df.rename(columns={"algorithm": "Algorithm", "dataset": "Dataset",
                              "rep": "Rep", "time": "Time", "ratio": "Ratio"})
    baseline = getattr(args, "baseline", None)
    summary = _significance_summary(raw, baseline)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as rf:
        raw.to_csv(rf.name, index=False)
        raw_path = rf.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as sf:
        summary.to_csv(sf.name, index=False)
        sum_path = sf.name
    out = str(output_dir / "significance_violins.pdf")
    P.plot_significance_violins(raw_path, sum_path, out, baseline)
    _ok(out)


def _significance_summary(raw, baseline: str | None):
    try:
        import argparse as _ap
        from scripts.benchmarks.significance_test import SignificanceTestBenchmark
        dummy = object.__new__(SignificanceTestBenchmark)
        dummy.args = _ap.Namespace(baseline_algo=baseline, alpha=0.05)
        dummy.knee_log = []
        dummy.bo_results = {}
        dummy.active_algos = {}
        return dummy._build_summary(raw)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


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


def _plot_update_cost(df, output_dir, P, logger) -> None:
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
    P.plot_update_cost(wide_df, algos, algo_types, out)
    _ok(out)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_logger():
    logger = logging.getLogger("analyze")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if not args.sessions:
        _interactive(_BENCHMARK_DIR)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()