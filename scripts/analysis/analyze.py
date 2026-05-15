"""Interactive analysis CLI.

Usage: ./run.sh analyze
"""
import sys
from pathlib import Path

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel

from scripts.analysis.plotters import Plotter
from scripts.analysis.plotters.base_plotter import get_plotter
from scripts.analysis.sessions import SessionInfo, load_session, scan_sessions
from scripts.config import EXPERIMENT_DIR

console = Console(highlight=False)

EXPERIMENT_PLOTS = {
    "benchmark": ["bar_chart_time", "bar_chart_time_log", "bar_chart_compression"],
    "processing_speed": ["bar_chart_time_micros"],
    "compression_checkpoints": ["line_chart_compression"],
    "sweep": ["line_chart_sweep"],
    "scalability": ["line_chart_scalability"],
    "bayesian": ["pareto_front", "trial_history"],
}

_STYLE = questionary.Style([
    ("qmark", "fg:#00afff bold"),
    ("question", "bold"),
    ("answer", "fg:#5fd7af bold"),
    ("pointer", "fg:#00afff bold"),
    ("highlighted", "fg:#00afff bold"),
    ("selected", "fg:#5fd7af"),
    ("instruction", "fg:#808080 italic"),
])


def _fmt_ts(ts: str) -> str:
    if len(ts) >= 15:
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    return ts


def _pick_type(grouped: dict[str, list[SessionInfo]]) -> str | None:
    choices = [
        Choice(title=f"{t}  ({len(grouped[t])} session{'s' if len(grouped[t]) != 1 else ''})", value=t)
        for t in sorted(grouped.keys())
    ]
    return questionary.select(
        "Experiment type:",
        choices=choices,
        style=_STYLE,
        instruction="(↑/↓ arrows, enter to confirm)",
    ).ask()


def _pick_session(sessions: list[SessionInfo]) -> SessionInfo | None:
    def label(s: SessionInfo) -> str:
        algos = ", ".join(s.algorithms) if s.algorithms else "?"
        ds_str = f"[{', '.join(s.datasets)}]" if s.datasets else "[?]"
        return f"{_fmt_ts(s.timestamp)}   algos: {algos}   datasets: {ds_str}"

    choices = [
        Choice(title=label(s), value=s)
        for s in sessions if s.has_db
    ]

    if not choices:
        console.print("[yellow]No analyzable sessions (with databases) found in this category.[/yellow]")
        return None

    return questionary.select(
        "Session:",
        choices=choices,
        style=_STYLE,
        instruction="(↑/↓ arrows, enter to confirm)",
    ).ask()


def _pick_algos(available: list[str]) -> list[str] | None:
    if len(available) == 1:
        return available

    choices = [Choice(title=a, value=a, checked=True) for a in available]
    while True:
        result = questionary.checkbox(
            "Algorithms (space to toggle, enter to confirm):",
            choices=choices,
            style=_STYLE,
        ).ask()
        if result is None:
            return None
        if result:
            return result
        console.print("[yellow]Select at least one algorithm.[/yellow]")


def _pick_datasets(available: list[str]) -> list[str]:
    if len(available) == 1:
        return available

    choices = [Choice(title=d, value=d, checked=True) for d in available]
    while True:
        result = questionary.checkbox(
            "Datasets (space to toggle, enter to confirm):",
            choices=choices,
            style=_STYLE,
        ).ask()
        if result:
            return result
        console.print("[yellow]Select at least one dataset.[/yellow]")


def _pick_plot(session_type: str) -> type['Plotter'] | None:
    allowed_plot_ids = EXPERIMENT_PLOTS.get(session_type, [])

    if not allowed_plot_ids:
        console.print(f"[red]No plot types configured for experiment: {session_type}[/red]")
        return None

    choices = []
    for pid in allowed_plot_ids:
        plotter_cls = get_plotter(pid)
        if plotter_cls:
            choices.append(Choice(title=f"{plotter_cls.description}", value=plotter_cls))

    if not choices:
        console.print(f"[red]No registered plotters found for {session_type}.[/red]")
        return None

    if len(choices) == 1:
        return choices[0].value

    res = questionary.select(
        "Select plot type:",
        choices=choices,
        style=_STYLE
    ).ask()

    return res


def _confirm(msg: str, default: bool = False) -> bool:
    res = questionary.confirm(msg, default=default, style=_STYLE).ask()
    return bool(res)


def main() -> None:
    console.clear()
    console.rule("[bold]MOSSO Analysis[/bold]")
    console.print(f"[dim]Scanning {EXPERIMENT_DIR} ...[/dim]\n")

    grouped = scan_sessions(Path(EXPERIMENT_DIR))
    if not grouped:
        console.print("[red]No experiment sessions found.[/red]")
        sys.exit(1)

    btype = _pick_type(grouped)
    if not btype:
        return

    session = _pick_session(grouped[btype])
    if not session:
        return

    df, meta = load_session(session.path)
    if df.empty:
        console.print("[red]Session has empty results table.[/red]")
        return

    if "algorithm" not in df.columns:
        console.print("[red]Results df missing 'algorithm' column.[/red]")
        return

    available_algos: list[str] = [str(x) for x in sorted(df["algorithm"].dropna().unique())]
    available_datasets: list[str] = (
        [str(x) for x in sorted(df["dataset"].dropna().unique())]
        if "dataset" in df.columns else []
    )


    plot_df = df.copy()
    algos = _pick_algos(available_algos)
    if not algos:
        return

    time_label = "Time (seconds)"
    datasets = None

    if available_datasets:
        datasets = _pick_datasets(available_datasets)
        if not datasets:
            return

    plotter_cls = _pick_plot(session.type)
    if not plotter_cls:
        console.print("[yellow]No plot to generate selected.[/yellow]")
        return

    # Print a beautiful summary panel right before generating
    summary_text = (
        f"[bold]Session:[/bold] {_fmt_ts(session.timestamp)}\n"
        f"[bold]Algorithms:[/bold] {', '.join(algos)}\n"
        f"[bold]Datasets:[/bold] {', '.join(datasets) if datasets else 'All'}\n"
        f"[bold]Plot Type:[/bold] {plotter_cls.description}"
    )
    console.print(Panel(summary_text, title="Summary", border_style="cyan"))

    out_dir = session.path / "analysis"

    options = {
        "datasets": datasets,
        "time_label": time_label,
        "db_path": session.path / "optuna_study.db",
    }

    try:
        out_paths = plotter_cls().process(plot_df, algos, out_dir, options)
        for path in out_paths:
            console.print(f"[green]✓[/green] Saved: [bold]{path}[/bold]")
    except Exception as e:
        console.print(f"[red]✗ Analysis failed:[/red] {e}\n")

if __name__ == "__main__":
    main()