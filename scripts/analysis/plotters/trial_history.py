import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from matplotlib.lines import Line2D
import optuna

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class OptunaHistoryPlotter(Plotter):
    plotter_id = "trial_history"
    description = "Optuna trial history tracking convergence of objective values"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()
        generated_files = []
        console = Console()

        db_path = options.get("db_path")
        if not db_path or not Path(db_path).exists():
            console.print(f"[bold red]Optuna database not found at {db_path}. Cannot generate history plots.[/bold red]")
            return generated_files

        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            storage_url = f"sqlite:///{db_path}"
        except ImportError:
            console.print("[bold red]Optuna library not installed.[/bold red]")
            return generated_files

        datasets = data["dataset"].unique() if "dataset" in data.columns else [context]
        time_label = options.get("time_label", "time (seconds)").lower()

        for ds in datasets:
            for algo_name in algos:
                try:
                    study = optuna.load_study(study_name=algo_name, storage=storage_url)

                    trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
                    if not trials:
                        continue

                    trial_numbers = []
                    times = []
                    ratios = []

                    for t in trials:
                        trial_numbers.append(t.number)
                        times.append(t.values[0])
                        ratios.append(t.values[1])

                    if not times:
                        continue

                    # --- Incorporate the global styling for this specific algorithm ---
                    style = self.get_algo_style(algo_name)
                    color_time = style["color"]
                    marker_time = style["marker"]

                    # Ensure the secondary axis (Ratio) contrasts well with the primary algorithm color
                    color_ratio = "#000000" if color_time.lower() != "#000000" else "#E69F00"
                    marker_ratio = "x"

                    time_series = pd.Series(times)
                    ratio_series = pd.Series(ratios)

                    best_times = time_series.cummin()
                    best_ratios = ratio_series.cummin()

                    import seaborn as sns
                    fig, ax1 = plt.subplots(figsize=(7, 4))

                    t_min, t_max = best_times.min(), best_times.max()
                    t_margin = (t_max - t_min) * 0.05 if t_max > t_min else t_max * 0.05
                    if t_margin == 0: t_margin = 0.01

                    r_min, r_max = best_ratios.min(), best_ratios.max()
                    r_margin = (r_max - r_min) * 0.05 if r_max > r_min else r_max * 0.05
                    if r_margin == 0: r_margin = 0.01

                    ax1.set_ylim(max(0, t_min - t_margin), t_max + t_margin)

                    # 1. Plot Execution Time (Left Y-Axis)
                    ax1.set_xlabel("trial number", fontsize=12, style='italic')
                    ax1.set_ylabel(time_label, fontsize=12, style='italic', color=color_time)

                    ax1.scatter(trial_numbers, times, color=color_time, alpha=0.25, s=20, marker=marker_time, label="_nolegend_", zorder=2)
                    ax1.plot(trial_numbers, best_times, color=color_time, linewidth=2, linestyle="-", label="Best Time", zorder=3)

                    ax1.tick_params(axis='y', labelcolor=color_time)
                    ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)

                    # 2. Plot Compression Ratio (Right Y-Axis)
                    ax2 = ax1.twinx()

                    ax2.set_ylim(max(0, r_min - r_margin), r_max + r_margin)

                    ax2.set_ylabel("relative size", fontsize=12, style='italic', color=color_ratio)

                    ax2.scatter(trial_numbers, ratios, color=color_ratio, alpha=0.25, s=20, marker=marker_ratio, label="_nolegend_", zorder=2)
                    ax2.plot(trial_numbers, best_ratios, color=color_ratio, linewidth=2, linestyle="--", label="Best Ratio", zorder=3)

                    ax2.tick_params(axis='y', labelcolor=color_ratio)

                    sns.despine(ax=ax1, top=True, right=True)
                    sns.despine(ax=ax2, top=True, right=False)

                    eval_window = max(5, int(len(trial_numbers) * 0.15))
                    time_converged = best_times.iloc[-1] == best_times.iloc[-eval_window]
                    ratio_converged = best_ratios.iloc[-1] == best_ratios.iloc[-eval_window]

                    title_suffix = " (Converged)" if (time_converged and ratio_converged) else ""
                    plt.title(f"{algo_name} on {ds}{title_suffix}", fontsize=12, style='italic')

                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)

                    fig.tight_layout()

                    out_path = out_dir / f"optuna_history_{ds}_{algo_name}.png"
                    pdf_path = out_dir / f"optuna_history_{ds}_{algo_name}.pdf"
                    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
                    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                    plt.close(fig)

                    generated_files.extend([out_path, pdf_path])

                except Exception as e:
                    console.print(f"[dim]Could not generate history for {algo_name}: {e}[/dim]")

        return generated_files