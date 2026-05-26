import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import optuna
from optuna.exceptions import OptunaError
from optuna.importance import PedAnovaImportanceEvaluator

from scripts.analysis.plotters.base_plotter import Plotter, register

@register
class ParamImportancePlotter(Plotter):
    plotter_id = "param_importance"
    description = "Optuna hyperparameter importance visualization"

    def __init__(self):
        super().__init__()
        self.generates_plots = True

    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        self.set_chart_theme()

        # Analyze.py provides the database path directly in options
        db_path = options.get("db_path")

        # Fallback to scanning parent directories if not provided in options
        if not db_path or not Path(db_path).exists():
            for parent in out_dir.parents:
                potential_db = parent / "optuna_study.db"
                if potential_db.exists():
                    db_path = potential_db
                    break

        if not db_path or not Path(db_path).exists():
            print(f"[bold red]Database not found. Cannot compute parameter importance.[/bold red]")
            return []

        storage_url = f"sqlite:///{db_path}"
        generated_files = []

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for algo_name in algos:
            try:
                study = optuna.load_study(study_name=algo_name, storage=storage_url)

                # Importance evaluation requires enough completed trials
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if len(completed_trials) < 2:
                    print(f"Skipping {algo_name}: Not enough completed trials for importance evaluation.")
                    continue

                # 1. Evaluate importance for Objective 0 using PED-ANOVA
                try:
                    importances_obj_0 = optuna.importance.get_param_importances(
                        study,
                        target=lambda t: t.values[0],
                        evaluator=PedAnovaImportanceEvaluator()
                    )
                    if importances_obj_0:
                        out_path = self._plot_importances(importances_obj_0, algo_name, "Objective 0", out_dir)
                        if out_path: generated_files.append(out_path)
                except Exception as e:
                    print(f"Could not calculate Objective 0 importance for {algo_name}: {e}")

                # 2. Evaluate importance for Objective 1 using PED-ANOVA
                try:
                    importances_obj_1 = optuna.importance.get_param_importances(
                        study,
                        target=lambda t: t.values[1],
                        evaluator=PedAnovaImportanceEvaluator()
                    )
                    if importances_obj_1:
                        out_path = self._plot_importances(importances_obj_1, algo_name, "Objective 1", out_dir)
                        if out_path: generated_files.append(out_path)
                except Exception as e:
                    print(f"Could not calculate Objective 1 importance for {algo_name}: {e}")

            except OptunaError as e:
                print(f"Error loading study for {algo_name}: {e}")
                continue

        return generated_files

    def _plot_importances(self, importances: dict, algo: str, metric: str, out_dir: Path) -> Path | None:
        if not importances:
            return None

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Sort importances ascending for a horizontal bar chart
        items = list(importances.items())
        items.sort(key=lambda x: x[1])

        keys = [x[0] for x in items]
        vals = [x[1] for x in items]

        # Match standard colors used in other plots
        style = self.get_algo_style(algo)
        color = style["color"]

        ax.barh(keys, vals, color=color, alpha=0.85, edgecolor='black', linewidth=1.2)

        ax.set_xlabel("Importance", fontsize=14, style='italic')
        ax.set_ylabel("Hyperparameter", fontsize=14, style='italic')
        ax.set_title(f"Parameter Importance for {algo}\n(Objective: {metric})", fontsize=12, pad=15)

        # Grid lines behind bars for readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(max(vals) * 1.1, 0.05)) # Give a 10% breathing room on the right

        fig.tight_layout()

        metric_slug = metric.lower().replace(' ', '_')
        out_path = out_dir / f"param_importance_{algo}_{metric_slug}.png"
        fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return out_path