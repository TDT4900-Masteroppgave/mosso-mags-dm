from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTTERS: dict[str, type['Plotter']] = {}

class Plotter(ABC):
    plot_id: str = ""
    description: str = ""

    def plot(self, df: pd.DataFrame, meta: dict, algos: list[str], out_dir: Path,
             options: dict | None = None) -> list[Path]:

        opts = options or {}
        datasets = opts.get("datasets")
        aggregate = opts.get("aggregate", "per_dataset")
        time_label = opts.get("time_label", "Time (seconds)")

        sub = df[df["algorithm"].isin(algos)].copy()
        if datasets:
            sub = sub[sub["dataset"].isin(datasets)]
        if sub.empty:
            raise ValueError("No rows to plot after filtering.")

        sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.15", "axes.linewidth": 1.25})

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = time_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "per").lower()

        out_paths = []

        numeric_cols = sub.select_dtypes(include='number').columns.tolist()
        group_cols = ["dataset", "algorithm"]

        # Dynamically preserve independent variables for time-series and sweeps
        if "change_ratio" in sub.columns: group_cols.append("change_ratio")
        if "param" in sub.columns: group_cols.append("param")
        if "trial" in sub.columns: group_cols.append("trial")

        grouped_runs = sub.groupby(group_cols, as_index=False)[numeric_cols].mean()

        # Convert algorithm to Categorical to guarantee consistent colors and sorting
        grouped_runs["algorithm"] = pd.Categorical(grouped_runs["algorithm"], categories=algos, ordered=True)

        if aggregate == "average":
            avg_cols = [c for c in group_cols if c != "dataset"]
            final_agg = grouped_runs.groupby(avg_cols, as_index=False)[numeric_cols].mean().sort_values(avg_cols)

            title_prefix = "Average Across Datasets"
            file_name = f"{self.plot_id}_average_{safe_label}_{ts}.png"

            fig = self.render_figure(final_agg, algos, title_prefix, time_label, opts)
            out_path = out_dir / file_name
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out_path)

        else:
            title_prefix = "Combined Datasets"
            file_name = f"{self.plot_id}_combined_{safe_label}_{ts}.png"

            # Sort to ensure predictable grouping in the plots
            final_agg = grouped_runs.sort_values(group_cols)

            fig = self.render_figure(final_agg, algos, title_prefix, time_label, opts)

            out_path = out_dir / file_name
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out_path)

        return out_paths

    @abstractmethod
    def render_figure(self, data: pd.DataFrame, algos: list[str], title_prefix: str, time_label: str, options: dict) -> plt.Figure:
        """Subclasses only need to implement this pure drawing logic."""
        ...

def register(cls: type['Plotter']) -> type['Plotter']:
    if cls.plot_id:
        PLOTTERS[cls.plot_id] = cls
    return cls

def get_plotter(plot_id: str) -> type[Plotter] | None:
    return PLOTTERS.get(plot_id)