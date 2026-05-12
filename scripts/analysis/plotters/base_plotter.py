import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from scripts.config import PARAM_CONFIG

PLOTTERS: dict[str, type['Plotter']] = {}

class Plotter(ABC):
    plotter_id: str = ""
    description: str = ""

    @staticmethod
    def set_chart_theme():
        # Grayscale colors for fallback
        colors = ["#FFFFFF", "#DDDDDD", "#888888", "#000000"]

        sns.set_theme(style="ticks", rc={
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "serif"],
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.fontsize": 12
        })
        sns.set_palette(sns.color_palette(colors))

    def process(self, df: pd.DataFrame, algos: list[str], out_dir: Path, options: dict) -> list[Path]:
        opts = options or {}
        datasets = opts.get("datasets")
        aggregate = opts.get("aggregate", "per_dataset")

        sub = df[df["algorithm"].isin(algos)].copy()
        if datasets:
            sub = sub[sub["dataset"].isin(datasets)]
        if sub.empty:
            raise ValueError("No rows to analyze after filtering.")

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        group_cols = ["dataset", "algorithm"]
        optional_group_cols = ["change_ratio", "param", "trial", "run", "edges_processed", "edges_evaluated",
                               "power_of_2", "is_streaming"]
        for col in optional_group_cols:
            if col in sub.columns:
                group_cols.append(col)

        numeric_cols = []
        possible_metrics = ["time", "ratio", "time_micros", "accumulated_time_sec"]
        for col in possible_metrics:
            if col in sub.columns:
                numeric_cols.append(col)

        # Handle algorithm parameters from config
        param_keys = [k for k in PARAM_CONFIG.keys() if k in sub.columns]
        for p in param_keys:
            sub[p] = pd.to_numeric(sub[p], errors='coerce')
            if p not in group_cols:
                numeric_cols.append(p)

        # Ensure we don't try to calculate the mean of a column we are grouping by
        numeric_cols = [c for c in numeric_cols if c not in group_cols]

        if numeric_cols:
            grouped_runs = sub.groupby(group_cols, as_index=False)[numeric_cols].mean()
        else:
            grouped_runs = sub.drop_duplicates(subset=group_cols)

        grouped_runs["algorithm"] = pd.Categorical(grouped_runs["algorithm"], categories=algos, ordered=True)

        if aggregate == "average":
            avg_cols = [c for c in group_cols if c != "dataset"]
            if numeric_cols:
                final_agg = grouped_runs.groupby(avg_cols, as_index=False)[numeric_cols].mean().sort_values(avg_cols)
            else:
                final_agg = grouped_runs.sort_values(avg_cols)
            context = "average"
        else:
            final_agg = grouped_runs.sort_values(group_cols)
            context = "combined"

        return self.generate_artifacts(final_agg, algos, context, out_dir, ts, opts)

    @abstractmethod
    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        ...

def register(cls: type['Plotter']) -> type['Plotter']:
    if cls.plotter_id:
        PLOTTERS[cls.plotter_id] = cls
    return cls

def get_plotter(plotter_id: str) -> type[Plotter] | None:
    return PLOTTERS.get(plotter_id)