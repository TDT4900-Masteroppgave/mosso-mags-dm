import pandas as pd
import seaborn as sns
from pathlib import Path
from abc import ABC, abstractmethod
from scripts.config import PARAM_CONFIG

PLOTTERS: dict[str, type['Plotter']] = {}

class Plotter(ABC):
    plotter_id: str = ""
    description: str = ""
    dataset_order: list[str] = ["PR", "CA", "BK", "EN", "EA", "FB", "SL", "EU", "HW", "UK", "DB", "YT", "SK", "LJ"]

    ALGO_STYLES = {
        "local": {"color": "#000000", "marker": "s"},          # Black
        "kdd20-mosso": {"color": "#0072B2", "marker": "o"},    # Dark Blue (Okabe-Ito)
        "sm": {"color": "#D55E00", "marker": "^"},             # Vermillion (Okabe-Ito)
        "top-b": {"color": "#009E73", "marker": "D"},          # Bluish Green (Okabe-Ito)
        "sm_thr": {"color": "#E69F00", "marker": "v"},         # Orange (Okabe-Ito)
        "cap": {"color": "#CC79A7", "marker": "p"},            # Reddish Purple (Okabe-Ito)
        "ds": {"color": "#56B4E9", "marker": "*"},             # Sky Blue (Okabe-Ito)
        "ds_thr": {"color": "#E6C122", "marker": "h"},         # Darkened Yellow (Accessibility contrast)
        "ds_sm_thr": {"color": "#882255", "marker": "X"},      # Wine (Paul Tol)
        "mags": {"color": "#332288", "marker": "<"},           # Deep Indigo (Paul Tol)
        "para_mags": {"color": "#88CCEE", "marker": ">"},      # Cyan (Paul Tol)
        "mags_dm": {"color": "#117733", "marker": "d"},        # Pine Green (Paul Tol)
        "para_mags_dm": {"color": "#999933", "marker": "P"}    # Olive (Paul Tol)
    }

    THEME_COLORS = [
        "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
        "#56B4E9", "#332288", "#117733", "#882255", "#999933",
        "#88CCEE", "#E6C122", "#000000"
    ]

    THEME_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<']

    def get_algo_style(self, algo_name: str) -> dict:
        if algo_name in self.ALGO_STYLES:
            return self.ALGO_STYLES[algo_name]

        idx = sum(ord(c) for c in algo_name)
        return {
            "color": self.THEME_COLORS[idx % len(self.THEME_COLORS)],
            "marker": self.THEME_MARKERS[idx % len(self.THEME_MARKERS)]
        }

    def get_dataset_order(self, data: pd.DataFrame) -> list[str]:
        current_datasets = data["dataset"].unique()
        ordered_display = [ds for ds in self.dataset_order if ds in current_datasets]

        remaining = [ds for ds in current_datasets if ds not in self.dataset_order]
        ordered_display.extend(remaining)
        return ordered_display

    def set_chart_theme(self):
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
        sns.set_palette(sns.color_palette(self.THEME_COLORS))

    def process(self, df: pd.DataFrame, algos: list[str], out_dir: Path, options: dict) -> list[Path]:
        opts = options or {}
        datasets = opts.get("datasets")

        sub = df[df["algorithm"].isin(algos)].copy()
        if datasets:
            sub = sub[sub["dataset"].isin(datasets)]
        if sub.empty:
            raise ValueError("No rows to analyze after filtering.")

        out_dir.mkdir(parents=True, exist_ok=True)

        group_cols = ["dataset", "algorithm"]
        optional_group_cols = ["change_ratio", "param", "trial", "run", "edges_processed", "edges_evaluated",
                               "power_of_2", "is_streaming", "param_name"]
        for col in optional_group_cols:
            if col in sub.columns:
                group_cols.append(col)

        numeric_cols = []
        possible_metrics = ["time", "ratio", "time_micros", "accumulated_time_sec"]
        for col in possible_metrics:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors='coerce')
                numeric_cols.append(col)

        param_keys = [k for k in PARAM_CONFIG.keys() if k in sub.columns]
        for p in param_keys:
            sub[p] = pd.to_numeric(sub[p], errors='coerce')
            if p not in group_cols:
                numeric_cols.append(p)

        numeric_cols = [c for c in numeric_cols if c not in group_cols]

        if numeric_cols:
            grouped_runs = sub.groupby(group_cols, as_index=False)[numeric_cols].mean()
        else:
            grouped_runs = sub.drop_duplicates(subset=group_cols)

        grouped_runs["algorithm"] = pd.Categorical(grouped_runs["algorithm"], categories=algos, ordered=True)

        final_agg = grouped_runs.sort_values(group_cols)
        context = "combined"

        return self.generate_artifacts(final_agg, algos, context, out_dir, opts)

    @abstractmethod
    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, options: dict) -> list[Path]:
        ...

def register(cls: type['Plotter']) -> type['Plotter']:
    if cls.plotter_id:
        PLOTTERS[cls.plotter_id] = cls
    return cls

def get_plotter(plotter_id: str) -> type[Plotter] | None:
    return PLOTTERS.get(plotter_id)