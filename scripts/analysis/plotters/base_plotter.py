import json
import pandas as pd
import seaborn as sns
from pathlib import Path
from abc import ABC, abstractmethod
from scripts.config import PARAM_CONFIG, DATASETS_DIR

PLOTTERS: dict[str, type['Plotter']] = {}

class Plotter(ABC):
    plotter_id: str = ""
    description: str = ""
    dataset_order: list[str] = ["CA", "PR", "BK", "EN", "EA", "SL", "FB", "DB", "AM", "CN", "YT", "SK", "IN", "EU", "LJ", "HW", "UK"]

    _metadata_cache = None

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
        "mags": {"color": "#002050", "marker": "o"},           # Deep Indigo (Paul Tol)
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

    def get_dataset_metadata(self) -> dict:
        """Retrieves and caches dataset metadata from preprocessing_log.json."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._metadata_cache = {}
        log_file = DATASETS_DIR / "preprocessing_log.json"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                prep_log = json.load(f)
                for ds_name, info in prep_log.items():
                    self._metadata_cache[ds_name] = info.get("metadata", {})
        return self._metadata_cache

    def set_chart_theme(self):
        sns.set_theme(style="ticks", rc={
            "axes.edgecolor": "#2D3748",  # Soft dark grey
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#E2E8F0",      # Soft light grey gridlines
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",     # Direct ticks outward
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,         # Left/bottom only
            "legend.frameon": False,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,           # Embed fonts as vector Type 42 in PDFs
            "ps.fonttype": 42
        })
        sns.set_palette(sns.color_palette(self.THEME_COLORS))

    def process(self, df: pd.DataFrame, algos: list[str], out_dir: Path, options: dict) -> list[Path]:
        opts = options or {}
        datasets = opts.get("datasets")

        sub = df[df["algorithm"].isin(algos)].copy()

        # Force mags and mags_dm to be batch
        if "is_streaming" in sub.columns:
            sub["is_streaming"] = sub["is_streaming"].astype(str)
            sub.loc[sub["algorithm"].isin(["mags", "mags_dm"]), "is_streaming"] = "0"

        if datasets:
            sub = sub[sub["dataset"].isin(datasets)]
        if sub.empty:
            raise ValueError("No rows to analyze after filtering.")

        out_dir.mkdir(parents=True, exist_ok=True)

        group_cols = ["dataset", "algorithm"]
        optional_group_cols = ["change_ratio", "param", "trial", "run", "edges_processed", "changes_evaluated",
                               "edges_evaluated", "power_of_2", "is_streaming", "param_name"]
        for k in PARAM_CONFIG.keys():
            if k not in optional_group_cols:
                optional_group_cols.append(k)

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
            grouped_runs = sub.groupby(group_cols, dropna=False, as_index=False)[numeric_cols].mean()
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