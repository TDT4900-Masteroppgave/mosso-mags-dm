import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from scripts.config import PARAM_CONFIG, DATASETS_DIR

PLOTTERS: dict[str, type['Plotter']] = {}


@dataclass(frozen=True)
class FigureProfile:
    figsize: tuple[float, float]
    font_size: float
    label_size: float
    tick_size: float
    legend_size: float
    line_width: float = 1.4
    marker_size: float = 5.0
    marker_edge_width: float = 0.9
    title_size: float | None = None
    extra_rc: dict | None = None

    def style_kwargs(self) -> dict:
        return {
            "font_size": self.font_size,
            "label_size": self.label_size,
            "tick_size": self.tick_size,
            "legend_size": self.legend_size,
            "line_width": self.line_width,
            "marker_size": self.marker_size,
            "marker_edge_width": self.marker_edge_width,
            "title_size": self.title_size,
            "extra_rc": self.extra_rc,
        }


class Plotter(ABC):
    plotter_id: str = ""
    description: str = ""
    dataset_order: list[str] = ["CA", "PR", "BK", "EN", "EA", "SL", "FB", "DB", "AM", "CN", "YT", "SK", "IN", "EU", "LJ", "HW", "UK"]

    _metadata_cache = None

    ALGO_STYLES = {
        "local": {"color": "#000000", "marker": "s"},
        "MoSSo": {"color": "#0072B2", "marker": "o"},
        "kdd20-mosso": {"color": "#0072B2", "marker": "o"},
        "sm": {"color": "#D55E00", "marker": "^"},
        "top-b": {"color": "#009E73", "marker": "D"},
        "sm_thr": {"color": "#E69F00", "marker": "v"},
        "cap": {"color": "#CC79A7", "marker": "p"},
        "ds": {"color": "#56B4E9", "marker": "*"},
        "ds_thr": {"color": "#E6C122", "marker": "h"},
        "ds_sm_thr": {"color": "#882255", "marker": "X"},
        "mags": {"color": "#002050", "marker": "o"},
        "para_mags": {"color": "#88CCEE", "marker": ">"},
        "mags_dm": {"color": "#117733", "marker": "d"},
        "para_mags_dm": {"color": "#999933", "marker": "P"}
    }

    ALGORITHM_RENAMES = {
        "kdd20-mosso": "MoSSo",
    }

    BATCH_ALGORITHMS = {"mags", "mags_dm"}

    THEME_COLORS = [
        "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
        "#56B4E9", "#332288", "#117733", "#882255", "#999933",
        "#88CCEE", "#E6C122", "#000000"
    ]

    THEME_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<']

    def get_algorithm_renames(self, options: dict | None = None) -> dict[str, str]:
        return {
            **self.ALGORITHM_RENAMES,
            **((options or {}).get("algorithm_renames") or {}),
        }

    def normalize_algorithm_name(self, algo_name: str, options: dict | None = None) -> str:
        return self.get_algorithm_renames(options).get(str(algo_name), str(algo_name))

    def normalize_algorithms(self, algos: list[str], options: dict | None = None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for algo in algos:
            label = self.normalize_algorithm_name(algo, options)
            if label not in seen:
                normalized.append(label)
                seen.add(label)
        return normalized

    def normalize_algorithm_column(self, df: pd.DataFrame, options: dict | None = None) -> pd.DataFrame:
        if "algorithm" not in df.columns:
            return df

        out = df.copy()
        out["algorithm"] = out["algorithm"].map(
            lambda value: self.normalize_algorithm_name(value, options) if pd.notnull(value) else value
        )
        return out

    def get_algo_style(self, algo_name: str) -> dict:
        algo_name = self.normalize_algorithm_name(algo_name)
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
            "axes.edgecolor": "#2D3748",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,

            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],

            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,

            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })
        sns.set_palette(sns.color_palette(self.THEME_COLORS))

    def set_plot_style(
        self,
        *,
        font_size: float,
        label_size: float,
        tick_size: float,
        legend_size: float,
        line_width: float = 1.4,
        marker_size: float = 5.0,
        marker_edge_width: float = 0.9,
        title_size: float | None = None,
        extra_rc: dict | None = None,
    ) -> None:
        rc = {
            "font.size": font_size,
            "axes.labelsize": label_size,
            "axes.titlesize": title_size if title_size is not None else label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "lines.markeredgewidth": marker_edge_width,
        }
        if extra_rc:
            rc.update(extra_rc)
        plt.rcParams.update(rc)

    def get_figure_profile(
        self,
        profile: FigureProfile,
        options: dict | None = None,
        *,
        profile_name: str | None = None,
        **overrides,
    ) -> FigureProfile:
        configured_overrides = {}
        if profile_name is not None:
            configured_overrides = ((options or {}).get("figure_profiles") or {}).get(profile_name, {})
        merged_overrides = {**configured_overrides, **overrides}

        if "figure_size" in merged_overrides and "figsize" not in merged_overrides:
            merged_overrides["figsize"] = merged_overrides.pop("figure_size")

        if "figsize" in merged_overrides:
            merged_overrides["figsize"] = tuple(merged_overrides["figsize"])

        if "extra_rc" in merged_overrides and profile.extra_rc:
            merged_overrides["extra_rc"] = {**profile.extra_rc, **(merged_overrides["extra_rc"] or {})}

        return replace(profile, **merged_overrides)

    def use_figure_profile(
        self,
        profile: FigureProfile,
        options: dict | None = None,
        *,
        profile_name: str | None = None,
        **overrides,
    ) -> FigureProfile:
        self.set_chart_theme()
        profile = self.get_figure_profile(profile, options, profile_name=profile_name, **overrides)
        self.set_plot_style(**profile.style_kwargs())
        return profile

    def create_figure(self, profile: FigureProfile, *args, **kwargs):
        return plt.subplots(*args, figsize=profile.figsize, **kwargs)

    @staticmethod
    def style_major_minor_grid(ax, *, minor_axis: str = "both") -> None:
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", axis=minor_axis, linestyle=":", alpha=0.3)

    @staticmethod
    def despine(ax, *, top: bool = True, right: bool = True) -> None:
        sns.despine(ax=ax, top=top, right=right)

    def style_paper_bar_chart(self, ax, *, ylabel: str, algos: list[str], legend_y: float = 1.12) -> None:
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, style="italic", labelpad=12)
        ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)
        self.despine(ax, top=True, right=True)

        ax.tick_params(axis="x", which="major", direction="out", length=7, width=2.0, pad=8, top=False, bottom=True)
        ax.tick_params(axis="y", which="major", direction="out", length=7, width=2.0, pad=8, right=False, left=True)

        ax.legend(
            title="",
            bbox_to_anchor=(0.5, legend_y),
            loc="lower center",
            ncol=min(len(algos), 4),
            frameon=False,
            handlelength=1.6,
            handleheight=0.7,
            handletextpad=0.6,
            columnspacing=1.5,
            borderaxespad=0.0,
        )

    @staticmethod
    def add_centered_legend(ax, *, y: float, loc: str = "upper center", ncol: int = 4, **kwargs) -> None:
        ax.legend(
            title="",
            bbox_to_anchor=(0.5, y),
            loc=loc,
            ncol=ncol,
            frameon=False,
            **kwargs,
        )

    @staticmethod
    def cv(values) -> float:
        mean = np.mean(values)
        return (np.std(values, ddof=1) / mean * 100) if mean != 0 and len(values) > 1 else 0

    coefficient_of_variation = cv

    @staticmethod
    def format_percent(value) -> str:
        return f"{value:.2f}%" if pd.notnull(value) else "-"

    def process(self, df: pd.DataFrame, algos: list[str], out_dir: Path, options: dict) -> list[Path]:
        opts = options or {}
        datasets = opts.get("datasets")
        algos = self.normalize_algorithms(algos, opts)
        df = self.normalize_algorithm_column(df, opts)

        sub = df[df["algorithm"].isin(algos)].copy()

        if "is_streaming" in sub.columns:
            sub["is_streaming"] = sub["is_streaming"].astype(str)
            batch_algorithms = self.BATCH_ALGORITHMS | {
                self.normalize_algorithm_name(algo, opts) for algo in self.BATCH_ALGORITHMS
            }
            sub.loc[sub["algorithm"].isin(batch_algorithms), "is_streaming"] = "0"

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
