from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import pandas as pd
from scripts.config import PARAM_CONFIG # <--- ADD THIS IMPORT

ANALYZERS: dict[str, type['Plotter']] = {}

class Plotter(ABC):
    analyzer_id: str = ""
    description: str = ""

    def __init__(self):
        self.generates_plots = False
        self.generates_data = False

    def process(self, df: pd.DataFrame, meta: dict, algos: list[str], out_dir: Path, options: dict) -> list[Path]:
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

        # 1. Start with the core numeric metrics we always want to average
        numeric_cols = ["time", "ratio"]
        if "time_micros" in sub.columns: numeric_cols.append("time_micros")

        # 2. Extract hyperparameters from the dataframe if they exist
        # We add them to numeric_cols so their values are preserved/averaged during the groupby
        param_keys = [k for k in PARAM_CONFIG.keys() if k in sub.columns]
        for p in param_keys:
            # Ensure they are numeric so .mean() doesn't drop them
            sub[p] = pd.to_numeric(sub[p], errors='coerce')
            numeric_cols.append(p)

        group_cols = ["dataset", "algorithm"]
        if "change_ratio" in sub.columns: group_cols.append("change_ratio")
        if "param" in sub.columns: group_cols.append("param")
        if "trial" in sub.columns: group_cols.append("trial")

        # Now when we groupby, the parameters are included in the .mean() calculation
        grouped_runs = sub.groupby(group_cols, as_index=False)[numeric_cols].mean()
        grouped_runs["algorithm"] = pd.Categorical(grouped_runs["algorithm"], categories=algos, ordered=True)

        if aggregate == "average":
            avg_cols = [c for c in group_cols if c != "dataset"]
            final_agg = grouped_runs.groupby(avg_cols, as_index=False)[numeric_cols].mean().sort_values(avg_cols)
            context = "average"
        else:
            final_agg = grouped_runs.sort_values(group_cols)
            context = "combined"

        return self.generate_artifacts(final_agg, algos, context, out_dir, ts, opts)

    @abstractmethod
    def generate_artifacts(self, data: pd.DataFrame, algos: list[str], context: str, out_dir: Path, ts: str, options: dict) -> list[Path]:
        ...

def register(cls: type['Plotter']) -> type['Plotter']:
    if cls.analyzer_id:
        ANALYZERS[cls.analyzer_id] = cls
    return cls

def get_analyzer(analyzer_id: str) -> type[Plotter] | None:
    return ANALYZERS.get(analyzer_id)