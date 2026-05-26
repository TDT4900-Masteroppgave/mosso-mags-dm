import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from scripts.analysis.plotters.base_plotter import Plotter, register


def format_power_of_2(val, _) -> str:
    if val <= 0:
        return ""

    exponent = int(round(math.log2(val)))
    if not math.isclose(val, 2 ** exponent, rel_tol=1e-6):
        return ""
    return rf"$2^{{{exponent}}}$"


def is_power_of_two(value: float) -> bool:
    if value <= 0 or not float(value).is_integer():
        return False
    int_value = int(value)
    return int_value & (int_value - 1) == 0


def choose_largest_slices(values: pd.Series, count: int = 4) -> list[float]:
    unique_values = sorted(values.dropna().unique())
    power_values = [value for value in unique_values if is_power_of_two(float(value))]
    if len(power_values) >= count:
        return power_values[-count:]
    return unique_values[-count:]


def choose_power_ticks(values: pd.Series) -> list[int]:
    min_power = math.floor(math.log2(values.min()))
    max_power = math.ceil(math.log2(values.max()))
    return [2 ** p for p in range(min_power, max_power + 1)]


def summarize_curve(data: pd.DataFrame, x_col: str) -> pd.DataFrame:
    curve = (
        data
        .groupby(x_col, as_index=False)["accumulated_time_sec"]
        .mean()
        .sort_values(x_col)
    )
    return curve[(curve[x_col] > 0) & (curve["accumulated_time_sec"] > 0)]


def log_log_slope(curve: pd.DataFrame, x_col: str) -> float | None:
    if len(curve) < 2:
        return None

    x = curve[x_col].apply(math.log2)
    y = curve["accumulated_time_sec"].apply(math.log2)
    variance = x.var()
    if variance == 0:
        return None
    return float(y.cov(x) / variance)


def set_shared_log_axes(ax, x_ticks: list[int], y_ticks: list[int]) -> None:
    x_min, x_max = min(x_ticks), max(x_ticks)
    y_min, y_max = min(y_ticks), max(y_ticks)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(2 ** (math.log2(x_min) - 0.15), 2 ** (math.log2(x_max) + 0.15))
    ax.set_ylim(2 ** (math.log2(y_min) - 0.15), 2 ** (math.log2(y_max) + 0.15))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_power_of_2))
    ax.yaxis.set_major_formatter(FuncFormatter(format_power_of_2))
    ax.minorticks_off()


@register
class LineChartScalabilityPlotter(Plotter):
    plotter_id = "line_chart_scalability"
    description = "Line chart for scalability"

    def generate_artifacts(
        self,
        data: pd.DataFrame,
        algos: list[str],
        context: str,
        out_dir: Path,
        options: dict,
    ) -> list[Path]:
        self.set_chart_theme()

        x_col = "changes_evaluated" if "changes_evaluated" in data.columns else "edges_evaluated"
        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
        data["accumulated_time_sec"] = pd.to_numeric(data["accumulated_time_sec"], errors="coerce")

        generated_files = []

        for ds in data["dataset"].unique():
            ds_data = data[data["dataset"] == ds].dropna(subset=[x_col, "accumulated_time_sec"])
            if ds_data.empty:
                continue

            selected_slices = choose_largest_slices(ds_data[x_col], count=4)
            ds_data = ds_data[ds_data[x_col].isin(selected_slices)]
            if ds_data.empty:
                continue

            ds_algos = [algo for algo in algos if algo in ds_data["algorithm"].values]
            if not ds_algos:
                continue

            x_ticks = [int(v) for v in selected_slices]
            y_ticks = choose_power_ticks(ds_data["accumulated_time_sec"])

            fig, ax = plt.subplots(figsize=(6, 3.5))
            plotted_algos = 0
            linear_baselines = []

            for algo in ds_algos:
                algo_data = ds_data[ds_data["algorithm"] == algo]
                curve = summarize_curve(algo_data, x_col)
                if curve.empty:
                    continue

                style = self.get_algo_style(algo)
                color = style["color"]
                marker = style["marker"]
                slope = log_log_slope(curve, x_col)
                label = f"{algo} (slope={slope:.2f})" if slope is not None else algo

                ax.plot(
                    curve[x_col],
                    curve["accumulated_time_sec"],
                    color=color,
                    marker=marker,
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    linewidth=1.8,
                    alpha=0.95,
                    label=label,
                    zorder=3,
                )
                plotted_algos += 1
                linear_baselines.append((algo, color, curve))

            if plotted_algos == 0:
                plt.close(fig)
                continue

            for baseline_idx, (algo, color, curve) in enumerate(linear_baselines):
                reference_x = [float(x) for x in curve[x_col]]
                reference_x0 = float(curve[x_col].iloc[0])
                reference_y0 = float(curve["accumulated_time_sec"].iloc[0])
                reference_y = [reference_y0 * (x / reference_x0) for x in reference_x]
                label = "Linear baseline (slope=1.00)" if baseline_idx == 0 else "_nolegend_"
                ax.plot(
                    reference_x,
                    reference_y,
                    color=color,
                    linestyle=(0, (1, 3)),
                    linewidth=1.4,
                    alpha=0.75,
                    label=label,
                    zorder=2,
                )

            set_shared_log_axes(ax, x_ticks, y_ticks)

            ax.set_xlabel("Number of Changes", fontsize=14, style="italic", labelpad=8)
            ax.set_ylabel("Accumulated Time (sec)", fontsize=14, style="italic", labelpad=10)

            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.4)

            ax.legend(
                title="",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(3, plotted_algos + 1),
                handlelength=1.8,
                handletextpad=0.5,
                columnspacing=1.2,
                frameon=False,
                fontsize=9,
            )

            fig.subplots_adjust(left=0.20, right=0.98, bottom=0.32, top=0.96)

            png_path = out_dir / f"scalability_line_{ds}.png"
            fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            generated_files.append(png_path)

        return generated_files
