"""Statistical helpers for cross-strategy comparison."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def knee_point(times: List[float], ratios: List[float]) -> int:
    """Index of the knee on a 2D Pareto front using max perpendicular distance
    from the line connecting the two endpoints. Inputs must already be on the
    Pareto front and sorted by `times` ascending.
    """
    n = len(times)
    if n == 0:
        raise ValueError("empty front")
    if n <= 2:
        return 0

    pts = np.column_stack([times, ratios]).astype(float)
    # Normalize so x and y are comparable.
    span = pts.max(axis=0) - pts.min(axis=0)
    span[span == 0] = 1.0
    norm = (pts - pts.min(axis=0)) / span

    p0, p1 = norm[0], norm[-1]
    line = p1 - p0
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return 0

    # Perpendicular distance from each point to the p0-p1 line.
    diffs = norm - p0
    cross = np.abs(diffs[:, 0] * line[1] - diffs[:, 1] * line[0])
    distances = cross / line_len
    return int(np.argmax(distances))


def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
    """Cliff's delta effect size with Romano et al. magnitude label.
    delta > 0 means x tends to be larger than y.
    """
    if not x or not y:
        return 0.0, "negligible"

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    # Pairwise comparison via broadcasting.
    diff = xa[:, None] - ya[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    delta = (greater - less) / (len(xa) * len(ya))

    a = abs(delta)
    if a < 0.147:
        magnitude = "negligible"
    elif a < 0.33:
        magnitude = "small"
    elif a < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return float(delta), magnitude


def paired_wilcoxon(
    strategy_medians: Dict[str, float],
    baseline_medians: Dict[str, float],
) -> Tuple[float, float]:
    """Cross-dataset paired Wilcoxon signed-rank. Returns (statistic, p_value)
    or (nan, nan) if fewer than one paired non-zero difference is available.
    Pairs by shared dataset key.
    """
    shared = sorted(set(strategy_medians) & set(baseline_medians))
    if len(shared) < 2:
        return float("nan"), float("nan")

    s = np.asarray([strategy_medians[k] for k in shared], dtype=float)
    b = np.asarray([baseline_medians[k] for k in shared], dtype=float)

    if np.all(s == b):
        return 0.0, 1.0

    try:
        # zero_method="wilcox" drops zero-differences (the standard).
        result = wilcoxon(s, b, zero_method="wilcox", alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def approx_hv2d(points: list[tuple[float, float]]) -> float:
    """Approximate 2D hypervolume dominated by a set of points (minimization)."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[0])
    ref_x = max(p[0] for p in pts) * 1.1
    ref_y = max(p[1] for p in pts) * 1.1
    hv = 0.0
    prev_y = ref_y
    prev_x = 0.0
    min_y = ref_y
    for x, y in pts:
        if y < min_y:
            hv += (x - prev_x) * (prev_y - y)
            prev_x = x
            prev_y = y
            min_y = y
    hv += (ref_x - prev_x) * prev_y
    return hv


def interp_time_at_ratio(pareto: pd.DataFrame, target_ratio: float) -> Optional[float]:
    """Linear interpolation on a Pareto front sorted by Ratio.

    Returns the time at which the front achieves `target_ratio`.
    If target_ratio is outside the front's range the nearest endpoint is returned.
    """
    pf = pareto.sort_values("Ratio").reset_index(drop=True)
    if pf.empty:
        return None
    if target_ratio <= float(pf["Ratio"].iloc[0]):
        return float(pf["Time"].iloc[0])
    if target_ratio >= float(pf["Ratio"].iloc[-1]):
        return float(pf["Time"].iloc[-1])
    for i in range(len(pf) - 1):
        r0, r1 = float(pf["Ratio"].iloc[i]), float(pf["Ratio"].iloc[i + 1])
        if r0 <= target_ratio <= r1:
            t0, t1 = float(pf["Time"].iloc[i]), float(pf["Time"].iloc[i + 1])
            frac = (target_ratio - r0) / (r1 - r0) if r1 != r0 else 0.0
            return t0 + frac * (t1 - t0)
    return None


def holm_bonferroni(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, bool]]:
    """Holm-Bonferroni step-down correction.
    Returns {key: (adjusted_p, is_significant)}. NaN inputs pass through as
    (nan, False) and do not consume a comparison slot.
    """
    valid = {k: v for k, v in p_values.items() if not np.isnan(v)}
    out: Dict[str, Tuple[float, bool]] = {
        k: (float("nan"), False) for k in p_values if k not in valid
    }

    if not valid:
        return out

    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    running_max = 0.0
    for i, (key, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        # Enforce monotonic non-decreasing adjusted p-values.
        running_max = max(running_max, adjusted)
        out[key] = (running_max, running_max < alpha)
    return out
