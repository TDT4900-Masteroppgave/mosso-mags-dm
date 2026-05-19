import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts import db


@dataclass
class SessionInfo:
    path: Path
    type: str
    timestamp: str
    algorithms: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    has_db: bool = True


def _read_meta(db_path: Path) -> dict:
    try:
        conn = sqlite3.connect(str(db_path))
        meta = db.read_metadata(conn)
        conn.close()
        return meta
    except sqlite3.DatabaseError:
        return {}


def _extract_algos(meta: dict) -> list[str]:
    algos = meta.get("algorithms", {})
    return list(algos.keys())


def _extract_datasets(meta: dict) -> list[str]:
    ds = meta.get("datasets", [])
    if isinstance(ds, list):
        return [d.get("short_name", d.get("filename", "?")) for d in ds if isinstance(d, dict)]
    return []


def scan_sessions(experiment_dir: Path) -> dict[str, list[SessionInfo]]:
    """Scan output/experiments/<type>/run_*/results.db. Group by benchmark_type from metadata."""
    grouped: dict[str, list[SessionInfo]] = {}
    if not experiment_dir.exists():
        return grouped

    for type_dir in sorted(experiment_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        for run_dir in sorted(type_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            db_path = run_dir / "results.db"
            has_db = db_path.exists()
            meta = _read_meta(db_path) if has_db else {}
            btype = meta.get("benchmark_type") or type_dir.name
            info = SessionInfo(
                path=run_dir,
                type=btype,
                timestamp=run_dir.name.replace("run_", ""),
                algorithms=_extract_algos(meta),
                datasets=_extract_datasets(meta),
                has_db=has_db,
            )
            grouped.setdefault(btype, []).append(info)
    return grouped


def load_session(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load results df + parsed metadata for a session."""
    conn = sqlite3.connect(str(run_dir / "results.db"))
    try:
        df = db.read_results(conn)
        meta = db.read_metadata(conn)
    finally:
        conn.close()
    return df, meta


def _merge_algorithm_metadata(metas: list[dict]) -> dict:
    merged = {}
    for meta in metas:
        algos = meta.get("algorithms", {})
        if not isinstance(algos, dict):
            continue
        for name, algo_meta in algos.items():
            merged.setdefault(name, algo_meta)
    return merged


def _merge_dataset_metadata(metas: list[dict], sessions: list[SessionInfo]) -> list[dict]:
    merged = {}
    for meta in metas:
        datasets = meta.get("datasets", [])
        if not isinstance(datasets, list):
            continue
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            key = dataset.get("short_name") or dataset.get("filename")
            if key:
                merged.setdefault(key, dataset)

    for session in sessions:
        for dataset in session.datasets:
            merged.setdefault(dataset, {"short_name": dataset})

    return list(merged.values())


def _dataset_names(datasets_meta: list[dict]) -> list[str]:
    names = []
    for dataset in datasets_meta:
        name = dataset.get("short_name") or dataset.get("filename")
        if name:
            names.append(str(name))
    return sorted(set(names))


def _merge_cli_args(metas: list[dict], dataset_names: list[str], sessions: list[SessionInfo]) -> dict:
    source_args = [
        meta.get("cli_args", {})
        for meta in metas
        if isinstance(meta.get("cli_args"), dict)
    ]
    return {
        "merged": True,
        "dataset": dataset_names,
        "source_sessions": [session.path.name for session in sessions],
        "source_cli_args": source_args,
    }


def merge_sessions(sessions: list[SessionInfo], experiment_dir: Path) -> SessionInfo:
    """Merge same-type session result databases into a new analyzable session."""
    if len(sessions) < 2:
        raise ValueError("Select at least two sessions to merge.")

    session_types = {session.type for session in sessions}
    if len(session_types) != 1:
        raise ValueError("Only sessions from the same experiment type can be merged.")

    benchmark_type = sessions[0].type
    if benchmark_type == "bayesian":
        raise ValueError("Bayesian sessions cannot be merged.")

    frames = []
    metas = []
    for session in sessions:
        df, meta = load_session(session.path)
        metas.append(meta)
        if df.empty:
            continue
        df = df.copy()
        df["source_session"] = session.path.name
        frames.append(df)

    if not frames:
        raise ValueError("Selected sessions did not contain any result rows.")

    merged_df = pd.concat(frames, ignore_index=True, sort=False)
    datasets_meta = _merge_dataset_metadata(metas, sessions)
    dataset_names = _dataset_names(datasets_meta)
    timestamp = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = experiment_dir / benchmark_type / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    conn = sqlite3.connect(str(out_dir / "results.db"))
    try:
        merged_df.to_sql("results", conn, if_exists="replace", index=False)
        db.write_metadata(conn, {
            "benchmark_type": benchmark_type,
            "timestamp": timestamp,
            "merged": True,
            "cli_args": _merge_cli_args(metas, dataset_names, sessions),
            "source_sessions": [
                {
                    "path": str(session.path),
                    "timestamp": session.timestamp,
                    "algorithms": session.algorithms,
                    "datasets": session.datasets,
                }
                for session in sessions
            ],
            "algorithms": _merge_algorithm_metadata(metas),
            "datasets": datasets_meta,
        })
    finally:
        conn.close()

    algorithms = [str(x) for x in sorted(merged_df["algorithm"].dropna().unique())] if "algorithm" in merged_df else []
    datasets = [str(x) for x in sorted(merged_df["dataset"].dropna().unique())] if "dataset" in merged_df else []

    return SessionInfo(
        path=out_dir,
        type=benchmark_type,
        timestamp=timestamp,
        algorithms=algorithms,
        datasets=datasets,
        has_db=True,
    )
