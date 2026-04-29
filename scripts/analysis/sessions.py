import sqlite3
from dataclasses import dataclass, field
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
