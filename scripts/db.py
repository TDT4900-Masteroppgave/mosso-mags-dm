"""SQLite persistence layer for benchmark results."""
import json
import sqlite3
from pathlib import Path

import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm  TEXT    NOT NULL,
    dataset    TEXT    NOT NULL,
    rep        INTEGER DEFAULT 0,
    checkpoint REAL    DEFAULT 1.0,
    trial      INTEGER DEFAULT 0,
    sample     INTEGER DEFAULT 0,
    time       REAL,
    ratio      REAL,
    memory_mb  REAL
);

CREATE TABLE IF NOT EXISTS result_params (
    result_id  INTEGER NOT NULL REFERENCES results(id),
    name       TEXT    NOT NULL,
    value      REAL,
    PRIMARY KEY (result_id, name)
);
"""


def init_db(session_dir: Path) -> sqlite3.Connection:
    db_path = Path(session_dir) / "results.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def write_metadata(conn: sqlite3.Connection, benchmark_type: str, cli_args: dict, session_dir) -> None:
    rows = [
        ("benchmark_type", benchmark_type),
        ("session_dir", str(session_dir)),
        ("cli_args", json.dumps(cli_args, default=str)),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", rows
    )
    conn.commit()


def write_result(
    conn: sqlite3.Connection,
    algorithm: str,
    dataset: str,
    time: float | None,
    ratio: float | None,
    memory_mb: float | None = None,
    rep: int = 0,
    checkpoint: float = 1.0,
    trial: int = 0,
    sample: int = 0,
    params: dict | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO results (algorithm, dataset, rep, checkpoint, trial, sample, time, ratio, memory_mb)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (algorithm, dataset, rep, checkpoint, trial, sample, time, ratio, memory_mb),
    )
    result_id = cur.lastrowid
    if params:
        conn.executemany(
            "INSERT OR IGNORE INTO result_params (result_id, name, value) VALUES (?, ?, ?)",
            [(result_id, k, float(v)) for k, v in params.items() if v is not None],
        )
    conn.commit()
    return result_id


def read_results(
    session_dirs: list[Path],
    algorithms: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for sdir in session_dirs:
        db_path = Path(sdir) / "results.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))

        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        benchmark_type = meta.get("benchmark_type", "unknown")

        df = pd.read_sql_query("SELECT * FROM results", conn)
        params_df = pd.read_sql_query("SELECT * FROM result_params", conn)
        conn.close()

        if not df.empty and not params_df.empty:
            pivot = params_df.pivot_table(index="result_id", columns="name", values="value", aggfunc="first")
            pivot.columns = [f"param_{c}" for c in pivot.columns]
            pivot = pivot.reset_index().rename(columns={"result_id": "id"})
            df = df.merge(pivot, on="id", how="left")

        df["session"] = str(sdir)
        df["benchmark_type"] = benchmark_type
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if algorithms:
        result = result[result["algorithm"].isin(algorithms)]
    if datasets:
        result = result[result["dataset"].isin(datasets)]

    return result


def get_metadata(session_dir: Path) -> dict:
    db_path = Path(session_dir) / "results.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
    conn.close()
    return meta
