import json
import sqlite3
import pandas as pd
from pathlib import Path

from scripts.config import PARAM_CONFIG


def init_db(session_dir: Path) -> sqlite3.Connection:
    db_path = session_dir / "results.db"
    return sqlite3.connect(str(db_path))

def init_results_schema(conn: sqlite3.Connection, first_metric_keys: list[str]) -> None:
    """Creates the result table using a combination of base columns,
    all possible parameters, and experiment-specific columns."""

    columns = ["dataset", "algorithm", "run", "time", "ratio"]

    columns.extend(list(PARAM_CONFIG.keys()))

    for key in first_metric_keys:
        if key not in columns:
            columns.append(key)

    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_sql("results", conn, if_exists="replace", index=False)

def write_results_bulk(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Writes the entire DataFrame to the DB in one go."""
    if not df.empty:
        # if_exists="append" will automatically add new columns if your DataFrame changes later!
        df.to_sql("results", conn, if_exists="append", index=False)

def write_result(conn: sqlite3.Connection, result: dict) -> None:
    """Writes a single result dictionary to the DB."""
    if result:
        df = pd.DataFrame([result])
        df.to_sql("results", conn, if_exists="append", index=False)

def read_results(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM results", conn)
    return df

def write_metadata(conn: sqlite3.Connection, manifest: dict) -> None:
    """Writes the experiment metadata to a single-row metadata table."""
    db_manifest = {}
    for key, value in manifest.items():
        # Convert nested dictionaries and lists to JSON strings for SQLite storage
        if isinstance(value, (dict, list)):
            db_manifest[key] = json.dumps(value)
        else:
            db_manifest[key] = value

    df = pd.DataFrame([db_manifest])
    df.to_sql("metadata", conn, if_exists="replace", index=False)

def read_metadata(conn: sqlite3.Connection) -> dict:
    """Reads the metadata from the DB and parses JSON strings back to dicts."""
    try:
        df = pd.read_sql("SELECT * FROM metadata", conn)
    except pd.io.sql.DatabaseError:
        return {} # Table doesn't exist

    if df.empty:
        return {}

    manifest = df.iloc[0].to_dict()
    for key, value in manifest.items():
        if isinstance(value, str):
            try:
                # Attempt to parse JSON strings back into Python lists/dicts
                manifest[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return manifest