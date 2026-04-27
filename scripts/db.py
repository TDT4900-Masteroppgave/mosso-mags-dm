import json
import sqlite3
import pandas as pd
from pathlib import Path

def init_db(session_dir: Path) -> sqlite3.Connection:
    db_path = session_dir / "results.db"
    return sqlite3.connect(str(db_path))

def write_results_bulk(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Writes the entire DataFrame to the DB in one go."""
    if not df.empty:
        # if_exists="append" will automatically add new columns if your DataFrame changes later!
        df.to_sql("results", conn, if_exists="append", index=False)

def read_results(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM results", conn)
    params_df = pd.json_normalize(df['params'].apply(json.loads))
    df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
    return df


