"""
Data extraction and transformation pipeline for VCT 2026.
Downloads the live DuckDB database from vct-reference.com,
extracts completed 2026 tier 1 map results, and populates SQLite and CSV outputs.
"""

import os
import sys
import sqlite3
import urllib.request
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd

DUCKDB_DOWNLOAD_URL = "https://vct-reference.com/dataset/vct.duckdb"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DUCKDB_PATH = RAW_DATA_DIR / "vct.duckdb"
DEFAULT_CSV_PATH = PROCESSED_DATA_DIR / "vct_2026_maps.csv"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "vct_analytics.db"


def ensure_directories():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_duckdb(target_path: Optional[Path] = None, force: bool = False) -> Path:
    """Downloads vct.duckdb from vct-reference.com if not already present."""
    dest = target_path or DEFAULT_DUCKDB_PATH
    ensure_directories()

    if dest.exists() and not force:
        print(f"[Pipeline] DuckDB database already exists at {dest}")
        return dest

    print(f"[Pipeline] Downloading latest DuckDB database from {DUCKDB_DOWNLOAD_URL}...")
    headers = {"User-Agent": "VCT-Map-Analytics-Pipeline/1.0"}
    req = urllib.request.Request(DUCKDB_DOWNLOAD_URL, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        total_size = response.length
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded * 100 / total_size
                print(f"\r[Pipeline] Downloading: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="")

    print(f"\n[Pipeline] Successfully saved DuckDB database to {dest}")
    return dest


def extract_2026_maps(duckdb_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Connects to vct.duckdb and extracts all completed 2026 tier 1 maps.
    Transforms raw match/map entries into two-way perspective observations.
    """
    db_file = duckdb_path or DEFAULT_DUCKDB_PATH
    if not db_file.exists():
        raise FileNotFoundError(f"DuckDB database not found at {db_file}. Run download_duckdb() first.")

    con = duckdb.connect(str(db_file), read_only=True)

    query = """
    SELECT 
        m.match_id,
        mp.game_id AS game_number,
        CAST(m.utc_timestamp AS DATE) AS match_date,
        m.event AS event_name,
        COALESCE(m.series_stage, '') AS stage_name,
        COALESCE(m.region, 'International') AS region,
        t0.team_name AS team1,
        t1.team_name AS team2,
        mp.map_name,
        mp.score0 AS team1_score,
        mp.score1 AS team2_score
    FROM matches m
    JOIN maps mp ON m.match_id = mp.match_id
    JOIN teams t0 ON m.team0_id = t0.team_id
    JOIN teams t1 ON m.team1_id = t1.team_id
    WHERE EXTRACT(year FROM m.utc_timestamp) = 2026
    AND mp.map_name IS NOT NULL
    AND mp.score0 IS NOT NULL
    AND mp.score1 IS NOT NULL
    AND (mp.score0 + mp.score1) > 0
    ORDER BY m.utc_timestamp ASC, m.match_id, mp.game_id;
    """

    try:
        raw_df = con.execute(query).df()
    finally:
        con.close()

    print(f"[Pipeline] Extracted {len(raw_df)} raw 2026 maps from DuckDB.")

    # Transform into balanced two-way perspective observations
    rows = []
    for _, row in raw_df.iterrows():
        match_id = str(row["match_id"])
        game_num = int(row["game_number"])
        date_val = str(row["match_date"])
        event = str(row["event_name"]).strip()
        stage = str(row["stage_name"]).strip()
        region = str(row["region"]).strip()
        t1 = str(row["team1"]).strip()
        t2 = str(row["team2"]).strip()
        map_name = str(row["map_name"]).strip().capitalize()
        s1 = int(row["team1_score"])
        s2 = int(row["team2_score"])

        if s1 == 0 and s2 == 0:
            continue
        if not map_name or map_name.lower() in ("unknown", "tbd"):
            continue

        # Perspective 1: Team 1
        rows.append({
            "match_id": match_id,
            "game_number": game_num,
            "date": date_val,
            "event": event,
            "stage": stage,
            "region": region,
            "team": t1,
            "opponent": t2,
            "map": map_name,
            "rounds_for": s1,
            "rounds_against": s2,
            "round_differential": s1 - s2,
            "result": "W" if s1 > s2 else "L",
        })

        # Perspective 2: Team 2
        rows.append({
            "match_id": match_id,
            "game_number": game_num,
            "date": date_val,
            "event": event,
            "stage": stage,
            "region": region,
            "team": t2,
            "opponent": t1,
            "map": map_name,
            "rounds_for": s2,
            "rounds_against": s1,
            "round_differential": s2 - s1,
            "result": "W" if s2 > s1 else "L",
        })

    clean_df = pd.DataFrame(rows)
    print(f"[Pipeline] Generated {len(clean_df)} balanced perspective observations.")
    return clean_df


def save_to_csv(df: pd.DataFrame, csv_path: Optional[Path] = None):
    """Saves cleaned DataFrame to CSV."""
    dest = csv_path or DEFAULT_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, encoding="utf-8")
    print(f"[Pipeline] Saved CSV to {dest}")


def save_to_sqlite(df: pd.DataFrame, sqlite_path: Optional[Path] = None, schema_path: Optional[Path] = None):
    """Loads cleaned DataFrame into SQLite database."""
    dest = sqlite_path or DEFAULT_SQLITE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    schema_file = schema_path or (PROJECT_ROOT / "sql" / "schema.sql")
    conn = sqlite3.connect(str(dest))
    
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

    cursor = conn.cursor()
    cursor.execute("DELETE FROM map_results;")
    insert_sql = """
    INSERT OR REPLACE INTO map_results (
        match_id, game_number, date, event, stage, region, team, opponent,
        map, rounds_for, rounds_against, round_differential, result
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    working_df = df.copy()
    if "region" not in working_df.columns:
        working_df["region"] = "International"

    records = working_df[[
        "match_id", "game_number", "date", "event", "stage", "region", "team", "opponent",
        "map", "rounds_for", "rounds_against", "round_differential", "result"
    ]].values.tolist()

    cursor.executemany(insert_sql, records)
    conn.commit()
    conn.close()
    print(f"[Pipeline] Loaded {len(records)} rows into SQLite database at {dest}")


def run_pipeline(force_download: bool = False):
    """Executes end-to-end data pipeline."""
    ensure_directories()
    duckdb_file = download_duckdb(force=force_download)
    df = extract_2026_maps(duckdb_file)
    save_to_csv(df)
    save_to_sqlite(df)
    print("[Pipeline] Data pipeline completed successfully!")
    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_pipeline(force_download=force)
