"""
Unit tests for database interaction and data structures.
"""

import sqlite3
import pandas as pd
import pytest
from src.database import init_db, query_dataframe, DEFAULT_DB_PATH
from src.data_pipeline import save_to_sqlite


def test_sqlite_save_and_query(tmp_path):
    test_db = tmp_path / "test_vct.db"
    
    # Create sample DataFrame
    data = [
        {
            "match_id": "m1",
            "game_number": 1,
            "date": "2026-07-15",
            "event": "VCT Americas",
            "stage": "Stage 2",
            "region": "Americas",
            "team": "G2 Esports",
            "opponent": "Sentinels",
            "map": "Ascent",
            "rounds_for": 13,
            "rounds_against": 8,
            "round_differential": 5,
            "result": "W"
        },
        {
            "match_id": "m1",
            "game_number": 1,
            "date": "2026-07-15",
            "event": "VCT Americas",
            "stage": "Stage 2",
            "region": "Americas",
            "team": "Sentinels",
            "opponent": "G2 Esports",
            "map": "Ascent",
            "rounds_for": 8,
            "rounds_against": 13,
            "round_differential": -5,
            "result": "L"
        }
    ]
    df = pd.DataFrame(data)

    save_to_sqlite(df, sqlite_path=test_db)

    # Query back
    res = query_dataframe("SELECT * FROM map_results WHERE team = ?;", ("G2 Esports",), db_path=test_db)
    assert len(res) == 1
    assert res.iloc[0]["map"] == "Ascent"
    assert res.iloc[0]["rounds_for"] == 13
    assert res.iloc[0]["result"] == "W"


def test_get_teams_by_league(tmp_path):
    from src.database import get_teams_by_league
    test_db = tmp_path / "test_vct.db"

    data = [
        {"match_id": "m1", "game_number": 1, "date": "2026-07-15", "event": "VCT Americas", "stage": "Stage 2", "region": "Americas", "team": "Sentinels", "opponent": "G2 Esports", "map": "Ascent", "rounds_for": 13, "rounds_against": 8, "round_differential": 5, "result": "W"},
        {"match_id": "m1", "game_number": 1, "date": "2026-07-15", "event": "VCT Americas", "stage": "Stage 2", "region": "Americas", "team": "G2 Esports", "opponent": "Sentinels", "map": "Ascent", "rounds_for": 8, "rounds_against": 13, "round_differential": -5, "result": "L"},
        {"match_id": "m2", "game_number": 1, "date": "2026-07-16", "event": "VCT EMEA", "stage": "Stage 2", "region": "EMEA", "team": "FNATIC", "opponent": "Team Liquid", "map": "Haven", "rounds_for": 13, "rounds_against": 7, "round_differential": 6, "result": "W"},
        {"match_id": "m2", "game_number": 1, "date": "2026-07-16", "event": "VCT EMEA", "stage": "Stage 2", "region": "EMEA", "team": "Team Liquid", "opponent": "FNATIC", "map": "Haven", "rounds_for": 7, "rounds_against": 13, "round_differential": -6, "result": "L"},
    ]
    save_to_sqlite(pd.DataFrame(data), sqlite_path=test_db)

    leagues = get_teams_by_league(db_path=test_db)
    assert "Americas" in leagues
    assert "EMEA" in leagues
    assert "Sentinels" in leagues["Americas"]
    assert "G2 Esports" in leagues["Americas"]
    assert "FNATIC" in leagues["EMEA"]
    assert "Team Liquid" in leagues["EMEA"]
