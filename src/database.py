"""
Database helper module for VCT Map Analytics.
Provides a clean SQLite interface for querying match and map results.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vct_analytics.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None, schema_path: Optional[Path] = None) -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_connection(db_path)
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
    
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    conn.close()


def query_dataframe(query: str, params: Optional[tuple] = None, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Executes a SQL query and returns results as a pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_all_teams(db_path: Optional[Path] = None) -> List[str]:
    """Retrieves unique teams sorted alphabetically."""
    query = "SELECT DISTINCT team FROM map_results ORDER BY team;"
    df = query_dataframe(query, db_path=db_path)
    return df["team"].tolist() if not df.empty else []


def get_all_maps(db_path: Optional[Path] = None) -> List[str]:
    """Retrieves unique maps sorted alphabetically."""
    query = "SELECT DISTINCT map FROM map_results ORDER BY map;"
    df = query_dataframe(query, db_path=db_path)
    return df["map"].tolist() if not df.empty else []


def get_team_map_summary(team: Optional[str] = None, map_name: Optional[str] = None, db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Returns aggregated record, win rate, and round margins for teams and maps.
    Demonstrates SQL aggregation and filtering.
    """
    conditions = []
    params = []
    if team:
        conditions.append("team = ?")
        params.append(team)
    if map_name:
        conditions.append("map = ?")
        params.append(map_name)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
    SELECT 
        team,
        map,
        COUNT(*) AS maps_played,
        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) AS wins,
        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) AS losses,
        ROUND(100.0 * SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate_pct,
        ROUND(AVG(rounds_for), 1) AS avg_rounds_for,
        ROUND(AVG(rounds_against), 1) AS avg_rounds_against,
        ROUND(AVG(round_differential), 2) AS avg_round_margin
    FROM map_results
    {where_clause}
    GROUP BY team, map
    ORDER BY team, win_rate_pct DESC, avg_round_margin DESC;
    """
    return query_dataframe(query, tuple(params) if params else None, db_path=db_path)


def get_head_to_head_matches(team_a: str, team_b: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Returns all 2026 map encounters between two teams."""
    query = """
    SELECT 
        date,
        event,
        stage,
        map,
        team,
        rounds_for,
        rounds_against,
        opponent,
        result
    FROM map_results
    WHERE team = ? AND opponent = ?
    ORDER BY date DESC;
    """
    return query_dataframe(query, (team_a, team_b), db_path=db_path)


def get_team_leagues(db_path: Optional[Path] = None) -> Dict[str, str]:
    """Returns a dictionary mapping each team to their primary league (Americas, EMEA, Pacific, China)."""
    query = """
    WITH TeamRegions AS (
        SELECT 
            team, 
            region, 
            COUNT(*) as cnt,
            ROW_NUMBER() OVER (PARTITION BY team ORDER BY COUNT(*) DESC) as rn
        FROM map_results
        WHERE region IN ('Americas', 'EMEA', 'Pacific', 'China')
        GROUP BY team, region
    )
    SELECT team, region
    FROM TeamRegions
    WHERE rn = 1
    ORDER BY team;
    """
    df = query_dataframe(query, db_path=db_path)
    return dict(zip(df["team"], df["region"])) if not df.empty else {}


CHAMPIONS_2026_TEAMS: List[str] = [
    "100 Thieves",
    "EDward Gaming",
    "FUT Esports",
    "G2 Esports",
    "Global Esports",
    "JD Gaming",
    "Karmine Corp",
    "LOUD",
    "Nongshim RedForce",
    "NRG",
    "Paper Rex",
    "T1",
    "Team Liquid",
    "Team Vitality",
    "TYLOO",
    "Xi Lai Gaming",
]


def get_teams_by_league(db_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Returns a dictionary mapping each league to a sorted list of teams."""
    team_to_league = get_team_leagues(db_path)
    league_to_teams: Dict[str, List[str]] = {
        "Americas": [],
        "EMEA": [],
        "Pacific": [],
        "China": [],
        "Teams at Champions 2026": list(sorted(CHAMPIONS_2026_TEAMS)),
    }
    for team, league in team_to_league.items():
        if league in league_to_teams:
            league_to_teams[league].append(team)
    for l in league_to_teams:
        league_to_teams[l].sort()
    return league_to_teams

