-- VCT 2026 Map Analytics Database Schema

CREATE TABLE IF NOT EXISTS map_results (
    match_id TEXT NOT NULL,
    game_number INTEGER NOT NULL,
    date DATE NOT NULL,
    event TEXT NOT NULL,
    stage TEXT,
    region TEXT,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    map TEXT NOT NULL,
    rounds_for INTEGER NOT NULL,
    rounds_against INTEGER NOT NULL,
    round_differential INTEGER NOT NULL,
    result TEXT CHECK(result IN ('W', 'L')) NOT NULL,
    PRIMARY KEY (match_id, game_number, team)
);

CREATE INDEX IF NOT EXISTS idx_map_results_team ON map_results(team);
CREATE INDEX IF NOT EXISTS idx_map_results_map ON map_results(map);
CREATE INDEX IF NOT EXISTS idx_map_results_date ON map_results(date);
CREATE INDEX IF NOT EXISTS idx_map_results_team_map ON map_results(team, map);
CREATE INDEX IF NOT EXISTS idx_map_results_region ON map_results(region);
