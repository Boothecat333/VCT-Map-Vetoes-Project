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

-- VCT 2026 Match Vetoes Schema
CREATE TABLE IF NOT EXISTS match_vetoes (
    match_id TEXT NOT NULL,
    date DATE NOT NULL,
    event TEXT NOT NULL,
    stage TEXT,
    region TEXT,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    action TEXT CHECK(action IN ('BAN', 'PICK', 'DECIDER')) NOT NULL,
    phase INTEGER NOT NULL,            -- 1: First Cycle Ban, 2: Pick, 3: Second Cycle Ban, 4: Decider
    phase_order INTEGER NOT NULL,      -- Absolute step order in veto (1 to 7)
    map TEXT NOT NULL,
    PRIMARY KEY (match_id, phase_order, team)
);

CREATE INDEX IF NOT EXISTS idx_vetoes_team ON match_vetoes(team);
CREATE INDEX IF NOT EXISTS idx_vetoes_map ON match_vetoes(map);
CREATE INDEX IF NOT EXISTS idx_vetoes_action_phase ON match_vetoes(action, phase);

