"""
Unit tests for VetoSimulator and BO3 pick/ban logic.
"""

import pytest
import pandas as pd
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL
from src.veto_simulator import VetoSimulator, ActionType


@pytest.fixture
def mock_model():
    data = [
        {"date": "2026-08-01", "team": "G2 Esports", "opponent": "Sentinels", "map": "Ascent", "rounds_for": 13, "rounds_against": 4, "result": "W"},
        {"date": "2026-08-01", "team": "Sentinels", "opponent": "G2 Esports", "map": "Ascent", "rounds_for": 4, "rounds_against": 13, "result": "L"},
        {"date": "2026-08-05", "team": "Sentinels", "opponent": "G2 Esports", "map": "Sunset", "rounds_for": 13, "rounds_against": 6, "result": "W"},
        {"date": "2026-08-05", "team": "G2 Esports", "opponent": "Sentinels", "map": "Sunset", "rounds_for": 6, "rounds_against": 13, "result": "L"},
    ]
    df = pd.DataFrame(data)
    return PowerScoreModel().fit(df)


def test_veto_simulation_completion(mock_model):
    sim = VetoSimulator(model=mock_model, map_pool=ACTIVE_MAP_POOL)
    state = sim.simulate_full_veto("G2 Esports", "Sentinels")

    assert state.is_complete
    assert len(state.picked_maps) == 2
    assert len(state.banned_maps) == 4
    assert state.decider_map is not None
    assert len(state.available_maps) == 0

    # Total 7 steps (4 bans, 2 picks, 1 decider)
    assert len(state.history) == 7

    # Series odds calculation
    odds = sim.calculate_series_odds(state)
    assert "team_a_pct" in odds
    assert "team_b_pct" in odds
    assert pytest.approx(odds["team_a_pct"] + odds["team_b_pct"], 0.1) == 100.0


def test_step_by_step_execution(mock_model):
    sim = VetoSimulator(model=mock_model, map_pool=ACTIVE_MAP_POOL)
    state = sim.create_session("G2 Esports", "Sentinels")

    # Step 1: Team A ban
    p1 = sim.get_current_prompt(state)
    assert p1["step_number"] == 1
    assert p1["action_type"] == ActionType.BAN
    assert p1["actor"] == "G2 Esports"

    # Execute ban on Sunset
    state = sim.execute_step(state, "Sunset", "Target ban")
    assert "Sunset" in state.banned_maps
    assert "Sunset" not in state.available_maps
    assert state.current_step_index == 1

    # Step 2: Team B ban
    p2 = sim.get_current_prompt(state)
    assert p2["step_number"] == 2
    assert p2["action_type"] == ActionType.BAN
    assert p2["actor"] == "Sentinels"
