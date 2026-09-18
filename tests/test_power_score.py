"""
Unit tests for the PowerScoreModel.
"""

from datetime import date, timedelta
import pandas as pd
import pytest
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL, DEFAULT_DECAY_DAYS, DEFAULT_RETENTION_RATE


def test_recency_weight_decay():
    model = PowerScoreModel(decay_days=60.0, retention_rate=0.75)
    ref_date = date(2026, 9, 18)
    
    # Same day should have weight 1.0
    assert model.compute_recency_weight(ref_date, ref_date) == 1.0
    
    # 60 days ago should have weight 0.75 (75% retention)
    date_60d_ago = ref_date - timedelta(days=60)
    w_60 = model.compute_recency_weight(date_60d_ago, ref_date)
    assert pytest.approx(w_60, 0.01) == 0.75
    
    # 120 days ago should have weight 0.75^2 = 0.5625
    date_120d_ago = ref_date - timedelta(days=120)
    w_120 = model.compute_recency_weight(date_120d_ago, ref_date)
    assert pytest.approx(w_120, 0.01) == 0.5625


def test_round_diff_transform():
    model = PowerScoreModel()
    assert model._transform_round_differential(0) == 0.0
    # Positive should be positive, negative should be negative
    pos = model._transform_round_differential(10)
    neg = model._transform_round_differential(-10)
    assert pos > 0
    assert neg < 0
    assert pytest.approx(pos, 0.01) == -neg


def test_model_fitting_and_power_scores():
    # Construct mock dataset
    data = [
        # G2 beats SEN on Ascent 13-5
        {"date": "2026-08-01", "team": "G2 Esports", "opponent": "Sentinels", "map": "Ascent", "rounds_for": 13, "rounds_against": 5, "result": "W"},
        {"date": "2026-08-01", "team": "Sentinels", "opponent": "G2 Esports", "map": "Ascent", "rounds_for": 5, "rounds_against": 13, "result": "L"},
        # G2 beats FNC on Ascent 13-7
        {"date": "2026-08-15", "team": "G2 Esports", "opponent": "Fnatic", "map": "Ascent", "rounds_for": 13, "rounds_against": 7, "result": "W"},
        {"date": "2026-08-15", "team": "Fnatic", "opponent": "G2 Esports", "map": "Ascent", "rounds_for": 7, "rounds_against": 13, "result": "L"},
        # SEN beats FNC on Haven 13-8
        {"date": "2026-08-20", "team": "Sentinels", "opponent": "Fnatic", "map": "Haven", "rounds_for": 13, "rounds_against": 8, "result": "W"},
        {"date": "2026-08-20", "team": "Fnatic", "opponent": "Sentinels", "map": "Haven", "rounds_for": 8, "rounds_against": 13, "result": "L"},
    ]
    df = pd.DataFrame(data)
    model = PowerScoreModel(decay_days=60.0, retention_rate=0.75, reference_date=date(2026, 9, 1))
    model.fit(df)

    g2_ascent = model.get_rating("G2 Esports", "Ascent")
    sen_ascent = model.get_rating("Sentinels", "Ascent")

    # G2 should have a significantly higher Power Score than Sentinels on Ascent
    assert g2_ascent.power_score > sen_ascent.power_score
    assert 0 <= g2_ascent.power_score <= 100
    assert 0 <= sen_ascent.power_score <= 100
    assert g2_ascent.wins == 2
    assert g2_ascent.losses == 0

    # Win probability for G2 over SEN on Ascent should be > 50%
    prob = model.predict_map_win_probability("G2 Esports", "Sentinels", "Ascent")
    assert prob > 0.5


def test_unplayed_map_baseline():
    model = PowerScoreModel()
    rating = model.get_rating("Team Liquid", "Abyss")
    assert rating.power_score == 50.0
    assert rating.maps_played == 0
