"""
Unit tests for VetoTendencyAnalyzer and permaban detection logic (90-day window, >=80% threshold).
"""

import pytest
import pandas as pd
from datetime import date
from src.veto_analysis import VetoTendencyAnalyzer, TeamVetoProfile
from src.power_score import PowerScoreModel, ACTIVE_MAP_POOL
from src.veto_simulator import VetoSimulator, ActionType


def test_permaban_threshold_and_sample_size():
    # Construct mock veto DataFrame all on the same date (within 90 days)
    # Team Alpha: 5 matches, bans Split 5 times in Phase 1 -> 100% (>=80%, >=3) -> Permaban
    # Team Beta: 5 matches, bans Haven 4 times in Phase 1 -> 80% (>=80%, >=3) -> Permaban
    # Team Gamma: 2 matches, bans Lotus 2 times in Phase 1 -> 100% (<3 matches) -> Insufficient Sample
    # Team Delta: 5 matches, bans Abyss 3 times in Phase 1 -> 60% (<80%) -> No Permaban
    # Team Epsilon: 4 matches, bans Ascent 3 times in Phase 1 -> 75% (<80%) -> No Permaban

    records = []
    # Team Alpha (5 series, 5 Split bans in Phase 1)
    for i in range(5):
        records.append({
            "match_id": f"alpha_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "Americas", "team": "Team Alpha", "opponent": "Team X",
            "action": "BAN", "phase": 1, "phase_order": 1, "map": "Split"
        })
        records.append({
            "match_id": f"alpha_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "Americas", "team": "Team Alpha", "opponent": "Team X",
            "action": "PICK", "phase": 2, "phase_order": 3, "map": "Ascent"
        })

    # Team Beta (5 series, 4 Haven bans in Phase 1, 1 Ascent ban) -> 4/5 = 80% (meets >= 80%)
    for i in range(4):
        records.append({
            "match_id": f"beta_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "EMEA", "team": "Team Beta", "opponent": "Team Y",
            "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"
        })
    records.append({
        "match_id": "beta_4", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
        "region": "EMEA", "team": "Team Beta", "opponent": "Team Y",
        "action": "BAN", "phase": 1, "phase_order": 1, "map": "Ascent"
    })

    # Team Gamma (2 series, 2 Lotus bans in Phase 1) -> 2/2 = 100%, but 2 < 3 series
    for i in range(2):
        records.append({
            "match_id": f"gamma_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "Pacific", "team": "Team Gamma", "opponent": "Team Z",
            "action": "BAN", "phase": 1, "phase_order": 1, "map": "Lotus"
        })

    # Team Epsilon (4 series, 3 Ascent bans in Phase 1) -> 3/4 = 75% (< 80%) -> No Permaban
    for i in range(3):
        records.append({
            "match_id": f"eps_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "EMEA", "team": "Team Epsilon", "opponent": "Team Y",
            "action": "BAN", "phase": 1, "phase_order": 1, "map": "Ascent"
        })
    records.append({
        "match_id": "eps_3", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
        "region": "EMEA", "team": "Team Epsilon", "opponent": "Team Y",
        "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"
    })

    df = pd.DataFrame(records)

    analyzer = VetoTendencyAnalyzer(
        permaban_threshold=0.80,
        min_sample_size=3,
        window_days=90,
        reference_date=date(2026, 8, 1)
    )
    analyzer._load_data = lambda: df
    analyzer._load_and_analyze()

    # Verify Team Alpha: Permaban = Split (100%)
    prof_alpha = analyzer.get_profile("Team Alpha")
    assert prof_alpha is not None
    assert prof_alpha.total_series == 5
    assert prof_alpha.permaban == "Split"
    assert prof_alpha.permaban_rate == 1.0
    assert prof_alpha.comfort_pick == "Ascent"
    assert analyzer.is_permaban("Team Alpha", "Split") is True

    # Verify Team Beta: 4/5 = 80.0% -> meets >= 80% -> Permaban = Haven
    prof_beta = analyzer.get_profile("Team Beta")
    assert prof_beta is not None
    assert prof_beta.total_series == 5
    assert prof_beta.permaban == "Haven"
    assert prof_beta.permaban_rate == 0.80

    # Verify Team Epsilon: 3/4 = 75.0% -> < 80% -> No permaban
    prof_eps = analyzer.get_profile("Team Epsilon")
    assert prof_eps is not None
    assert prof_eps.permaban is None
    assert prof_eps.top_ban == "Ascent"
    assert "No 80% permaban" in prof_eps.status_note

    # Verify Team Gamma: 2/2 = 100% but sample size 2 < 3 -> permaban is None, insufficient_sample = True
    prof_gamma = analyzer.get_profile("Team Gamma")
    assert prof_gamma is not None
    assert prof_gamma.total_series == 2
    assert prof_gamma.permaban is None
    assert prof_gamma.insufficient_sample is True
    assert "Sample size too small" in prof_gamma.status_note


def test_90_day_window_filtering():
    ref = date(2026, 9, 1)

    # Team Omega:
    # 3 old matches 100 days ago (2026-05-24): bans Breeze
    # 4 recent matches within 90 days (2026-08-15): bans Haven
    # With 90-day window: the 3 old Breeze bans are ignored!
    # Only the 4 recent Haven bans are analyzed -> Haven 4/4 = 100% -> Permaban = Haven.
    records = [
        {"match_id": "old_1", "date": "2026-05-24", "event": "VCT", "stage": "Stage 1", "region": "Americas",
         "team": "Team Omega", "opponent": "Team X", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Breeze"},
        {"match_id": "old_2", "date": "2026-05-24", "event": "VCT", "stage": "Stage 1", "region": "Americas",
         "team": "Team Omega", "opponent": "Team Y", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Breeze"},
        {"match_id": "old_3", "date": "2026-05-24", "event": "VCT", "stage": "Stage 1", "region": "Americas",
         "team": "Team Omega", "opponent": "Team Z", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Breeze"},
        {"match_id": "new_1", "date": "2026-08-15", "event": "VCT", "stage": "Stage 2", "region": "Americas",
         "team": "Team Omega", "opponent": "Team A", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"},
        {"match_id": "new_2", "date": "2026-08-15", "event": "VCT", "stage": "Stage 2", "region": "Americas",
         "team": "Team Omega", "opponent": "Team B", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"},
        {"match_id": "new_3", "date": "2026-08-15", "event": "VCT", "stage": "Stage 2", "region": "Americas",
         "team": "Team Omega", "opponent": "Team C", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"},
        {"match_id": "new_4", "date": "2026-08-15", "event": "VCT", "stage": "Stage 2", "region": "Americas",
         "team": "Team Omega", "opponent": "Team D", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Haven"},
    ]

    analyzer = VetoTendencyAnalyzer(
        permaban_threshold=0.80,
        min_sample_size=3,
        window_days=90,
        reference_date=ref
    )
    analyzer._load_data = lambda: pd.DataFrame(records)
    analyzer._load_and_analyze()

    prof = analyzer.get_profile("Team Omega")
    assert prof is not None
    # Only the 4 matches in the last 90 days are counted
    assert prof.total_series == 4
    assert prof.permaban == "Haven"
    assert prof.permaban_rate == 1.0
    assert "Breeze" not in prof.first_cycle_bans or prof.first_cycle_bans["Breeze"] == 0


def test_picklist_generation():
    map_data = [
        {"date": "2026-08-01", "team": "Team Alpha", "opponent": "Team Beta", "map": "Split", "rounds_for": 13, "rounds_against": 2, "result": "W"},
        {"date": "2026-08-01", "team": "Team Beta", "opponent": "Team Alpha", "map": "Split", "rounds_for": 2, "rounds_against": 13, "result": "L"},
        {"date": "2026-08-01", "team": "Team Alpha", "opponent": "Team Beta", "map": "Ascent", "rounds_for": 13, "rounds_against": 5, "result": "W"},
        {"date": "2026-08-01", "team": "Team Beta", "opponent": "Team Alpha", "map": "Ascent", "rounds_for": 5, "rounds_against": 13, "result": "L"},
        {"date": "2026-08-01", "team": "Team Alpha", "opponent": "Team Beta", "map": "Haven", "rounds_for": 13, "rounds_against": 11, "result": "W"},
        {"date": "2026-08-01", "team": "Team Beta", "opponent": "Team Alpha", "map": "Haven", "rounds_for": 11, "rounds_against": 13, "result": "L"},
    ]
    model = PowerScoreModel().fit(pd.DataFrame(map_data))

    # Team Alpha has permaban on Split in the 90-day window
    veto_records = [
        {"match_id": f"m_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2", "region": "Americas",
         "team": "Team Alpha", "opponent": "Team Beta", "action": "BAN", "phase": 1, "phase_order": 1, "map": "Split"}
        for i in range(5)
    ]
    analyzer = VetoTendencyAnalyzer(
        permaban_threshold=0.80,
        min_sample_size=3,
        window_days=90,
        reference_date=date(2026, 8, 1)
    )
    analyzer._load_data = lambda: pd.DataFrame(veto_records)
    analyzer._load_and_analyze()

    pool = ["Ascent", "Split", "Haven", "Lotus", "Sunset", "Bind", "Breeze"]
    picklist = analyzer.generate_picklist("Team Alpha", "Team Beta", model, map_pool=pool)

    assert len(picklist) == 7
    # Split must be at the very bottom (index 6, Rank 7) despite high score, because it is Team Alpha's permaban!
    last_item = picklist[-1]
    assert last_item["rank"] == 7
    assert last_item["map"] == "Split"
    assert last_item["is_permaban"] is True
    assert "Permaban" in last_item["note"]

    # All regular maps (first 6) must be ordered descending by diff
    diffs = [item["diff"] for item in picklist[:-1]]
    assert diffs == sorted(diffs, reverse=True)


def test_veto_simulator_permaban_recommendation():
    map_data = [
        {"date": "2026-08-01", "team": "Team Alpha", "opponent": "Team Beta", "map": "Ascent", "rounds_for": 13, "rounds_against": 4, "result": "W"},
        {"date": "2026-08-01", "team": "Team Beta", "opponent": "Team Alpha", "map": "Ascent", "rounds_for": 4, "rounds_against": 13, "result": "L"},
        {"date": "2026-08-01", "team": "Team Alpha", "opponent": "Team Beta", "map": "Split", "rounds_for": 13, "rounds_against": 11, "result": "W"},
        {"date": "2026-08-01", "team": "Team Beta", "opponent": "Team Alpha", "map": "Split", "rounds_for": 11, "rounds_against": 13, "result": "L"},
    ]
    model = PowerScoreModel().fit(pd.DataFrame(map_data))

    veto_records = []
    for i in range(5):
        veto_records.append({
            "match_id": f"m_{i}", "date": "2026-08-01", "event": "VCT", "stage": "Stage 2",
            "region": "Americas", "team": "Team Alpha", "opponent": "Team Beta",
            "action": "BAN", "phase": 1, "phase_order": 1, "map": "Split"
        })
    analyzer = VetoTendencyAnalyzer(
        permaban_threshold=0.80,
        min_sample_size=3,
        window_days=90,
        reference_date=date(2026, 8, 1)
    )
    analyzer._load_data = lambda: pd.DataFrame(veto_records)
    analyzer._load_and_analyze()

    sim = VetoSimulator(model=model, map_pool=["Ascent", "Split", "Haven", "Lotus", "Sunset", "Bind", "Breeze"], analyzer=analyzer)
    state = sim.create_session("Team Alpha", "Team Beta")

    # Step 1: Team Alpha Turn 1 ban -> Should recommend "Split" due to Permaban
    rec_map, rec_reason = sim.recommend_action(state)
    assert rec_map == "Split"
    assert "Permaban" in rec_reason
    assert "Team Alpha bans Split in 100.0%" in rec_reason
