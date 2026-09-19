"""
Veto tendency analysis and permaban detection for VCT 2026.
Analyzes historical pick and ban logs to identify team habits, comfort picks,
and true permabans based on empirical first-cycle ban rates.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "vct_analytics.db"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "vct_2026_vetoes.csv"

DEFAULT_WINDOW_DAYS = 90
DEFAULT_PERMABAN_THRESHOLD = 0.80
DEFAULT_MIN_SAMPLE_SIZE = 3


@dataclass
class TeamVetoProfile:
    team: str
    total_series: int
    first_cycle_bans: Dict[str, int] = field(default_factory=dict)
    first_cycle_ban_rates: Dict[str, float] = field(default_factory=dict)
    permaban: Optional[str] = None
    permaban_rate: Optional[float] = None
    top_ban: Optional[str] = None
    top_ban_rate: Optional[float] = None
    insufficient_sample: bool = False
    status_note: str = ""
    picks: Dict[str, int] = field(default_factory=dict)
    pick_rates: Dict[str, float] = field(default_factory=dict)
    comfort_pick: Optional[str] = None
    comfort_pick_rate: Optional[float] = None


class VetoTendencyAnalyzer:
    """
    Analyzes historical veto logs from the past 90 days (matching the active map pool era)
    to detect permabans and pick tendencies.
    
    A team's Permaban is defined as:
    1. Banned in the first cycle (Phase 1 / Turn 1 or Turn 2).
    2. Ban rate of at least 80% (ban_rate >= 0.80) within the 90-day window.
    3. Minimum sample size of >= 3 matches in the window.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        permaban_threshold: float = DEFAULT_PERMABAN_THRESHOLD,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        window_days: Optional[int] = DEFAULT_WINDOW_DAYS,
        reference_date: Optional[date] = None,
    ):
        self.db_path = db_path or DEFAULT_SQLITE_PATH
        self.csv_path = csv_path or DEFAULT_CSV_PATH
        self.permaban_threshold = permaban_threshold
        self.min_sample_size = min_sample_size
        self.window_days = window_days
        self.reference_date = reference_date
        self.profiles: Dict[str, TeamVetoProfile] = {}
        self._load_and_analyze()

    def _load_data(self) -> pd.DataFrame:
        """Loads veto data from SQLite database or fallback CSV."""
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path))
                df = pd.read_sql_query("SELECT * FROM match_vetoes", conn)
                conn.close()
                if not df.empty:
                    return df
            except Exception:
                pass

        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)

        return pd.DataFrame(columns=[
            "match_id", "date", "event", "stage", "region", "team", "opponent",
            "action", "phase", "phase_order", "map"
        ])

    def _load_and_analyze(self):
        """Processes veto records and populates team profiles using the 90-day window."""
        df = self._load_data()
        if df.empty:
            return

        working_df = df.copy()
        working_df["date_obj"] = pd.to_datetime(working_df["date"]).dt.date
        ref_date = self.reference_date or (working_df["date_obj"].max() if not working_df.empty else date.today())

        # Filter to matches within window_days of reference_date
        if self.window_days is not None:
            cutoff_date = ref_date - pd.Timedelta(days=self.window_days)
            working_df = working_df[working_df["date_obj"] >= cutoff_date]

        # Unique matches per team in the window
        team_matches_df = working_df[working_df["team"] != "Decider"][["team", "match_id"]].drop_duplicates()
        team_total_series = team_matches_df.groupby("team")["match_id"].nunique().to_dict()

        # Phase 1 bans (First cycle bans)
        p1_bans = working_df[(working_df["action"] == "BAN") & (working_df["phase"] == 1)]
        p1_counts = (
            p1_bans.groupby(["team", "map"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        # Phase 2 picks
        picks_df = working_df[working_df["action"] == "PICK"]
        pick_counts = (
            picks_df.groupby(["team", "map"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        window_label = f"past {self.window_days}d" if self.window_days else "all matches"

        for team, total_matches in team_total_series.items():
            raw_bans = {m: int(cnt) for m, cnt in p1_counts.get(team, {}).items()}

            # Ban rates in window
            ban_rates = {
                m: round(cnt / total_matches, 3) for m, cnt in raw_bans.items() if total_matches > 0
            }

            # Identify permaban (>= 80% rate and >= min_sample_size)
            permaban_map = None
            permaban_rate = None
            top_ban_map = None
            top_ban_rate = None
            insufficient_sample = False
            status_note = "No veto data"

            if ban_rates:
                sorted_bans = sorted(ban_rates.items(), key=lambda x: x[1], reverse=True)
                top_ban_map, top_ban_rate = sorted_bans[0]
                top_ban_count = raw_bans.get(top_ban_map, 0)

                if total_matches >= self.min_sample_size and top_ban_rate >= self.permaban_threshold:
                    permaban_map = top_ban_map
                    permaban_rate = top_ban_rate
                    status_note = f"🔒 Permaban: {permaban_map} ({permaban_rate*100:.0f}%, {window_label})"
                elif total_matches < self.min_sample_size and top_ban_rate >= self.permaban_threshold:
                    insufficient_sample = True
                    status_note = (
                        f"Top Ban: {top_ban_map} ({top_ban_rate*100:.0f}%) • "
                        f"Sample size too small ({top_ban_count}/{total_matches} series in {window_label}, min {self.min_sample_size} required)"
                    )
                elif total_matches < self.min_sample_size:
                    insufficient_sample = True
                    status_note = (
                        f"Top Ban: {top_ban_map} ({top_ban_rate*100:.0f}%) • "
                        f"Sample size too small ({total_matches} series in {window_label}, min {self.min_sample_size} required)"
                    )
                else:
                    status_note = f"Top Ban: {top_ban_map} ({top_ban_rate*100:.0f}%) • No {int(self.permaban_threshold*100)}% permaban ({window_label})"
            elif total_matches > 0:
                status_note = f"No first-cycle bans recorded ({total_matches} series in {window_label})"

            # Picks & comfort pick
            raw_picks = {m: int(cnt) for m, cnt in pick_counts.get(team, {}).items()}
            pick_rates = {
                m: round(cnt / total_matches, 3) for m, cnt in raw_picks.items() if total_matches > 0
            }
            comfort_pick = None
            comfort_pick_rate = None
            if pick_rates:
                sorted_picks = sorted(pick_rates.items(), key=lambda x: x[1], reverse=True)
                comfort_pick, comfort_pick_rate = sorted_picks[0]

            self.profiles[team] = TeamVetoProfile(
                team=team,
                total_series=total_matches,
                first_cycle_bans=raw_bans,
                first_cycle_ban_rates=ban_rates,
                permaban=permaban_map,
                permaban_rate=permaban_rate,
                top_ban=top_ban_map,
                top_ban_rate=top_ban_rate,
                insufficient_sample=insufficient_sample,
                status_note=status_note,
                picks=raw_picks,
                pick_rates=pick_rates,
                comfort_pick=comfort_pick,
                comfort_pick_rate=comfort_pick_rate,
            )

    def get_profile(self, team: str) -> Optional[TeamVetoProfile]:
        """Returns the veto profile for a specific team."""
        return self.profiles.get(team)

    def is_permaban(self, team: str, map_name: str) -> bool:
        """Checks if a given map is a confirmed permaban for the team."""
        prof = self.get_profile(team)
        if not prof or not prof.permaban:
            return False
        return prof.permaban.lower() == map_name.lower()

    def get_permabans(self) -> Dict[str, Tuple[str, float]]:
        """Returns all teams that have a detected permaban: {team: (map, rate)}."""
        result = {}
        for team, prof in self.profiles.items():
            if prof.permaban and prof.permaban_rate is not None:
                result[team] = (prof.permaban, prof.permaban_rate)
        return result

    def generate_picklist(
        self,
        focus_team: str,
        opponent_team: str,
        model,
        map_pool: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Generates a 7-map strategic picklist for focus_team against opponent_team.
        Maps are ranked from highest to lowest Power Score difference (focus - opponent).
        If focus_team has a permaban, that map is pinned to the very bottom (Rank 7)
        regardless of score difference.
        """
        from src.power_score import ACTIVE_MAP_POOL
        pool = list(map_pool or ACTIVE_MAP_POOL)
        prof = self.get_profile(focus_team)
        permaban_map = prof.permaban if prof else None

        items = []
        for m in pool:
            ps_focus = model.get_rating(focus_team, m).power_score
            ps_opp = model.get_rating(opponent_team, m).power_score
            diff = round(ps_focus - ps_opp, 1)
            win_prob = round(model.predict_map_win_probability(focus_team, opponent_team, m) * 100, 1)
            is_pban = (permaban_map is not None and m.lower() == permaban_map.lower())

            items.append({
                "map": m,
                "focus_score": round(ps_focus, 1),
                "opponent_score": round(ps_opp, 1),
                "diff": diff,
                "win_prob": win_prob,
                "is_permaban": is_pban,
            })

        # Separate permaban map from non-permaban maps
        regular_maps = [item for item in items if not item["is_permaban"]]
        pban_maps = [item for item in items if item["is_permaban"]]

        # Sort regular maps by diff descending (most positive to least positive)
        regular_maps.sort(key=lambda x: x["diff"], reverse=True)

        # Combine: regular maps followed by permaban at the very bottom
        combined = regular_maps + pban_maps

        # Assign ranks and strategic notes
        for idx, item in enumerate(combined, start=1):
            item["rank"] = idx
            if item["is_permaban"]:
                rate_pct = round(prof.permaban_rate * 100) if (prof and prof.permaban_rate) else 0
                item["note"] = f"🔒 Permaban ({rate_pct}%, past {self.window_days}d) — Never Pick"
            elif idx == 1:
                item["note"] = "⭐ Primary Pick Recommendation"
            elif item["diff"] > 5.0:
                item["note"] = "✅ Strong Advantage"
            elif item["diff"] > 0:
                item["note"] = "👍 Slight Advantage"
            elif item["diff"] > -5.0:
                item["note"] = "⚠️ Slight Disadvantage"
            else:
                item["note"] = "❌ Significant Disadvantage"

        return combined
